# File: pipeline_pyspark.py
"""
Modular PySpark pipeline with logging, metrics, and unit tests.

Files provided in this document (copy each to its own file):
 - pipeline_pyspark.py    <-- main module
 - tests/test_pipeline.py <-- pytest unit tests (requires local Spark)

Features:
 - Modular functions for each step
 - Structured logging
 - Metrics abstraction that uses prometheus_client when available, otherwise in-memory
 - Safe handling of missing files
 - Unit tests for core transformations (run with pytest)

Notes:
 - The tests create a local SparkSession; ensure pyspark is available in your test environment.
 - To run the pipeline in production, run:
     python pipeline_pyspark.py --settings settings.yaml

"""

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import coalesce, col, countDistinct, explode, lit
from pyspark.sql.functions import sum as _sum
from pyspark.sql.functions import to_date, when
from pyspark.sql.types import DoubleType, StringType

# ---------------------------
# Logging
# ---------------------------
logger = logging.getLogger("pyspark_pipeline")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
)
logger.addHandler(handler)


# ---------------------------
# Metrics abstraction
# ---------------------------
class Metrics:
    """Simple metrics wrapper. Tries to use prometheus_client if present,
    otherwise falls back to an in-memory counter dictionary for tests/CI.
    """

    def __init__(self):
        self._enabled = False
        self._counters = {}
        try:
            from prometheus_client import Counter, Gauge

            self.Counter = Counter
            self.Gauge = Gauge
            self._use_prom = True
            self._prom_registry = {}
            logger.info(
                "Prometheus client is available; metrics will be exported if configured."
            )
        except Exception:
            self._use_prom = False
            logger.info("Prometheus client not available; using in-memory metrics.")

    def counter(self, name: str, description: str = ""):
        if self._use_prom:
            if name not in self._prom_registry:
                self._prom_registry[name] = self.Counter(name, description)
            return self._prom_registry[name]
        else:
            # simple lambda simulating .inc()
            self._counters.setdefault(name, 0)

            class _Ctr:
                def __init__(self, store, key):
                    self._store = store
                    self._key = key

                def inc(self, v=1):
                    self._store[self._key] = self._store.get(self._key, 0) + v

            return _Ctr(self._counters, name)

    def gauge(self, name: str, description: str = ""):
        if self._use_prom:
            if name not in self._prom_registry:
                self._prom_registry[name] = self.Gauge(name, description)
            return self._prom_registry[name]
        else:
            self._counters.setdefault(name, 0)

            class _Gauge:
                def __init__(self, store, key):
                    self._store = store
                    self._key = key

                def set(self, v):
                    self._store[self._key] = v

            return _Gauge(self._counters, name)

    def snapshot(self) -> Dict[str, int]:
        return dict(self._counters)


metrics = Metrics()
# common counters
orders_processed_ctr = metrics.counter(
    "orders_processed_total", "Number of orders processed"
)
orders_rejected_ctr = metrics.counter(
    "orders_rejected_total", "Number of rejected item lines (negative price)"
)


# ---------------------------
# Settings
# ---------------------------
DEFAULT_SETTINGS = {
    "input_dir": "./data/input",
    "output_dir": "./data/out",
    "db_path": "./data/sales_db.db",
    "csv_sep": ";",
    "csv_encoding": "utf-8",
    "csv_float_format": "%.2f",
}


def load_settings(path: str = "settings.yaml") -> Dict:
    p = Path(path)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    merged = {**DEFAULT_SETTINGS, **cfg}
    return merged


# ---------------------------
# Spark utilities
# ---------------------------


def create_spark(
    app_name: str = "OrdersPipeline", master: Optional[str] = None
) -> SparkSession:
    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)
    spark = builder.getOrCreate()
    return spark


# ---------------------------
# I/O readers
# ---------------------------


def read_customers(spark: SparkSession, path: Path) -> DataFrame:
    if not path.exists():
        logger.warning("customers.csv not found: %s", path)
        # return empty DataFrame with expected schema columns
        return spark.createDataFrame(
            [], schema="customer_id string, city string, is_active boolean"
        )
    df = spark.read.option("header", True).csv(str(path))
    # normalize is_active
    df = df.withColumn(
        "is_active",
        when(col("is_active").isin("1", "true", "yes", "y", "t"), True)
        .when(col("is_active").isin("0", "false", "no", "n", "f"), False)
        .otherwise(False),
    )
    df = df.withColumn("customer_id", col("customer_id").cast(StringType()))
    return df


def read_refunds(spark: SparkSession, path: Path) -> DataFrame:
    if not path.exists():
        logger.warning("refunds.csv not found: %s", path)
        return spark.createDataFrame(
            [], schema="order_id string, amount double, created_at string"
        )
    df = spark.read.option("header", True).csv(str(path))
    df = df.withColumn("amount", col("amount").cast(DoubleType()))
    df = df.withColumn("order_id", col("order_id").cast(StringType()))
    return df


def read_orders_for_month(
    spark: SparkSession, input_dir: Path, year: int, month: int
) -> DataFrame:
    from calendar import monthrange

    days = monthrange(year, month)[1]
    dfs = []
    for day in range(1, days + 1):
        fname = input_dir / f"orders_{year}-{month:02d}-{day:02d}.json"
        if not fname.exists():
            logger.debug("missing: %s", fname)
            continue
        try:
            df = spark.read.option("multiLine", True).json(str(fname))
            dfs.append(df)
        except Exception as e:
            logger.warning("failed to read %s: %s", fname, e)
    if not dfs:
        return spark.createDataFrame([], schema=[])  # empty
    # union all with allowMissingColumns
    base = dfs[0]
    for d in dfs[1:]:
        base = base.unionByName(d, allowMissingColumns=True)
    return base


# ---------------------------
# Transformations
# ---------------------------


def explode_items(df: DataFrame) -> DataFrame:
    """
    Explodes 'items' array and flattens common fields, handling missing keys safely.
    """
    if df.rdd.isEmpty():
        return df

    exploded = df.withColumn("item", explode(col("items")))

    # Detect available fields in the struct
    item_fields = exploded.schema["item"].dataType.fieldNames()

    # Determine which columns to use for quantity and unit_price
    qty_cols = [c for c in ["qty", "quantity"] if c in item_fields]
    price_cols = [c for c in ["unit_price", "price"] if c in item_fields]

    # Safe coalesce for missing fields
    item_qty_expr = (
        coalesce(*[col(f"item.{c}") for c in qty_cols]).cast(DoubleType())
        if qty_cols
        else col("item.qty").cast(DoubleType())
    )
    item_price_expr = (
        coalesce(*[col(f"item.{c}") for c in price_cols]).cast(DoubleType())
        if price_cols
        else col("item.unit_price").cast(DoubleType())
    )

    # Flatten
    out = exploded.select(
        "order_id",
        "customer_id",
        "channel",
        "created_at",
        item_qty_expr.alias("item_qty"),
        item_price_expr.alias("item_unit_price"),
    )

    return out


def detect_and_save_rejects(
    df: DataFrame, out_dir: Path, csv_encoding: str = "utf-8"
) -> DataFrame:
    if df.rdd.isEmpty():
        return df
    neg = df.filter(col("item_unit_price") < 0)
    n_neg = neg.count()
    orders_rejected_ctr.inc(n_neg)
    if n_neg > 0:
        out_path = out_dir / "rejects_items.csv"
        neg.write.mode("overwrite").option("header", True).csv(str(out_path))
        logger.info("Saved %d rejected lines to %s", n_neg, out_path)
    df_good = df.filter(col("item_unit_price") >= 0)
    return df_good


def deduplicate_keep_first(df: DataFrame) -> DataFrame:
    from pyspark.sql.functions import row_number
    from pyspark.sql.window import Window

    if df.rdd.isEmpty():
        return df
    if not {"order_id", "created_at"}.issubset(set(df.columns)):
        return df.dropDuplicates(["order_id"]) if "order_id" in df.columns else df
    w = Window.partitionBy("order_id").orderBy("created_at")
    df2 = df.withColumn("rn", row_number().over(w)).filter(col("rn") == 1).drop("rn")
    return df2


def compute_per_order_and_agg(
    items_df: DataFrame, customers_df: DataFrame, refunds_df: DataFrame
) -> Tuple[DataFrame, DataFrame]:
    if items_df.rdd.isEmpty():
        return items_df, items_df
    # line gross
    items_df = items_df.withColumn(
        "line_gross", col("item_qty") * col("item_unit_price")
    )
    per_order = items_df.groupBy(
        "order_id", "customer_id", "channel", "created_at"
    ).agg(
        _sum("item_qty").alias("items_sold"),
        _sum("line_gross").alias("gross_revenue_eur"),
    )
    # join customers and filter active
    if not customers_df.rdd.isEmpty():
        per_order = per_order.join(
            customers_df.select("customer_id", "city", "is_active"),
            "customer_id",
            "left",
        )
        per_order = per_order.filter(col("is_active"))
    # parse order_date
    per_order = per_order.withColumn("order_date", to_date(col("created_at")))
    # refunds
    if not refunds_df.rdd.isEmpty():
        refunds_sum = refunds_df.groupBy("order_id").agg(
            _sum(col("amount")).alias("refunds_eur")
        )
    else:
        refunds_sum = None
    if refunds_sum is not None:
        per_order = per_order.join(refunds_sum, "order_id", "left").na.fill(
            {"refunds_eur": 0.0}
        )
    else:
        per_order = per_order.withColumn("refunds_eur", lit(0.0))
    # aggregate
    agg = per_order.groupBy("order_date", "city", "channel").agg(
        countDistinct("order_id").alias("orders_count"),
        countDistinct("customer_id").alias("unique_customers"),
        _sum("items_sold").alias("items_sold"),
        _sum("gross_revenue_eur").alias("gross_revenue_eur"),
        _sum("refunds_eur").alias("refunds_eur"),
    )
    agg = agg.withColumn(
        "net_revenue_eur", col("gross_revenue_eur") - col("refunds_eur")
    ).withColumnRenamed("order_date", "date")
    return per_order, agg


# ---------------------------
# Writers
# ---------------------------


def write_sqlite(df: DataFrame, db_path: Path, table_name: str):
    if df.rdd.isEmpty():
        logger.info("No rows to write for %s", table_name)
        return
    # collect to pandas then to_sql (because Spark lacks direct sqlite connector in pure python)
    p = db_path
    p.parent.mkdir(parents=True, exist_ok=True)
    pdf = df.toPandas()
    conn = sqlite3.connect(str(p))
    pdf.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    logger.info("Wrote %d rows to %s:%s", len(pdf), db_path, table_name)


def export_daily_csvs(
    agg_df: DataFrame, out_dir: Path, sep: str = ";", ffmt: str = "%.2f"
):
    if agg_df.rdd.isEmpty():
        logger.info("No aggregate to export")
        return
    # create a safe date string and write per-day CSVs
    rows = agg_df.select("date").distinct().collect()
    for r in rows:
        d = r[0]
        if d is None:
            continue
        safe = (
            d.strftime("%Y%m%d") if hasattr(d, "strftime") else str(d).replace("-", "")
        )
        sub = agg_df.filter(col("date") == lit(str(d)))
        sub.coalesce(1).write.mode("overwrite").option("header", True).csv(
            str(out_dir / f"daily_summary_{safe}.csv")
        )
    # write all
    agg_df.coalesce(1).write.mode("overwrite").option("header", True).csv(
        str(out_dir / "daily_summary_all.csv")
    )
    logger.info("Exported daily CSVs to %s", out_dir)


# ---------------------------
# Orchestration main
# ---------------------------


def run_pipeline(
    settings_path: str,
    year: int = 2025,
    month: int = 3,
    spark_master: Optional[str] = None,
):
    cfg = load_settings(settings_path)
    input_dir = (
        Path(cfg["input_dir"])
        if cfg.get("input_dir")
        else Path(DEFAULT_SETTINGS["input_dir"])
    )
    output_dir = (
        Path(cfg["output_dir"])
        if cfg.get("output_dir")
        else Path(DEFAULT_SETTINGS["output_dir"])
    )
    db_path = Path(cfg.get("db_path", DEFAULT_SETTINGS["db_path"]))
    sep = cfg.get("csv_sep", DEFAULT_SETTINGS["csv_sep"])
    enc = cfg.get("csv_encoding", DEFAULT_SETTINGS["csv_encoding"])
    ffmt = cfg.get("csv_float_format", DEFAULT_SETTINGS["csv_float_format"])

    output_dir.mkdir(parents=True, exist_ok=True)

    spark = create_spark(master=spark_master)

    customers = read_customers(spark, input_dir / "customers.csv")
    refunds = read_refunds(spark, input_dir / "refunds.csv")
    orders_raw = read_orders_for_month(spark, input_dir, year, month)
    logger.info("Loaded orders: %s", str(orders_raw.count()))

    # only paid
    orders_paid = orders_raw.filter(col("payment_status") == "paid")
    orders_processed_ctr.inc(orders_paid.count())

    items = explode_items(orders_paid)
    items_clean = detect_and_save_rejects(items, output_dir, csv_encoding=enc)
    items_dedup = deduplicate_keep_first(items_clean)

    per_order, agg = compute_per_order_and_agg(items_dedup, customers, refunds)

    # write outputs
    if not per_order.rdd.isEmpty():
        # convert to pandas subset like original
        per_order_save = per_order.select(
            "order_id",
            "customer_id",
            "city",
            "channel",
            "order_date",
            "items_sold",
            "gross_revenue_eur",
        )
        write_sqlite(per_order_save, db_path, "orders_clean")
    write_sqlite(agg, db_path, "daily_city_sales")
    export_daily_csvs(agg, output_dir, sep=sep, ffmt=ffmt)

    logger.info("Pipeline finished. Metrics snapshot: %s", metrics.snapshot())
    spark.stop()


# ---------------------------
# CLI
# ---------------------------
# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="settings.yaml")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--month", type=int, default=3)
    parser.add_argument("--master", default=None, help="Spark master (e.g. local[*])")
    args = parser.parse_args()
    run_pipeline(args.settings, args.year, args.month, spark_master=args.master)
