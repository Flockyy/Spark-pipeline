# File: pipeline_pandas.py
"""
optimized_partie2.py

Usage:
    python optimized_partie2.py               # uses settings.yaml
    python optimized_partie2.py --in-dir /path/to/input --out-dir /path/to/out

Notes:
 - Expects settings.yaml with keys: input_dir, output_dir, db_path, csv_sep, csv_encoding, csv_float_format
 - Designed to be runnable outside a Jupyter notebook (no IPython.display calls).
"""

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import yaml


DEFAULT_SETTINGS = {
    "input_dir": "./data/input",
    "output_dir": "./data/out",
    "db_path": "./data/sales_db.db",
    "csv_sep": ";",
    "csv_encoding": "utf-8",
    "csv_float_format": "%.2f",
}


def load_settings(path: str = "settings.yaml") -> dict:
    p = Path(path)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    # overlay defaults
    merged = {**DEFAULT_SETTINGS, **cfg}
    return merged


def read_customers(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] customers file missing: {path}")
        return pd.DataFrame(columns=["customer_id", "city", "is_active"])
    df = pd.read_csv(path, dtype={"customer_id": "string", "city": "string"}, encoding="utf-8")
    # normalize boolean-ish column
    def _to_bool(v):
        if isinstance(v, bool):
            return v
        if pd.isna(v):
            return False
        s = str(v).strip().lower()
        if s in {"1", "true", "yes", "y", "t"}:
            return True
        if s in {"0", "false", "no", "n", "f"}:
            return False
        # fallback: try numeric
        try:
            return bool(int(s))
        except Exception:
            return False
    if "is_active" in df.columns:
        df["is_active"] = df["is_active"].apply(_to_bool)
    else:
        df["is_active"] = False
    # enforce dtypes
    df = df.astype({"customer_id": "string", "city": "string", "is_active": "boolean"})
    return df


def read_refunds(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] refunds file missing: {path}")
        return pd.DataFrame(columns=["order_id", "amount", "created_at"])
    df = pd.read_csv(path, dtype={"order_id": "string"}, encoding="utf-8")
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    else:
        df["amount"] = 0.0
    if "created_at" in df.columns:
        df["created_at"] = df["created_at"].astype("string")
    return df


def read_orders_for_month(input_dir: Path, year: int, month: int) -> pd.DataFrame:
    """
    Read day-by-day JSON order files named like orders_YYYY-MM-DD.json.
    Returns concatenated DataFrame of all found files.
    """
    records: List[pd.DataFrame] = []
    days_in_month = pd.Period(f"{year}-{month:02d}").days_in_month
    for day in range(1, days_in_month + 1):
        fname = input_dir / f"orders_{year}-{month:02d}-{day:02d}.json"
        if not fname.exists():
            # be verbose but not fatal
            print(f"[INFO] missing order file {fname} (skipped)")
            continue
        try:
            df = pd.read_json(fname, convert_dates=False)
            records.append(df)
        except ValueError as e:
            # if file is not valid JSON or empty, skip with warning
            print(f"[WARN] failed to read {fname}: {e}")
            continue
    if not records:
        return pd.DataFrame()
    orders = pd.concat(records, ignore_index=True)
    return orders


def explode_items_and_normalize(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Explodes the 'items' column, flattens JSON into prefixed item_* columns,
    and concatenates with order-level columns.
    """
    if orders.empty:
        return orders
    # keep order-level columns except 'items'
    order_cols = [c for c in orders.columns if c != "items"]
    orders = orders.copy()
    orders = orders.explode("items", ignore_index=True)
    # Some rows may have None in 'items'
    items = pd.json_normalize(orders["items"].dropna()).add_prefix("item_")
    # align index for concat: items corresponds to rows where items not null; create full items_frame with NaNs
    items_full = pd.DataFrame(index=orders.index, columns=items.columns)
    items_full.loc[items.index, items.columns] = items.values
    result = pd.concat([orders[order_cols].reset_index(drop=True), items_full.reset_index(drop=True)], axis=1)
    # normalize types for item_qty and item_unit_price
    for col in ("item_qty", "item_unit_price"):
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
    return result


def detect_and_save_rejects(df: pd.DataFrame, out_dir: Path, enc: str) -> pd.DataFrame:
    """
    Detect negative unit price items and save them to rejects_items.csv, return df without rejects.
    """
    if "item_unit_price" not in df.columns:
        return df
    neg_mask = df["item_unit_price"] < 0
    n_neg = int(neg_mask.sum())
    print(f"[INFO] negative price lines: {n_neg}")
    if n_neg > 0:
        rejects = df.loc[neg_mask].copy()
        rejects_path = out_dir / "rejects_items.csv"
        rejects.to_csv(rejects_path, index=False, encoding=enc)
        print(f"[INFO] rejects saved to {rejects_path}")
        df = df.loc[~neg_mask].copy()
    return df


def aggregate_orders(per_item: pd.DataFrame, customers: pd.DataFrame, refunds: pd.DataFrame) -> pd.DataFrame:
    """
    Build per_order and then aggregate per (date, city, channel).
    """
    if per_item.empty:
        return pd.DataFrame()
    # compute line_gross and per-order aggregates
    per_item["line_gross"] = per_item["item_qty"] * per_item["item_unit_price"]
    # choose created_at present in per_item for grouping - keep first created_at per order
    per_order = (
        per_item
        .groupby(["order_id", "customer_id", "channel", "created_at"], as_index=False)
        .agg(items_sold=("item_qty", "sum"), gross_revenue_eur=("line_gross", "sum"))
    )
    # join customer info and keep active customers only
    cust_subset = customers[["customer_id", "city", "is_active"]].drop_duplicates(subset=["customer_id"])
    per_order = per_order.merge(cust_subset, on="customer_id", how="left")
    per_order = per_order[per_order["is_active"].fillna(False)].copy()
    # parse created_at into a date string ISO
    def to_iso_date(s):
        if pd.isna(s):
            return None
        s2 = str(s)
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(s2, fmt).date().isoformat()
            except Exception:
                continue
        # fallback: try pandas
        try:
            return pd.to_datetime(s2, errors="coerce").date().isoformat()
        except Exception:
            return None
    per_order["order_date"] = per_order["created_at"].apply(to_iso_date)
    # refunds sum per order
    if not refunds.empty:
        refunds_sum = refunds.groupby("order_id", as_index=False)["amount"].sum().rename(columns={"amount": "refunds_eur"})
    else:
        refunds_sum = pd.DataFrame(columns=["order_id", "refunds_eur"])
    per_order = per_order.merge(refunds_sum, on="order_id", how="left").fillna({"refunds_eur": 0.0})
    # save clean orders to DB later; here we compute aggregated metrics
    agg = (
        per_order
        .groupby(["order_date", "city", "channel"], as_index=False)
        .agg(
            orders_count=("order_id", "nunique"),
            unique_customers=("customer_id", "nunique"),
            items_sold=("items_sold", "sum"),
            gross_revenue_eur=("gross_revenue_eur", "sum"),
            refunds_eur=("refunds_eur", "sum"),
        )
    )
    # net revenue should be gross - refunds
    agg["net_revenue_eur"] = agg["gross_revenue_eur"] - agg["refunds_eur"]
    # normalize column name to `date`
    agg = agg.rename(columns={"order_date": "date"}).sort_values(["date", "city", "channel"]).reset_index(drop=True)
    return per_order, agg


def save_to_sqlite(df: pd.DataFrame, db_path: Path, table_name: str):
    if df.empty:
        print(f"[INFO] no data to write for {table_name}")
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"[INFO] wrote table `{table_name}` to SQLite DB at {db_path}")


def export_csvs(agg: pd.DataFrame, out_dir: Path, sep: str, enc: str, ffmt: str):
    if agg.empty:
        print("[INFO] no aggregate to export")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for d, sub in agg.groupby("date"):
        # d may be None if parsing failed; skip
        if pd.isna(d) or d is None:
            continue
        # ensure date string has only digits
        safe_date = str(d).replace("-", "")
        out_path = out_dir / f"daily_summary_{safe_date}.csv"
        sub[[
            "date",
            "city",
            "channel",
            "orders_count",
            "unique_customers",
            "items_sold",
            "gross_revenue_eur",
            "refunds_eur",
            "net_revenue_eur",
        ]].to_csv(out_path, index=False, sep=sep, encoding=enc, float_format=ffmt)
        print(f"[INFO] exported {out_path}")
    all_path = out_dir / "daily_summary_all.csv"
    agg.to_csv(all_path, index=False, sep=sep, encoding=enc, float_format=ffmt)
    print(f"[INFO] exported {all_path}")


def main(args):
    cfg = load_settings(args.settings)
    # CLI overrides
    in_dir = Path(args.in_dir or cfg.get("input_dir"))
    out_dir = Path(args.out_dir or cfg.get("output_dir"))
    db_path = Path(args.db_path or cfg.get("db_path"))
    sep = args.csv_sep or cfg.get("csv_sep", ";")
    enc = args.csv_encoding or cfg.get("csv_encoding", "utf-8")
    ffmt = args.csv_float_format or cfg.get("csv_float_format", "%.2f")

    print(f"[INFO] Input dir: {in_dir}")
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] DB path: {db_path}")

    # read supporting files
    customers = read_customers(in_dir / "customers.csv")
    refunds = read_refunds(in_dir / "refunds.csv")

    # read orders for given month/year (default March 2025)
    year = args.year or 2025
    month = args.month or 3
    orders = read_orders_for_month(in_dir, year, month)
    print(f"[INFO] orders shape (raw concat): {orders.shape}")

    if orders.empty:
        print("[WARN] no orders found for the period. Exiting.")
        return

    # keep only paid orders
    orders_paid = orders[orders.get("payment_status") == "paid"].copy()
    print(f"[INFO] filtered paid orders: {len(orders)} -> {len(orders_paid)}")

    # explode items, normalize
    orders_items = explode_items_and_normalize(orders_paid)
    # rename normalized item columns to match original script expectations
    # there was usage of item_qty and item_unit_price
    # try to coerce common names
    if "item_quantity" in orders_items.columns and "item_qty" not in orders_items.columns:
        orders_items = orders_items.rename(columns={"item_quantity": "item_qty"})
    if "item_price" in orders_items.columns and "item_unit_price" not in orders_items.columns:
        orders_items = orders_items.rename(columns={"item_price": "item_unit_price"})

    # detect negative prices and save rejects
    orders_items = detect_and_save_rejects(orders_items, out_dir, enc)

    # deduplicate: keep first row per order_id (by created_at)
    if {"order_id", "created_at"}.issubset(orders_items.columns):
        before = len(orders_items)
        orders_items = orders_items.sort_values(["order_id", "created_at"]).drop_duplicates(subset=["order_id"], keep="first")
        after = len(orders_items)
        print(f"[INFO] deduplication: {before} -> {after}")
    else:
        print("[INFO] cannot deduplicate by order_id/created_at (columns missing)")

    # aggregate to per_order and to daily city sales
    per_order, agg = aggregate_orders(orders_items, customers, refunds)

    # save clean orders subset to sqlite (matching original selected columns)
    if not per_order.empty:
        per_order_save = per_order[[
            "order_id",
            "customer_id",
            "city",
            "channel",
            "order_date",
            "items_sold",
            "gross_revenue_eur",
        ]].copy()
        save_to_sqlite(per_order_save, db_path, "orders_clean")

    # save aggregate to sqlite and export CSVs
    save_to_sqlite(agg, db_path, "daily_city_sales")
    export_csvs(agg, out_dir, sep, enc, ffmt)

    print("[INFO] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and aggregate orders for a month.")
    parser.add_argument("--settings", default="settings.yaml", help="path to settings yaml")
    parser.add_argument("--in-dir", help="input directory overriding settings")
    parser.add_argument("--out-dir", help="output directory overriding settings")
    parser.add_argument("--db-path", help="sqlite db path overriding settings")
    parser.add_argument("--csv-sep", help="CSV separator")
    parser.add_argument("--csv-encoding", help="CSV encoding")
    parser.add_argument("--csv-float-format", help="float format for CSVs")
    parser.add_argument("--year", type=int, help="year to process (default 2025)")
    parser.add_argument("--month", type=int, help="month to process (default 3)")
    args = parser.parse_args()
    main(args)
