# File: tests/test_pipeline.py
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType
from pyspark.sql.functions import col

from pipeline_pyspark import explode_items, deduplicate_keep_first, compute_per_order_and_agg, read_customers

@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder.master("local[2]").appName("test").getOrCreate()
    yield spark
    spark.stop()


def test_explode_items(spark):
    """Test that explode_items correctly explodes nested item arrays."""
    items_schema = ArrayType(
        StructType([
            StructField("qty", DoubleType(), True),
            StructField("unit_price", DoubleType(), True)
        ])
    )
    schema = StructType([
        StructField("order_id", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("channel", StringType(), True),
        StructField("created_at", StringType(), True),
        StructField("items", items_schema, True)
    ])
    data = [
        {"order_id": "o1", "customer_id": "c1", "channel": "web",
         "created_at": "2025-03-01 10:00:00", "items": [{"qty": 2.0, "unit_price": 5.0}]},
        {"order_id": "o2", "customer_id": "c2", "channel": "web",
         "created_at": "2025-03-02 11:00:00", "items": [{"qty": 1.0, "unit_price": 10.0}, {"qty": 1.0, "unit_price": -1.0}]},
    ]
    df = spark.createDataFrame(data, schema=schema)

    exploded = explode_items(df)

    # There should be 3 rows after exploding
    assert exploded.count() == 3
    # Columns should exist
    assert "item_qty" in exploded.columns
    assert "item_unit_price" in exploded.columns
    # Specific check for order 'o2'
    o2_rows = exploded.filter(col("order_id") == "o2").collect()
    assert len(o2_rows) == 2
    values = [row.item_unit_price for row in o2_rows]
    assert 10.0 in values and -1.0 in values


def test_deduplicate_keep_first(spark):
    """Test that deduplicate_keep_first keeps only the first record per order_id."""
    data = [
        {"order_id": "o1", "customer_id": "c1", "channel": "web", "created_at": "2025-03-01 10:00:00", "item_qty": 2.0, "item_unit_price": 5.0},
        {"order_id": "o1", "customer_id": "c1", "channel": "web", "created_at": "2025-03-01 12:00:00", "item_qty": 1.0, "item_unit_price": 3.0},
        {"order_id": "o2", "customer_id": "c2", "channel": "web", "created_at": "2025-03-02 11:00:00", "item_qty": 1.0, "item_unit_price": 10.0},
    ]
    schema = StructType([
        StructField("order_id", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("channel", StringType(), True),
        StructField("created_at", StringType(), True),
        StructField("item_qty", DoubleType(), True),
        StructField("item_unit_price", DoubleType(), True),
    ])
    df = spark.createDataFrame(data, schema=schema)

    deduped = deduplicate_keep_first(df)

    # Each order_id should appear only once
    assert deduped.count() == 2
    o1_row = deduped.filter(col("order_id") == "o1").collect()[0]
    # Should keep the earliest created_at
    assert o1_row.created_at == "2025-03-01 10:00:00"
