# -*- coding: utf-8 -*-
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def read_raw_tables(data_dir: str) -> dict:
    """يقرأ ملفات الـ CSV الخام من مجلد البيانات."""
    tables = {
        "orders": pd.read_csv(f"{data_dir}/olist_orders_dataset.csv"),
        "order_items": pd.read_csv(f"{data_dir}/olist_order_items_dataset.csv"),
        "payments": pd.read_csv(f"{data_dir}/olist_order_payments_dataset.csv"),
        "customers": pd.read_csv(f"{data_dir}/olist_customers_dataset.csv"),
    }
    logger.info("تم تحميل %d جداول من %s", len(tables), data_dir)
    return tables


def build_ml_table(orders, order_items, payments, customers) -> pd.DataFrame:
    """يطابق تمامًا دمج نوتبوك 1: تجميع القطع والدفعات، ثم الدمج مع orders و customers."""
    items_agg = order_items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
    ).reset_index()

    payments_agg = payments.groupby("order_id").agg(
        total_payment_value=("payment_value", "sum"),
        n_payment_methods=("payment_type", "nunique"),
    ).reset_index()

    ml_table = orders.merge(items_agg, on="order_id", how="left")
    ml_table = ml_table.merge(payments_agg, on="order_id", how="left")
    ml_table = ml_table.merge(customers, on="customer_id", how="left")

    logger.info("ml_table جاهز بشكل: %s", ml_table.shape)
    return ml_table


def parse_order_dates(df: pd.DataFrame) -> pd.DataFrame:
    """يحوّل أعمدة التواريخ النصية لـ datetime (نفس نوتبوك 2)."""
    date_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    df = df.copy()
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def create_labels(ml_table: pd.DataFrame) -> pd.DataFrame:
    """يطابق نوتبوك 2: يستبعد الطلبات بدون تسليم فعلي، ويحسب is_late."""
    labeled = ml_table[ml_table["order_delivered_customer_date"].notna()].copy()
    labeled["is_late"] = (
        labeled["order_delivered_customer_date"] > labeled["order_estimated_delivery_date"]
    ).astype(int)
    logger.info("عدد الصفوف بعد الوسم: %d", len(labeled))
    return labeled

