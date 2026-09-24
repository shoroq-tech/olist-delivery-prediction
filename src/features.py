# -*- coding: utf-8 -*-
import logging
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NUMERIC_TO_LOG = ["n_items", "total_price", "total_freight", "total_payment_value"]

COLS_TO_DROP = [
    "order_status", "customer_city", "customer_state",
    "n_items", "total_price", "total_freight", "total_payment_value",
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date", "delivery_days",
    "purchase_year", "n_payment_methods", "customer_zip_code_prefix",
    "customer_id", "customer_unique_id", "order_id",
]


def load_state_encoder(encoder_path: str):
    """يحمّل الـ OneHotEncoder المدرب مسبقًا (بدون إعادة تدريب)."""
    encoder = joblib.load(encoder_path)
    logger.info("تم تحميل state_encoder من %s", encoder_path)
    return encoder


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """يطابق نوتبوك 5: estimated_days, purchase_month/year, is_strike_period, is_black_friday_season."""
    df = df.copy()
    df["estimated_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days

    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_year"] = df["order_purchase_timestamp"].dt.year

    df["is_strike_period"] = (
        (df["purchase_year"] == 2018) & (df["purchase_month"].isin([2, 3]))
    ).astype(int)

    df["is_black_friday_season"] = (df["purchase_month"] == 11).astype(int)
    return df


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """يطابق نوتبوك 5: log1p على الأعمدة الرقمية."""
    df = df.copy()
    for col in NUMERIC_TO_LOG:
        df[f"{col}_log"] = np.log1p(df[col])
    return df


def add_multi_item_flag(df: pd.DataFrame) -> pd.DataFrame:
    """يطابق نوتبوك 5: is_multi_item = عدد القطع أكبر من 1."""
    df = df.copy()
    df["is_multi_item"] = (df["n_items"] > 1).astype(int)
    return df


def encode_customer_state(df: pd.DataFrame, encoder) -> pd.DataFrame:
    """يطبّق الـ encoder المحفوظ (transform فقط) على customer_state."""
    state_cols = encoder.get_feature_names_out(["customer_state"])
    encoded = pd.DataFrame(
        encoder.transform(df[["customer_state"]]),
        columns=state_cols,
        index=df.index,
    )
    return pd.concat([df, encoded], axis=1)


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """يطابق نوتبوك 5: حذف الأعمدة الأصلية غير المستخدمة بالتدريب."""
    return df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])


def engineer_features(df: pd.DataFrame, encoder) -> pd.DataFrame:
    """يشغّل كل خطوات الفيتشر إنجنيرينغ بالترتيب الصحيح (نوتبوك 5 كامل)."""
    df = add_date_features(df)
    df = add_log_features(df)
    df = add_multi_item_flag(df)
    df = encode_customer_state(df, encoder)
    df = drop_unused_columns(df)
    logger.info("الفيتشرز جاهزة بشكل: %s", df.shape)
    return df


def align_to_feature_list(df: pd.DataFrame, feature_list_path: str) -> pd.DataFrame:
    """يرتّب/يكمل الأعمدة بالضبط متل feature_list.txt (يملأ الناقص بـ 0)."""
    with open(feature_list_path, "r", encoding="utf-8") as f:
        feature_list = [line.strip() for line in f if line.strip()]
    return df.reindex(columns=feature_list, fill_value=0)


