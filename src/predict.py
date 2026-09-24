# -- coding: utf-8 --
import logging
import joblib
import pandas as pd

from src.preprocessing import parse_order_dates
from src.features import load_state_encoder, engineer_features, align_to_feature_list

logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """يحمّل الموديل المدرب والمحفوظ من نوتبوك 6 — بدون إعادة تدريب."""
    model = joblib.load(model_path)
    logger.info("تم تحميل الموديل من %s", model_path)
    return model

class DeliveryPredictor:
    """
    يجمع الموديل + الـ encoder + قائمة الفيتشرز بمكان واحد،
    عشان نحملهم مرة وحدة بس وقت تشغيل الخدمة، مش مع كل طلب.
    """

    def __init__(self, config: dict):
    
        self.model_version = config["model"]["version"]
        self.model = load_model(config["model"]["path"])
        self.encoder = load_state_encoder(config["model"]["encoder_path"])
        self.feature_list_path = config["model"]["feature_list_path"]

    def predict(self, order: dict) -> dict:
        """
        ياخذ طلب واحد جديد (dictionary فيه نفس أعمدة ml_table)
        ويرجع: prediction (0/1)، probability، model_version.
        """
        df = pd.DataFrame([order])

        df = parse_order_dates(df)
        df = engineer_features(df, self.encoder)
        df = align_to_feature_list(df, self.feature_list_path)

        prediction = int(self.model.predict(df)[0])
        probability = float(self.model.predict_proba(df)[0][1])

        logger.info(
            "توقع جديد: is_late=%d, probability=%.4f, model_version=%s",
            prediction, probability, self.model_version,
        )

        return {
            "is_late": prediction,
            "probability": round(probability, 4),
            "model_version": self.model_version,
        }
