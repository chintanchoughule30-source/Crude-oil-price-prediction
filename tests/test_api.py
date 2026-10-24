import math
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from backend import main
from backend.pipeline.features import MODEL_FEATURES


class RecordingModel:
    def __init__(self, prediction):
        self.prediction = prediction
        self.features = None

    def predict(self, features):
        self.features = features
        return [self.prediction]


def valid_payload():
    return {
        "price_lag_1": 78.5,
        "price_lag_2": 78.1,
        "price_lag_5": 76.9,
        "price_lag_20": 74.2,
        "sma_5": 77.8,
        "sma_20": 75.1,
        "return_1d": 0.005,
        "volatility_20": 0.012,
    }


class PredictionRequestTests(unittest.TestCase):
    def test_rejects_negative_prices(self):
        payload = valid_payload()
        payload["price_lag_1"] = -0.01

        with self.assertRaises(ValidationError):
            main.PredictionRequest.model_validate(payload)

    def test_rejects_non_finite_values(self):
        payload = valid_payload()
        payload["sma_5"] = math.nan

        with self.assertRaises(ValidationError):
            main.PredictionRequest.model_validate(payload)

    def test_rejects_unexpected_features(self):
        payload = valid_payload()
        payload["unrecognized"] = 1

        with self.assertRaises(ValidationError):
            main.PredictionRequest.model_validate(payload)


class PredictionEndpointTests(unittest.TestCase):
    def test_prediction_uses_canonical_feature_order(self):
        recorder = RecordingModel(81.25)
        request = main.PredictionRequest.model_validate(valid_payload())

        with patch.object(main, "model", recorder):
            response = main.predict_price(request)

        self.assertEqual(recorder.features, [[valid_payload()[key] for key in MODEL_FEATURES]])
        self.assertEqual(response.predicted_price, 81.25)
        self.assertEqual(response.currency, "USD")
        self.assertEqual(response.unit, "Barrel")

    def test_unavailable_model_returns_service_unavailable(self):
        request = main.PredictionRequest.model_validate(valid_payload())

        with patch.object(main, "model", None):
            with self.assertRaises(HTTPException) as raised:
                main.predict_price(request)

        self.assertEqual(raised.exception.status_code, 503)

    def test_non_finite_model_output_is_rejected(self):
        recorder = RecordingModel(float("nan"))
        request = main.PredictionRequest.model_validate(valid_payload())

        with patch.object(main, "model", recorder):
            with self.assertRaises(HTTPException) as raised:
                main.predict_price(request)

        self.assertEqual(raised.exception.status_code, 500)

    def test_health_reports_prediction_readiness(self):
        with patch.object(main, "model", None):
            self.assertEqual(main.health_check(), {
                "status": "degraded",
                "model_loaded": False,
                "ready": False,
            })

        with patch.object(main, "model", RecordingModel(1.0)):
            self.assertEqual(main.health_check(), {
                "status": "ready",
                "model_loaded": True,
                "ready": True,
            })

    def test_readiness_returns_service_unavailable_without_model(self):
        with patch.object(main, "model", None):
            with self.assertRaises(HTTPException) as raised:
                main.readiness_check()

        self.assertEqual(raised.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
