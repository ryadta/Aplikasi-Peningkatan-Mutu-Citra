from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

MODEL_PATH = Path("models/leaf_classifier.keras")
LABELS_PATH = Path("models/labels.json")
DESCRIPTIONS_PATH = Path("models/leaf_descriptions.json")
IMAGE_SIZE = (224, 224)


class ClassifierNotReadyError(Exception):
    """Raised when classifier artifacts are not available yet."""


@dataclass
class ClassificationResult:
    class_name: str
    confidence: float
    confidence_percent: float
    description: str


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if isinstance(data, dict):
        return data
    return {}


@st.cache_resource(show_spinner=False)
def _load_model():
    try:
        from tensorflow import keras
    except Exception as exc:  # pragma: no cover
        raise ClassifierNotReadyError(
            "TensorFlow belum tersedia. Install dependency terlebih dahulu."
        ) from exc

    if not MODEL_PATH.exists():
        raise ClassifierNotReadyError(
            "Model belum ditemukan. Jalankan training dulu dengan train_classifier.py."
        )

    try:
        model = keras.models.load_model(MODEL_PATH)
    except Exception as exc:
        raise ClassifierNotReadyError(
            f"Gagal memuat model dari '{MODEL_PATH}'. Cek artefak model."
        ) from exc
    return model


@st.cache_data(show_spinner=False)
def _load_labels() -> list[str]:
    if not LABELS_PATH.exists():
        raise ClassifierNotReadyError(
            "File labels.json belum ada. Jalankan training untuk membuat metadata label."
        )
    with LABELS_PATH.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    if isinstance(data, list):
        labels = [str(item) for item in data]
    elif isinstance(data, dict):
        labels = [str(v) for _, v in sorted(data.items(), key=lambda kv: int(kv[0]))]
    else:
        labels = []

    if not labels:
        raise ClassifierNotReadyError("labels.json kosong atau format tidak valid.")
    return labels


@st.cache_data(show_spinner=False)
def _load_descriptions() -> dict:
    return _load_json(DESCRIPTIONS_PATH)


def _preprocess(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None or image_bgr.ndim != 3:
        raise ValueError("Input image harus berupa citra warna (BGR).")

    resized = cv2.resize(image_bgr, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    batch = np.expand_dims(rgb, axis=0)
    return batch


def classify_image_top1(image_bgr: np.ndarray) -> ClassificationResult:
    model = _load_model()
    labels = _load_labels()
    descriptions = _load_descriptions()

    batch = _preprocess(image_bgr)
    probabilities = model.predict(batch, verbose=0)[0]
    if len(probabilities) != len(labels):
        raise ClassifierNotReadyError(
            "Jumlah output model tidak cocok dengan jumlah label. Jalankan training ulang."
        )

    top_index = int(np.argmax(probabilities))
    confidence = float(probabilities[top_index])
    class_name = labels[top_index]
    description = descriptions.get(class_name, "Deskripsi belum tersedia.")

    return ClassificationResult(
        class_name=class_name,
        confidence=confidence,
        confidence_percent=confidence * 100.0,
        description=description,
    )
