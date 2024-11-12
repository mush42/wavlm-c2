import os
from pathlib import Path

import numpy as np
from transformers import AutoModel
from ctranslate2 import Generator as C2Encoder

import wavlm2c2


HERE = Path.cwd()
TRANSFORMERS_WLM_PATH = os.fspath(HERE / "microsoft-wavlm-large")
C2_WLM_PATH = os.fspath(HERE / "wavlm-c2")


def load_transformers_model(model_path: os.PathLike=TRANSFORMERS_WLM_PATH) -> AutoModel:
    return AutoModel.from_pretrained(model_path)


def load_c2_model(model_path: os.PathLike=C2_WLM_PATH) -> C2Encoder:
    return C2Encoder(model_path)


def infer_transformers(model: AutoModel, x: np.ndarray) -> np.ndarray:
    return model(x)


def infer_c2(model: C2Encoder, x: np.ndarray) -> np.ndarray:
    return model.forward_batch(x)


if __name__ == '__main__':
    transformers_model = load_transformers_model()
    c2_model = load_c2_model()
