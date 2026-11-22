import asyncio
import threading

import numpy as np

from ._config import get_settings
from ._utils import wrap_embedding_func_with_attrs


_bge_m3_model = None
_model_load_lock = threading.Lock()
_encode_lock = threading.Lock()


def get_bge_m3_model():
    global _bge_m3_model

    if _bge_m3_model is None:
        with _model_load_lock:
            if _bge_m3_model is None:
                settings = get_settings()
                import torch
                from sentence_transformers import SentenceTransformer

                if (
                    settings.embedding_device.startswith("cuda")
                    and not torch.cuda.is_available()
                ):
                    raise RuntimeError(
                        "EMBEDDING_DEVICE is cuda, but CUDA is not available. "
                        "Install a CUDA-enabled PyTorch or set "
                        "EMBEDDING_DEVICE=cpu."
                    )

                _bge_m3_model = SentenceTransformer(
                    str(settings.bge_m3_model_path),
                    device=settings.embedding_device,
                )

    return _bge_m3_model


@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
async def bge_m3_embedding(texts: list[str]) -> np.ndarray:
    model = get_bge_m3_model()

    def encode():
        with _encode_lock:
            return model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

    return np.asarray(await asyncio.to_thread(encode))
