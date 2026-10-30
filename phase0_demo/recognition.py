from __future__ import annotations

import sys
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from .domain import MatchResult
from .image_validation import InvalidNoseImage, validate_nose_image

AI_SOURCE = Path(__file__).resolve().parents[1] / "ai-model" / "src"


@lru_cache(maxsize=1)
def _ai_functions():
    """Import TensorFlow-backed helpers only when recognition starts."""
    if str(AI_SOURCE) not in sys.path:
        sys.path.insert(0, str(AI_SOURCE))
    from feature_extractor import extract_features, warm_up
    from preprocessing import preprocess_nose_image
    from similarity import find_most_similar

    return extract_features, warm_up, preprocess_nose_image, find_most_similar


class RecognitionService:
    def __init__(self, store, threshold: float):
        self.store = store
        self.threshold = threshold
        self._lock = threading.Lock()
        self._features = np.empty((0, 2048), dtype=np.float32)
        self._dog_ids: list[str] = []
        self._image_ids: list[str] = []

    def start(self, warm_model: bool = True) -> None:
        if warm_model:
            _, warm_up, _, _ = _ai_functions()
            warm_up()
        self.reload_cache()

    def reload_cache(self) -> None:
        vectors: list[np.ndarray] = []
        dogs: list[str] = []
        images: list[str] = []
        for item in self.store.iter_embeddings():
            profile = self.store.get_profile(str(item["dog_id"]))
            if profile is None:
                continue
            try:
                validate_nose_image(self.store.read_image(profile.image_key))
            except (FileNotFoundError, InvalidNoseImage):
                # Preserve the record for audit/recovery, but do not let a
                # legacy invalid enrollment contaminate the live gallery.
                continue
            vector = np.frombuffer(item["embedding"], dtype=np.float32)
            if vector.size != 2048:
                continue
            vectors.append(vector)
            dogs.append(str(item["dog_id"]))
            images.append(str(item["image_id"]))
        with self._lock:
            self._features = np.vstack(vectors) if vectors else np.empty((0, 2048), dtype=np.float32)
            self._dog_ids = dogs
            self._image_ids = images

    def embed(self, image_bytes: bytes) -> np.ndarray:
        extract_features, _, preprocess_nose_image, _ = _ai_functions()
        processed = preprocess_nose_image(image_bytes)
        return extract_features(processed).reshape(-1).astype(np.float32)

    def match_embedding(self, embedding: np.ndarray) -> MatchResult:
        with self._lock:
            features = self._features.copy()
            dog_ids = list(self._dog_ids)
            image_ids = list(self._image_ids)
        if len(features) == 0:
            return MatchResult(matched=False, score=0.0)
        _, _, _, find_most_similar = _ai_functions()
        dog_id, image_id, score = find_most_similar(
            embedding.reshape(1, -1), features, dog_ids, image_ids
        )
        score = float(score)
        return MatchResult(
            matched=score >= self.threshold,
            score=score,
            dog_id=str(dog_id) if score >= self.threshold else None,
            image_id=str(image_id) if score >= self.threshold else None,
        )

    def identify(self, image_bytes: bytes) -> tuple[MatchResult, np.ndarray]:
        # Never let an arbitrary image become a biometric template. ResNet50
        # will produce an embedding for any pixels, including eyes and scenery.
        validate_nose_image(image_bytes)
        embedding = self.embed(image_bytes)
        return self.match_embedding(embedding), embedding

    def add_embedding(self, dog_id: str, image_id: str, embedding: np.ndarray) -> None:
        vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        with self._lock:
            self._features = np.vstack([self._features, vector])
            self._dog_ids.append(dog_id)
            self._image_ids.append(image_id)
