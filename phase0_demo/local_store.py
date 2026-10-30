from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import quote

import numpy as np

from .config import Settings
from .domain import DogProfile


class LocalStore:
    """Single-machine persistence for the complete Phase 0 demo."""

    def __init__(self, settings: Settings):
        self.db_path = Path(settings.embedding_db_path)
        self.image_root = Path(settings.image_root).resolve()
        self.public_base_url = settings.public_base_url.rstrip("/")
        self.media_signing_secret = settings.media_signing_secret.encode("utf-8")
        self.presigned_url_seconds = settings.presigned_url_seconds
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.image_root.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS dogs (
                    dog_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    location TEXT NOT NULL,
                    vaccinated INTEGER NOT NULL CHECK (vaccinated IN (0, 1)),
                    sterilized INTEGER NOT NULL CHECK (sterilized IN (0, 1)),
                    image_key TEXT NOT NULL,
                    registration_status TEXT NOT NULL CHECK (registration_status IN ('pending', 'active')),
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    dog_id TEXT NOT NULL REFERENCES dogs(dog_id) ON DELETE CASCADE,
                    image_id TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    source_image TEXT,
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY (dog_id, image_id)
                );

                CREATE TABLE IF NOT EXISTS pending_embeddings (
                    pending_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS whatsapp_sessions (
                    phone_hash TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    data TEXT NOT NULL,
                    expires_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS processed_messages (
                    message_id TEXT PRIMARY KEY,
                    processed_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL
                );
                """
            )

    @staticmethod
    def phone_hash(phone: str) -> str:
        return hashlib.sha256(phone.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _blob(embedding: np.ndarray | bytes) -> bytes:
        raw = (
            embedding
            if isinstance(embedding, bytes)
            else np.asarray(embedding, dtype=np.float32).reshape(-1).tobytes()
        )
        if len(raw) != 2048 * 4:
            raise ValueError("Embedding must contain exactly 2,048 float32 values")
        return raw

    def _path(self, key: str) -> Path:
        normalized = PurePosixPath(key)
        if normalized.is_absolute() or ".." in normalized.parts:
            raise ValueError("Invalid image key")
        path = (self.image_root / Path(*normalized.parts)).resolve()
        if self.image_root not in path.parents:
            raise ValueError("Image key escapes the image root")
        return path

    def claim_message(self, message_id: str) -> bool:
        now = int(time.time())
        try:
            with self._write_lock, self._connect() as connection:
                connection.execute(
                    "INSERT INTO processed_messages VALUES (?, ?, ?)",
                    (message_id, now, now + 86400),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def release_message(self, message_id: str) -> None:
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM processed_messages WHERE message_id = ?", (message_id,)
            )

    def get_session(self, phone: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state, data FROM whatsapp_sessions WHERE phone_hash = ?",
                (self.phone_hash(phone),),
            ).fetchone()
        return {"state": row["state"], "data": json.loads(row["data"])} if row else {}

    def put_session(self, phone: str, state: str, data: dict[str, Any] | None = None) -> None:
        now = int(time.time())
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO whatsapp_sessions (phone_hash, state, data, expires_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(phone_hash) DO UPDATE SET state=excluded.state,
                    data=excluded.data, expires_at=excluded.expires_at,
                    updated_at=excluded.updated_at
                """,
                (self.phone_hash(phone), state, json.dumps(data or {}), now + 86400, now),
            )

    def clear_session(self, phone: str) -> None:
        self.put_session(phone, "READY", {})

    def get_profile(self, dog_id: str) -> DogProfile | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT dog_id, name, location, vaccinated, sterilized, image_key
                FROM dogs WHERE dog_id = ? AND registration_status = 'active'
                """,
                (dog_id,),
            ).fetchone()
        if not row:
            return None
        return DogProfile(
            dog_id=row["dog_id"], name=row["name"], location=row["location"],
            vaccinated=bool(row["vaccinated"]), sterilized=bool(row["sterilized"]),
            image_key=row["image_key"],
        )

    def put_profile(self, profile: DogProfile, status: str = "active") -> None:
        now = int(time.time())
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO dogs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dog_id) DO UPDATE SET name=excluded.name,
                    location=excluded.location, vaccinated=excluded.vaccinated,
                    sterilized=excluded.sterilized, image_key=excluded.image_key,
                    registration_status=excluded.registration_status,
                    updated_at=excluded.updated_at
                """,
                (profile.dog_id, profile.name, profile.location, int(profile.vaccinated),
                 int(profile.sterilized), profile.image_key, status, now, now),
            )

    def delete_profile(self, dog_id: str) -> None:
        with self._write_lock, self._connect() as connection:
            connection.execute("DELETE FROM dogs WHERE dog_id = ?", (dog_id,))

    def update_profile(self, dog_id: str, field: str, value: str | bool) -> DogProfile:
        columns = {"location", "vaccinated", "sterilized"}
        if field not in columns:
            raise ValueError(f"Unsupported profile field: {field}")
        stored_value = int(value) if isinstance(value, bool) else value
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE dogs SET {field} = ?, updated_at = ? "
                "WHERE dog_id = ? AND registration_status = 'active'",
                (stored_value, int(time.time()), dog_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Active dog not found: {dog_id}")
        profile = self.get_profile(dog_id)
        if profile is None:
            raise RuntimeError("Profile disappeared after update")
        return profile

    def put_embedding(self, dog_id: str, image_id: str, embedding, source_image: str = "") -> None:
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO embeddings VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(dog_id, image_id) DO UPDATE SET
                    embedding=excluded.embedding, source_image=excluded.source_image
                """,
                (dog_id, image_id, self._blob(embedding), source_image, int(time.time())),
            )

    def iter_embeddings(self) -> Iterable[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT dog_id, image_id, embedding, source_image FROM embeddings ORDER BY dog_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def put_pending_embedding(self, pending_id: str, embedding) -> None:
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO pending_embeddings VALUES (?, ?, ?)
                ON CONFLICT(pending_id) DO UPDATE SET embedding=excluded.embedding,
                    created_at=excluded.created_at
                """,
                (pending_id, self._blob(embedding), int(time.time())),
            )

    def get_pending_embedding(self, pending_id: str) -> bytes | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT embedding FROM pending_embeddings WHERE pending_id = ?", (pending_id,)
            ).fetchone()
        return bytes(row["embedding"]) if row else None

    def delete_pending_embedding(self, pending_id: str) -> None:
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM pending_embeddings WHERE pending_id = ?", (pending_id,)
            )

    def upload_image(self, key: str, image_bytes: bytes, content_type: str = "image/jpeg") -> None:
        if content_type not in {"image/jpeg", "image/png"}:
            raise ValueError("Only JPEG and PNG images are supported")
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(image_bytes)
        os.replace(temporary, path)

    def read_image(self, key: str) -> bytes:
        """Read a locally stored image after applying the normal path checks."""
        return self._path(key).read_bytes()

    def delete_image(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def cancel_pending(self, pending_key: str, pending_id: str) -> None:
        self.delete_image(pending_key)
        self.delete_pending_embedding(pending_id)

    def complete_registration(self, profile, image_id, pending_key, pending_id) -> np.ndarray:
        raw = self.get_pending_embedding(pending_id)
        if raw is None:
            raise RuntimeError("Pending embedding is missing; resend the nose image")
        source = self._path(pending_key)
        destination = self._path(profile.image_key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        try:
            with self._write_lock, self._connect() as connection:
                now = int(time.time())
                connection.execute(
                    "INSERT INTO dogs VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)",
                    (profile.dog_id, profile.name, profile.location,
                     int(profile.vaccinated), int(profile.sterilized), profile.image_key,
                     now, now),
                )
                connection.execute(
                    "INSERT INTO embeddings VALUES (?, ?, ?, ?, ?)",
                    (profile.dog_id, image_id, raw, pending_key, now),
                )
                connection.execute(
                    "DELETE FROM pending_embeddings WHERE pending_id = ?", (pending_id,)
                )
        except Exception:
            source.parent.mkdir(parents=True, exist_ok=True)
            os.replace(destination, source)
            raise
        return np.frombuffer(raw, dtype=np.float32).copy()

    def image_url(self, key: str) -> str:
        expires = int(time.time()) + self.presigned_url_seconds
        payload = f"{key}|{expires}".encode("utf-8")
        signature = hmac.new(self.media_signing_secret, payload, hashlib.sha256).hexdigest()
        return f"{self.public_base_url}/media/{quote(key)}?expires={expires}&signature={signature}"

    def authorized_media_path(self, key: str, expires: int, signature: str) -> Path | None:
        if expires < int(time.time()):
            return None
        payload = f"{key}|{expires}".encode("utf-8")
        expected = hmac.new(self.media_signing_secret, payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        path = self._path(key)
        return path if path.is_file() else None

    def cleanup_expired(self) -> None:
        now = int(time.time())
        with self._write_lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT data FROM whatsapp_sessions WHERE expires_at < ?", (now,)
            ).fetchall()
            for row in rows:
                data = json.loads(row["data"])
                if data.get("pending_key"):
                    self.delete_image(data["pending_key"])
            connection.execute("DELETE FROM whatsapp_sessions WHERE expires_at < ?", (now,))
            connection.execute("DELETE FROM processed_messages WHERE expires_at < ?", (now,))
            connection.execute(
                "DELETE FROM pending_embeddings WHERE created_at < ?", (now - 86400,)
            )
