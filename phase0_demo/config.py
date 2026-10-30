from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    embedding_db_path: str = "data/embeddings.sqlite3"
    image_root: str = "data/images"
    public_base_url: str = ""
    media_signing_secret: str = ""
    whatsapp_access_token: str = ""
    phone_number_id: str = ""
    verify_token: str = ""
    meta_app_secret: str = ""
    graph_api_version: str = ""
    whatsapp_provider: str = "meta"
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_from: str = ""
    match_threshold: float = 0.95
    presigned_url_seconds: int = 900
    local_mode: bool = False

    @classmethod
    def from_environment(cls) -> "Settings":
        mapping = {
            "embedding_db_path": os.getenv("EMBEDDING_DB_PATH", "data/embeddings.sqlite3"),
            "image_root": os.getenv("IMAGE_ROOT", "data/images"),
            "public_base_url": os.getenv("PUBLIC_BASE_URL", ""),
            "media_signing_secret": os.getenv("MEDIA_SIGNING_SECRET", ""),
            "whatsapp_access_token": os.getenv("WHATSAPP_ACCESS_TOKEN", ""),
            "phone_number_id": os.getenv("PHONE_NUMBER_ID", ""),
            "verify_token": os.getenv("VERIFY_TOKEN", ""),
            "meta_app_secret": os.getenv("META_APP_SECRET", ""),
            "graph_api_version": os.getenv("GRAPH_API_VERSION", ""),
            "whatsapp_provider": os.getenv("WHATSAPP_PROVIDER", "meta").lower(),
            "twilio_account_sid": os.getenv("TWILIO_ACCOUNT_SID", ""),
            "twilio_auth_token": os.getenv("TWILIO_AUTH_TOKEN", ""),
            "twilio_whatsapp_from": os.getenv("TWILIO_WHATSAPP_FROM", ""),
            "match_threshold": float(os.getenv("MATCH_THRESHOLD", "0.95")),
            "presigned_url_seconds": int(os.getenv("PRESIGNED_URL_SECONDS", "900")),
            "local_mode": os.getenv("LOCAL_MODE", "false").lower() == "true",
        }
        if not mapping["public_base_url"] and os.getenv("NGROK_DOMAIN"):
            mapping["public_base_url"] = f"https://{os.environ['NGROK_DOMAIN']}"
        return cls(**mapping)

    def validate(self) -> None:
        if self.whatsapp_provider not in {"meta", "twilio"}:
            raise RuntimeError("WHATSAPP_PROVIDER must be either 'meta' or 'twilio'")

        required = []
        if not self.local_mode:
            required += ["public_base_url", "media_signing_secret"]
            if self.whatsapp_provider == "meta":
                required += [
                    "whatsapp_access_token",
                    "phone_number_id",
                    "verify_token",
                    "meta_app_secret",
                    "graph_api_version",
                ]
            else:
                required += [
                    "twilio_account_sid",
                    "twilio_auth_token",
                    "twilio_whatsapp_from",
                ]
        missing = [name for name in required if not getattr(self, name)]
        if missing:
            raise RuntimeError(f"Missing required configuration: {', '.join(missing)}")
