from __future__ import annotations

import hashlib
import hmac
from typing import Any

import requests

from .config import Settings
from .domain import IncomingMessage


class MetaClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {settings.whatsapp_access_token}"})
        self.base_url = f"https://graph.facebook.com/{settings.graph_api_version}"

    def validate_signature(self, body: bytes, signature: str | None) -> bool:
        if not signature or not signature.startswith("sha256="):
            return False
        expected = hmac.new(
            self.settings.meta_app_secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(signature[7:], expected)

    def download_media(self, media_id: str) -> tuple[bytes, str]:
        metadata = self.session.get(f"{self.base_url}/{media_id}", timeout=10)
        metadata.raise_for_status()
        media_url = metadata.json()["url"]
        response = self.session.get(media_url, timeout=20)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "image/jpeg").split(";", 1)[0]
        if content_type not in {"image/jpeg", "image/png"}:
            raise ValueError(f"Unsupported Meta media type: {content_type}")
        if len(response.content) > 5 * 1024 * 1024:
            raise ValueError("Image exceeds the 5 MB demo limit")
        return response.content, content_type

    def _send(self, recipient: str, payload: dict[str, Any]) -> None:
        response = self.session.post(
            f"{self.base_url}/{self.settings.phone_number_id}/messages",
            json={"messaging_product": "whatsapp", "to": recipient, **payload},
            timeout=20,
        )
        response.raise_for_status()

    def send_text(self, recipient: str, text: str) -> None:
        self._send(recipient, {"type": "text", "text": {"preview_url": False, "body": text}})

    def send_image(self, recipient: str, image_url: str, caption: str) -> None:
        self._send(
            recipient,
            {"type": "image", "image": {"link": image_url, "caption": caption[:1024]}},
        )


def parse_messages(payload: dict[str, Any]) -> list[IncomingMessage]:
    if payload.get("object") != "whatsapp_business_account":
        return []
    parsed: list[IncomingMessage] = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            for message in change.get("value", {}).get("messages", []):
                message_type = message.get("type", "")
                parsed.append(
                    IncomingMessage(
                        message_id=message.get("id", ""),
                        sender=message.get("from", ""),
                        message_type=message_type,
                        text=message.get("text", {}).get("body", "") if message_type == "text" else "",
                        media_id=message.get("image", {}).get("id", "") if message_type == "image" else "",
                    )
                )
    return [message for message in parsed if message.message_id and message.sender]
