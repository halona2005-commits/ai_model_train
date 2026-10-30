from __future__ import annotations

from collections.abc import Mapping

import requests
from twilio.request_validator import RequestValidator
from twilio.twiml.messaging_response import MessagingResponse

from .config import Settings
from .domain import IncomingMessage


SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png"}


class TwilioClient:
    """Twilio WhatsApp transport using the maintained request validator."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.request_validator = RequestValidator(settings.twilio_auth_token)
        self.session = requests.Session()
        self.session.auth = (settings.twilio_account_sid, settings.twilio_auth_token)
        self.messages_url = (
            "https://api.twilio.com/2010-04-01/Accounts/"
            f"{settings.twilio_account_sid}/Messages.json"
        )

    def validate_signature(
        self, url: str, params: Mapping[str, str], signature: str | None
    ) -> bool:
        return bool(signature) and self.request_validator.validate(url, params, signature)

    def download_media(self, media_url: str) -> tuple[bytes, str]:
        if not media_url.lower().startswith("https://"):
            raise ValueError("Twilio media URL must use HTTPS")
        response = self.session.get(media_url, timeout=20)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        if content_type not in SUPPORTED_IMAGE_TYPES:
            raise ValueError(f"Unsupported Twilio media type: {content_type or 'unknown'}")
        if len(response.content) > 5 * 1024 * 1024:
            raise ValueError("Image exceeds the 5 MB demo limit")
        return response.content, content_type

    def _send(self, recipient: str, text: str, image_url: str | None = None) -> None:
        payload = {
            "From": self.settings.twilio_whatsapp_from,
            "To": recipient,
            "Body": text,
        }
        if image_url:
            payload["MediaUrl"] = image_url
        response = self.session.post(self.messages_url, data=payload, timeout=20)
        response.raise_for_status()

    def send_text(self, recipient: str, text: str) -> None:
        self._send(recipient, text)

    def send_image(self, recipient: str, image_url: str, caption: str) -> None:
        self._send(recipient, caption, image_url)


def parse_message(params: Mapping[str, str]) -> IncomingMessage | None:
    message_id = params.get("MessageSid", "")
    sender = params.get("From", "")
    if not message_id or not sender:
        return None
    media_url = params.get("MediaUrl0", "")
    if media_url:
        return IncomingMessage(message_id, sender, "image", media_id=media_url)
    return IncomingMessage(message_id, sender, "text", text=params.get("Body", ""))


def render_twiml(outputs) -> str:
    response = MessagingResponse()
    for output in outputs:
        message = response.message()
        message.body(output.text)
        if output.image_key:
            message.media(output.image_key)
    return str(response)
