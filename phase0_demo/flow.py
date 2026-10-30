from __future__ import annotations

import re
import uuid

from .domain import DogProfile, IncomingMessage, OutboundMessage
from .image_validation import InvalidNoseImage


YES_NO = {"YES": True, "NO": False}


def profile_text(profile: DogProfile, heading: str = "Dog identified") -> str:
    return (
        f"{heading}\n\n"
        f"Dog ID: {profile.dog_id}\n"
        f"Name: {profile.name}\n"
        f"Location: {profile.location}\n"
        f"Vaccination: {'Yes' if profile.vaccinated else 'No'}\n"
        f"Sterilization: {'Yes' if profile.sterilized else 'No'}\n\n"
        "1. Update Location\n"
        "2. Update Vaccination\n"
        "3. Update Sterilization\n"
        "4. Main Menu"
    )


class ConversationService:
    def __init__(self, store, recognition):
        self.store = store
        self.recognition = recognition

    def handle(self, message: IncomingMessage, image: bytes | None = None, content_type: str = "image/jpeg") -> list[OutboundMessage]:
        if message.message_type == "image":
            if image is None:
                raise ValueError("Image bytes are required")
            return self._handle_image(message, image, content_type)
        if message.message_type != "text":
            return [OutboundMessage("Please send a JPEG/PNG close-up nose image or a text reply.")]
        return self._handle_text(message.sender, message.text.strip())

    def _handle_image(self, message: IncomingMessage, image: bytes, content_type: str) -> list[OutboundMessage]:
        session = self.store.get_session(message.sender)
        state = session.get("state", "READY")
        session_data = dict(session.get("data", {}))
        if state not in {"READY", "KNOWN_MENU", "UNKNOWN_MENU"}:
            return [OutboundMessage("Please complete or cancel the current form before sending another image.")]
        if state == "UNKNOWN_MENU" and session_data.get("pending_key"):
            self.store.cancel_pending(
                session_data["pending_key"], session_data["pending_id"]
            )

        try:
            result, embedding = self.recognition.identify(image)
        except InvalidNoseImage as exc:
            return [
                OutboundMessage(
                    f"This image cannot be used as a nose print: {exc}\n\n"
                    "Send a square, tightly cropped close-up showing only the dog's nose."
                )
            ]
        if result.matched and result.dog_id:
            profile = self.store.get_profile(result.dog_id)
            if profile is None:
                raise RuntimeError(f"Embedding points to missing profile {result.dog_id}")
            self.store.put_session(message.sender, "KNOWN_MENU", {"dog_id": profile.dog_id})
            return [OutboundMessage(profile_text(profile), profile.image_key)]

        phone_hash = self.store.phone_hash(message.sender)
        extension = ".png" if content_type == "image/png" else ".jpg"
        pending_key = f"pending/{phone_hash}/{message.message_id}{extension}"
        pending_id = message.message_id
        uploaded = False
        pending_saved = False
        try:
            self.store.upload_image(pending_key, image, content_type)
            uploaded = True
            self.store.put_pending_embedding(pending_id, embedding)
            pending_saved = True
            self.store.put_session(
                message.sender,
                "UNKNOWN_MENU",
                {"pending_key": pending_key, "pending_id": pending_id},
            )
        except Exception:
            if pending_saved:
                self.store.delete_pending_embedding(pending_id)
            if uploaded:
                self.store.delete_image(pending_key)
            raise
        return [
            OutboundMessage(
                "No registered dog matched this nose print.\n\n1. Register\n2. Cancel"
            )
        ]

    def _handle_text(self, phone: str, text: str) -> list[OutboundMessage]:
        command = text.strip()
        session = self.store.get_session(phone)
        state = session.get("state", "READY")
        data = dict(session.get("data", {}))
        if command.lower() in {"hi", "hello", "start", "menu"}:
            if data.get("pending_key") and data.get("pending_id"):
                self.store.cancel_pending(data["pending_key"], data["pending_id"])
            self.store.clear_session(phone)
            return [OutboundMessage("Send a clear close-up dog nose image to identify or register the dog.")]

        if state == "READY":
            return [OutboundMessage("Send a clear close-up dog nose image to begin.")]
        if state == "KNOWN_MENU":
            choices = {"1": "UPDATE_LOCATION", "2": "UPDATE_VACCINATION", "3": "UPDATE_STERILIZATION"}
            if command == "4":
                self.store.clear_session(phone)
                return [OutboundMessage("Send another close-up dog nose image.")]
            next_state = choices.get(command)
            if not next_state:
                return [OutboundMessage("Reply with 1, 2, 3, or 4.")]
            self.store.put_session(phone, next_state, data)
            prompt = "Enter the new location." if command == "1" else "Reply YES or NO."
            return [OutboundMessage(prompt)]
        if state == "UPDATE_LOCATION":
            if not self._valid_text(command):
                return [OutboundMessage("Enter a valid location using letters and normal punctuation.")]
            return self._finish_update(phone, data["dog_id"], "location", command)
        if state in {"UPDATE_VACCINATION", "UPDATE_STERILIZATION"}:
            value = YES_NO.get(command.upper())
            if value is None:
                return [OutboundMessage("Invalid answer. Reply only YES or NO.")]
            field = "vaccinated" if state == "UPDATE_VACCINATION" else "sterilized"
            return self._finish_update(phone, data["dog_id"], field, value)
        if state == "UNKNOWN_MENU":
            if command == "2":
                self.store.cancel_pending(data["pending_key"], data["pending_id"])
                self.store.clear_session(phone)
                return [OutboundMessage("Registration cancelled. Send another nose image when ready.")]
            if command != "1":
                return [OutboundMessage("Reply 1 to Register or 2 to Cancel.")]
            self.store.put_session(phone, "REGISTER_NAME", data)
            return [OutboundMessage("Enter the dog's name.")]
        if state == "REGISTER_NAME":
            if not self._valid_text(command):
                return [OutboundMessage("Enter a valid dog name.")]
            data["name"] = command
            self.store.put_session(phone, "REGISTER_LOCATION", data)
            return [OutboundMessage("Enter the dog's location.")]
        if state == "REGISTER_LOCATION":
            if not self._valid_text(command):
                return [OutboundMessage("Enter a valid location.")]
            data["location"] = command
            self.store.put_session(phone, "REGISTER_VACCINATION", data)
            return [OutboundMessage("Is the dog vaccinated? Reply YES or NO.")]
        if state == "REGISTER_VACCINATION":
            return self._registration_boolean(phone, data, command, "vaccinated", "REGISTER_STERILIZATION", "Is the dog sterilized? Reply YES or NO.")
        if state == "REGISTER_STERILIZATION":
            value = YES_NO.get(command.upper())
            if value is None:
                return [OutboundMessage("Invalid answer. Reply only YES or NO.")]
            data["sterilized"] = value
            self.store.put_session(phone, "REGISTER_CONFIRM", data)
            summary = (
                "Confirm registration:\n\n"
                f"Name: {data['name']}\nLocation: {data['location']}\n"
                f"Vaccination: {'Yes' if data['vaccinated'] else 'No'}\n"
                f"Sterilization: {'Yes' if data['sterilized'] else 'No'}\n\n"
                "1. Confirm\n2. Cancel"
            )
            return [OutboundMessage(summary)]
        if state == "REGISTER_CONFIRM":
            if command == "2":
                self.store.cancel_pending(data["pending_key"], data["pending_id"])
                self.store.clear_session(phone)
                return [OutboundMessage("Registration cancelled.")]
            if command != "1":
                return [OutboundMessage("Reply 1 to Confirm or 2 to Cancel.")]
            return self._complete_registration(phone, data)

        self.store.clear_session(phone)
        return [OutboundMessage("Session reset. Send a close-up dog nose image.")]

    def _finish_update(self, phone: str, dog_id: str, field: str, value: str | bool) -> list[OutboundMessage]:
        profile = self.store.update_profile(dog_id, field, value)
        self.store.put_session(phone, "KNOWN_MENU", {"dog_id": dog_id})
        return [OutboundMessage(profile_text(profile, "Dog record updated"), profile.image_key)]

    def _registration_boolean(self, phone: str, data: dict, command: str, field: str, next_state: str, prompt: str) -> list[OutboundMessage]:
        value = YES_NO.get(command.upper())
        if value is None:
            return [OutboundMessage("Invalid answer. Reply only YES or NO.")]
        data[field] = value
        self.store.put_session(phone, next_state, data)
        return [OutboundMessage(prompt)]

    def _complete_registration(self, phone: str, data: dict) -> list[OutboundMessage]:
        dog_id = "DOG-" + uuid.uuid4().hex[:8].upper()
        image_id = uuid.uuid4().hex
        suffix = ".png" if data["pending_key"].lower().endswith(".png") else ".jpg"
        permanent_key = f"dogs/{dog_id}/nose/{image_id}{suffix}"
        profile = DogProfile(
            dog_id=dog_id,
            name=data["name"],
            location=data["location"],
            vaccinated=bool(data["vaccinated"]),
            sterilized=bool(data["sterilized"]),
            image_key=permanent_key,
        )
        embedding = self.store.complete_registration(
            profile, image_id, data["pending_key"], data["pending_id"]
        )
        self.recognition.add_embedding(dog_id, image_id, embedding)
        self.store.put_session(phone, "KNOWN_MENU", {"dog_id": dog_id})
        return [OutboundMessage(profile_text(profile, "Dog registered successfully"), profile.image_key)]

    @staticmethod
    def _valid_text(value: str) -> bool:
        return bool(1 <= len(value) <= 80 and re.search(r"[A-Za-z]", value) and not re.search(r"[<>]", value))
