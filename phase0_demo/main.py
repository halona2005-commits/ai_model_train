from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from urllib.parse import parse_qsl

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response

from .config import Settings
from .domain import OutboundMessage
from .flow import ConversationService
from .local_store import LocalStore
from .meta import MetaClient, parse_messages
from .recognition import RecognitionService
from .twilio import SUPPORTED_IMAGE_TYPES, TwilioClient, parse_message, render_twiml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("phase0")


def create_app(
    settings: Settings | None = None,
    store=None,
    meta=None,
    twilio=None,
    recognition=None,
    start_worker: bool = True,
) -> FastAPI:
    settings = settings or Settings.from_environment()
    settings.validate()
    store = store or LocalStore(settings)
    if settings.whatsapp_provider == "meta":
        meta = meta or MetaClient(settings)
    else:
        twilio = twilio or TwilioClient(settings)
    recognition = recognition or RecognitionService(store, settings.match_threshold)
    flow = ConversationService(store, recognition)
    queue: asyncio.Queue = asyncio.Queue(maxsize=20)
    twilio_processing_lock = asyncio.Lock()

    async def handle_incoming(transport, message):
        image = None
        content_type = "image/jpeg"
        if message.message_type == "image":
            image, content_type = await asyncio.to_thread(
                transport.download_media, message.media_id
            )
        return await asyncio.to_thread(flow.handle, message, image, content_type)

    async def send_outputs(transport, recipient, outputs):
        for output in outputs:
            if output.image_key:
                await asyncio.to_thread(
                    transport.send_image,
                    recipient,
                    store.image_url(output.image_key),
                    output.text,
                )
            else:
                await asyncio.to_thread(transport.send_text, recipient, output.text)

    async def process_incoming(transport, message):
        try:
            outputs = await handle_incoming(transport, message)
            await send_outputs(transport, message.sender, outputs)
        except Exception:
            logger.exception("Message processing failed message_id=%s", message.message_id)
            await asyncio.to_thread(
                transport.send_text,
                message.sender,
                "Sorry, processing failed. Please try again.",
            )

    async def worker():
        while True:
            transport, message = await queue.get()
            try:
                await process_incoming(transport, message)
            finally:
                queue.task_done()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await asyncio.to_thread(store.cleanup_expired)
        await asyncio.to_thread(recognition.start, not settings.local_mode)
        task = asyncio.create_task(worker()) if start_worker else None
        app.state.queue = queue
        app.state.flow = flow
        try:
            yield
        finally:
            if task:
                task.cancel()

    app = FastAPI(title="Living Bridges Phase 0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "queue_depth": queue.qsize()}

    @app.get("/media/{key:path}")
    def serve_media(key: str, expires: int, signature: str):
        path = store.authorized_media_path(key, expires, signature)
        if path is None:
            raise HTTPException(status_code=403, detail="Invalid or expired media URL")
        return FileResponse(path)

    @app.get("/webhook", response_class=PlainTextResponse)
    def verify_webhook(
        hub_mode: str | None = Query(None, alias="hub.mode"),
        hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
        hub_challenge: str | None = Query(None, alias="hub.challenge"),
    ):
        if settings.whatsapp_provider != "meta":
            raise HTTPException(status_code=503, detail="Meta transport is inactive")
        if hub_mode == "subscribe" and hub_verify_token == settings.verify_token and hub_challenge:
            return hub_challenge
        raise HTTPException(status_code=403, detail="Verification failed")

    @app.post("/webhook", status_code=200)
    async def whatsapp_webhook(
        request: Request,
        x_hub_signature_256: str | None = Header(None),
    ):
        if settings.whatsapp_provider != "meta":
            raise HTTPException(status_code=503, detail="Meta transport is inactive")
        body = await request.body()
        if not settings.local_mode and not meta.validate_signature(body, x_hub_signature_256):
            raise HTTPException(status_code=401, detail="Invalid Meta signature")
        payload = await request.json()
        accepted = 0
        for message in parse_messages(payload):
            if message.message_type not in {"text", "image"}:
                continue
            if await asyncio.to_thread(store.claim_message, message.message_id):
                try:
                    queue.put_nowait((meta, message))
                    accepted += 1
                except asyncio.QueueFull:
                    await asyncio.to_thread(store.release_message, message.message_id)
                    raise HTTPException(status_code=503, detail="Processing queue is full")
        return {"status": "accepted", "messages": accepted}

    @app.post("/twilio/whatsapp", status_code=200)
    async def twilio_whatsapp_webhook(
        request: Request,
        x_twilio_signature: str | None = Header(None),
    ):
        if settings.whatsapp_provider != "twilio":
            raise HTTPException(status_code=503, detail="Twilio transport is inactive")
        body = await request.body()
        try:
            params = dict(parse_qsl(body.decode("utf-8"), keep_blank_values=True))
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid form encoding") from exc
        webhook_url = f"{settings.public_base_url.rstrip('/')}/twilio/whatsapp"
        if not settings.local_mode and not twilio.validate_signature(
            webhook_url, params, x_twilio_signature
        ):
            raise HTTPException(status_code=401, detail="Invalid Twilio signature")

        message = parse_message(params)
        if message is None:
            raise HTTPException(status_code=400, detail="MessageSid and From are required")
        if message.message_type == "image":
            content_type = params.get("MediaContentType0", "").split(";", 1)[0].lower()
            if content_type not in SUPPORTED_IMAGE_TYPES:
                raise HTTPException(status_code=415, detail="Only JPEG and PNG images are supported")

        if not await asyncio.to_thread(store.claim_message, message.message_id):
            return Response(content=render_twiml([]), media_type="application/xml")

        try:
            # Trial accounts reject free-form REST API sends with error 21654.
            # The legacy WhatsApp Sandbox supports replies in webhook TwiML.
            async with twilio_processing_lock:
                outputs = await handle_incoming(twilio, message)
            for output in outputs:
                if output.image_key:
                    output.image_key = store.image_url(output.image_key)
        except Exception:
            logger.exception("Message processing failed message_id=%s", message.message_id)
            outputs = [OutboundMessage("Sorry, processing failed. Please try again.")]
        return Response(
            content=render_twiml(outputs),
            media_type="application/xml",
        )

    return app
