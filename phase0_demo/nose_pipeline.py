from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .image_validation import InvalidNoseImage, validate_nose_image


MODEL_ROOT = Path(__file__).resolve().parents[1] / "models"


@dataclass(frozen=True)
class NoseSample:
    image_bytes: bytes
    content_type: str
    input_type: str
    face_confidence: float | None = None
    nose_confidence: float | None = None
    validator_probability: float | None = None


class NosePipeline:
    """Optional upstream localization adapter for the existing embedding service.

    The upstream validator is advisory for already cropped input because it
    rejected this project's unknown nose sample. It is required after a full
    image is localized, where it rejected the previously enrolled eye crop.
    """

    def __init__(
        self,
        model_root: Path = MODEL_ROOT,
        *,
        face_threshold: float = 0.25,
        nose_threshold: float = 0.25,
        fallback_threshold: float = 0.15,
        validator_threshold: float = 0.50,
        enforce_validator: bool = False,
        enhance: bool = False,
    ) -> None:
        self.model_root = Path(model_root)
        self.face_threshold = face_threshold
        self.nose_threshold = nose_threshold
        self.fallback_threshold = fallback_threshold
        self.validator_threshold = validator_threshold
        self.enforce_validator = enforce_validator
        self.enhance = enhance
        self._face_model = None
        self._nose_model = None
        self._fallback_model = None
        self._validator = None
        self._upsampler = None

    def _paths(self) -> dict[str, Path]:
        root = self.model_root
        return {
            "face": root / "face_detector/yolo11n_face_final/weights/best.pt",
            "nose": root / "nose_detector/yolo11n_nose_tight/weights/best.pt",
            "fallback": root / "nose_detector/yolo11n_nose_augmented/weights/last.pt",
            "validator": root / "dog_nose_validator.keras",
            "enhancement": root / "RealESRGAN_x4plus.pth",
        }

    def start(self) -> None:
        paths = self._paths()
        required = ("face", "nose", "fallback", "validator")
        if self.enhance:
            required += ("enhancement",)
        missing = [str(paths[name]) for name in required if not paths[name].is_file()]
        pointers = [
            str(paths[name]) for name in required
            if paths[name].is_file()
            and paths[name].stat().st_size < 1024
            and paths[name].read_bytes().startswith(b"version https://git-lfs")
        ]
        if missing or pointers:
            raise RuntimeError(
                "Nose model weights are missing or are Git LFS pointers. "
                "Run `git lfs install` and `git lfs pull`. "
                f"Affected files: {', '.join(missing + pointers)}"
            )

        from ultralytics import YOLO
        from tensorflow.keras.models import load_model

        self._face_model = YOLO(str(paths["face"]))
        self._nose_model = YOLO(str(paths["nose"]))
        self._fallback_model = YOLO(str(paths["fallback"]))
        self._validator = load_model(str(paths["validator"]), compile=False)

        if self.enhance:
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
            except ImportError as exc:
                raise RuntimeError(
                    "Enhancement requires the optional BasicSR and Real-ESRGAN packages"
                ) from exc
            self._upsampler = RealESRGANer(
                scale=4,
                model_path=str(paths["enhancement"]),
                model=RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4,
                ),
                tile=128,
                tile_pad=10,
                pre_pad=0,
                half=False,
            )

    @staticmethod
    def _best_box(model, image: np.ndarray, *, size: int, threshold: float):
        result = model.predict(
            source=image, imgsz=size, conf=threshold, verbose=False
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None, None
        scores = boxes.conf.cpu().numpy()
        index = int(np.argmax(scores))
        return boxes.xyxy[index].cpu().numpy(), float(scores[index])

    @staticmethod
    def _crop(image: np.ndarray, box, padding: float, square: bool) -> np.ndarray | None:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = map(float, box)
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        box_width = (x2 - x1) * (1 + 2 * padding)
        box_height = (y2 - y1) * (1 + 2 * padding)
        if square:
            side = max(1, int(round(max(box_width, box_height))))
            left = max(0, min(int(round(center_x - side / 2)), width - side))
            top = max(0, min(int(round(center_y - side / 2)), height - side))
            right = min(width, left + side)
            bottom = min(height, top + side)
        else:
            left = max(0, int(round(center_x - box_width / 2)))
            top = max(0, int(round(center_y - box_height / 2)))
            right = min(width, int(round(center_x + box_width / 2)))
            bottom = min(height, int(round(center_y + box_height / 2)))
        crop = image[top:bottom, left:right]
        if not crop.size:
            return None
        if square and crop.shape[0] != crop.shape[1]:
            pad_bottom = side - crop.shape[0]
            pad_right = side - crop.shape[1]
            crop = cv2.copyMakeBorder(
                crop, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE
            )
        return crop

    def _nose_probability(self, image: np.ndarray) -> float:
        from tensorflow.keras.applications.resnet50 import preprocess_input

        rgb = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB)
        batch = preprocess_input(rgb.astype(np.float32))[None, ...]
        # Upstream class 0 is nose, class 1 is non-nose.
        return 1.0 - float(self._validator.predict(batch, verbose=0)[0][0])

    def process(self, image_bytes: bytes, content_type: str = "image/jpeg") -> NoseSample:
        if self._face_model is None:
            raise RuntimeError("NosePipeline.start() must run before processing images")
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidNoseImage("The uploaded file is not a readable image.")
        if min(image.shape[:2]) < 128:
            raise InvalidNoseImage("The image is too small. Send a closer nose photo.")

        # Keep already cropped near-square inputs on the existing embedding
        # path. The face detector often mislabels close-up noses as full faces.
        # A low validator score triggers a face check for square full-dog
        # photos; a face covering most of the frame still counts as a crop.
        height, width = image.shape[:2]
        near_square = max(width / height, height / width) <= 1.4
        face_box = face_confidence = None
        direct_probability = None
        if near_square:
            validate_nose_image(image_bytes)
            direct_probability = self._nose_probability(image)
            if direct_probability < self.validator_threshold:
                face_box, face_confidence = self._best_box(
                    self._face_model, image, size=640,
                    threshold=self.face_threshold,
                )
                if face_box is not None:
                    x1, y1, x2, y2 = face_box
                    coverage = max(0, x2 - x1) * max(0, y2 - y1) / (width * height)
                    if coverage >= 0.8:
                        face_box = face_confidence = None

        if near_square and face_box is None:
            probability = direct_probability
            if self.enforce_validator and probability < self.validator_threshold:
                raise InvalidNoseImage("No usable dog nose was detected.")
            crop = image
            input_type = "DOG_NOSE"
            face_confidence = nose_confidence = None
        else:
            if face_box is None:
                face_box, face_confidence = self._best_box(
                    self._face_model, image, size=640, threshold=self.face_threshold
                )
            if face_box is None:
                raise InvalidNoseImage("No dog face was detected in the image.")
            face_crop = self._crop(image, face_box, padding=0.15, square=False)
            if face_crop is None:
                raise InvalidNoseImage("The dog face could not be cropped.")
            nose_box, nose_confidence = self._best_box(
                self._nose_model, face_crop, size=1024,
                threshold=self.nose_threshold,
            )
            if nose_box is None:
                nose_box, nose_confidence = self._best_box(
                    self._fallback_model, face_crop, size=1024,
                    threshold=self.fallback_threshold,
                )
            if nose_box is None:
                raise InvalidNoseImage("A dog face was found, but no nose was detected.")
            crop = self._crop(face_crop, nose_box, padding=0.10, square=True)
            if crop is None or min(crop.shape[:2]) < 128:
                raise InvalidNoseImage("The detected nose is too small for matching.")
            ok, encoded = cv2.imencode(".jpg", crop)
            if not ok:
                raise InvalidNoseImage("The detected nose could not be encoded.")
            validate_nose_image(encoded.tobytes())
            probability = self._nose_probability(crop)
            if probability < self.validator_threshold:
                raise InvalidNoseImage("The detected region is not a usable dog nose.")
            input_type = "FULL_DOG"

        if self.enhance:
            crop, _ = self._upsampler.enhance(crop, outscale=4)
            crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            ok, encoded = cv2.imencode(".jpg", crop)
            if not ok:
                raise InvalidNoseImage("The enhanced nose could not be encoded.")
            return NoseSample(
                encoded.tobytes(), "image/jpeg", input_type,
                face_confidence, nose_confidence, probability,
            )

        if input_type == "DOG_NOSE":
            return NoseSample(
                image_bytes, content_type, input_type,
                face_confidence, nose_confidence, probability,
            )
        return NoseSample(
            encoded.tobytes(), "image/jpeg", input_type,
            face_confidence, nose_confidence, probability,
        )
