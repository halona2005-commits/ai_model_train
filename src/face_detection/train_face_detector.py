import os
import torch
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"E:\canine-nose-print-recognition\ai-model"

DATA_YAML = os.path.join(
    BASE_DIR,
    "dataset",
    "face_detection_yolo",
    "data.yaml"
)

MODEL_PATH = "yolo11n.pt"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "models",
    "face_detector"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("          FINAL DOG FACE DETECTOR TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if torch.xpu.is_available():

        device = "xpu:0"

        print()
        print("Intel GPU detected.")
        print(
            f"GPU: {torch.xpu.get_device_name(0)}"
        )

    else:

        device = "cpu"

        print()
        print(
            "Intel XPU not available."
        )
        print(
            "Using CPU."
        )

    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    print()
    print(f"Dataset : {DATA_YAML}")
    print(f"Model   : {MODEL_PATH}")
    print(f"Device  : {device}")

    # --------------------------------------------------------
    # LOAD YOLO
    # --------------------------------------------------------

    model = YOLO(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train(

        data=DATA_YAML,

        epochs=30,

        imgsz=640,

        batch=4,

        patience=8,

        device=device,

        workers=2,

        amp=False,

        pretrained=True,

        project=OUTPUT_DIR,

        name="yolo11n_face_final",

        exist_ok=True,

        verbose=True
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("          FACE DETECTOR TRAINING COMPLETE")
    print("=" * 70)

    print()
    print("Best model should be here:")

    print(
        os.path.join(
            OUTPUT_DIR,
            "yolo11n_face_final",
            "weights",
            "best.pt"
        )
    )


if __name__ == "__main__":
    main()