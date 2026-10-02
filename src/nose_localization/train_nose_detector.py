import os
import torch

from ultralytics import YOLO


BASE_DIR = r"E:\canine-nose-print-recognition\ai-model"

DATA_YAML = os.path.join(
    BASE_DIR,
    "dataset",
    "nose_detection_yolo_tight",
    "data.yaml"
)

MODEL_PATH = "yolo11n.pt"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "models",
    "nose_detector"
)


def main():

    print()
    print("=" * 70)
    print("              NOSE DETECTOR TRAINING")
    print("=" * 70)

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
        print("Intel XPU not available. Using CPU.")

    print()
    print(f"Dataset : {DATA_YAML}")
    print(f"Model   : {MODEL_PATH}")
    print(f"Device  : {device}")

    model = YOLO(MODEL_PATH)

    model.train(
        data=DATA_YAML,

        epochs=30,

        imgsz=640,

        batch=8,

        patience=8,

        device=device,

        workers=4,

        project=OUTPUT_DIR,

        name="yolo11n_nose_tight",

        exist_ok=True,

        pretrained=True,

        amp=False,

        verbose=True
    )

    print()
    print("=" * 70)
    print("              TRAINING COMPLETE")
    print("=" * 70)

    print()
    print("Best model should be saved at:")

    print(
        os.path.join(
            OUTPUT_DIR,
            "yolo11n_nose_tight",
            "weights",
            "best.pt"
        )
    )


if __name__ == "__main__":
    main()