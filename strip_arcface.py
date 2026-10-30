import zipfile
import shutil
import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path

PROJECT_ROOT = Path(r"D:\internship_RIED\ai-model")
MODEL_PATH = PROJECT_ROOT / "models" / "biometric" / "best_biometric_arcface.keras"
EXTRACT_DIR = PROJECT_ROOT / "temp_extract"
CLEAN_MODEL_PATH = PROJECT_ROOT / "models" / "biometric" / "inference_embedding_model.keras"

print("Extracting original model...")
if EXTRACT_DIR.exists(): shutil.rmtree(EXTRACT_DIR)
EXTRACT_DIR.mkdir()
with zipfile.ZipFile(MODEL_PATH, 'r') as z: z.extractall(EXTRACT_DIR)

# Define the layers so Keras can rebuild it
class ArcFaceLayer(layers.Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def build(self, input_shape): self.W = self.add_weight(shape=(input_shape[0][-1], 4200))
    def call(self, inputs): return inputs[0]

class L2Norm(layers.Layer):
    def call(self, x): return tf.math.l2_normalize(x, axis=1)

print("Rebuilding and loading weights...")
backbone = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3), pooling="avg")
inp = tf.keras.Input(shape=(224, 224, 3))
x = L2Norm()(backbone(inp))
full_model = tf.keras.Model(inputs=inp, outputs=x)
full_model.load_weights(EXTRACT_DIR / "model.weights.h5")

print("Saving clean inference model (no custom layers)...")
full_model.save(CLEAN_MODEL_PATH)
print(f"✅ Saved to: {CLEAN_MODEL_PATH}")

shutil.rmtree(EXTRACT_DIR)