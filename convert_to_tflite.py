import tensorflow as tf
import os
import traceback

# Path to your trained model
model_path = os.path.join("GRU_model", "signlanguage_model.h5")

print("Loading model from:", model_path)

# Compatibility: allow loading older/specialized Dense configs
from tensorflow.keras.layers import Dense as KerasDense

class DenseCompat(KerasDense):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)

try:
    # Load Keras model with compatibility wrapper in case saved model contains
    # extra Dense kwargs (e.g. `quantization_config`).
    model = tf.keras.models.load_model(model_path, custom_objects={"Dense": DenseCompat})

    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # GRU and TensorList ops may require Select TF ops; allow them and
    # disable experimental lowering of tensor list ops so conversion succeeds.
    try:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter._experimental_lower_tensor_list_ops = False
    except Exception:
        pass

    tflite_model = converter.convert()

    # Save the TFLite model
    with open("isl_model.tflite", "wb") as f:
        f.write(tflite_model)

    print("✅ SUCCESS: isl_model.tflite created in project root")
except Exception:
    print("Conversion failed — see traceback below:")
    traceback.print_exc()
