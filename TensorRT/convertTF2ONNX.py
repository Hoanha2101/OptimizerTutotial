import tf2onnx
import onnx 
from library import *

model_cls_bottle = load_model("model_set/classification/org/model_combine.h5")
input_signature_1 = [tf.TensorSpec([1,270, 90,3], tf.float32, name='x')]
onnx_model_1, _ = tf2onnx.convert.from_keras(model_cls_bottle, input_signature_1, opset=13)
onnx.save(onnx_model_1, "model_set/classification/onnx/model_combine.onnx")