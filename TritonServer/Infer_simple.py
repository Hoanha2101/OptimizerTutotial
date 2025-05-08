import numpy as np
import tritonclient.http as httpclient
import cv2
import torchvision.transforms as transforms

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

labels = {
    0: "good",
    1: "error"
}

# Model name and version (if needed)
model_name = "pepsi_onnx"
model_version = "1"  # Can be left empty if there's only one version

# Connect to Triton Server
triton_client = httpclient.InferenceServerClient(url="localhost:8000")

# Prepare input data
# Adjust the shape and data type according to your model
image_path = "images/g.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image, (90, 270))
image_normalized = (image_resized / 255.0 - mean) / std
input_data = np.transpose(image_normalized,(2, 0, 1))
input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

'''
tritonclient.utils.InferenceServerException: input_tensor must be a numpy array
'''

# Create input tensor
inputs = []
inputs.append(httpclient.InferInput("images", input_data.shape, "FP32"))
inputs[0].set_data_from_numpy(input_data)

# Request output
outputs = []
outputs.append(httpclient.InferRequestedOutput("output"))

# Send inference request
results = triton_client.infer(model_name=model_name,
                              inputs=inputs,
                              outputs=outputs,
                              model_version=model_version)

# Get the result
output_data = results.as_numpy("output")
print("Inference Result:", labels[np.argmax(output_data[0])])
