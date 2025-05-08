import numpy as np
import tritonclient.http as httpclient
import cv2
import torchvision.transforms as transforms


labels = {
    0: "good",
    1: "error"
}

# Model name and version (if needed)
model_name = "image_preprocess_model"
model_version = "1"  # Can be left empty if there's only one version

# Connect to Triton Server
triton_client = httpclient.InferenceServerClient(url="localhost:8000")

# Prepare input data
# Adjust the shape and data type according to your model
image_path = "images/g.jpg"
image = cv2.imread(image_path).astype(np.uint8)
input_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_data= cv2.resize(input_data, (90, 270))

# Create input tensor
inputs = []
inputs.append(httpclient.InferInput("IMAGE", input_data.shape, "UINT8"))
inputs[0].set_data_from_numpy(input_data)

# Request output
outputs = []
outputs.append(httpclient.InferRequestedOutput("PREPROCESSED_OUTPUT"))

# Send inference request
results = triton_client.infer(model_name=model_name,
                              inputs=inputs,
                              outputs=outputs,
                              model_version=model_version)

# Get the result
output_data = results.as_numpy("PREPROCESSED_OUTPUT")
print("Inference Result:", output_data.shape)
