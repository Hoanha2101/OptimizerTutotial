import torch
import torch.nn as nn

#model 
class ModelCNN(nn.Module):
    def __init__(self, classes):
        super(ModelCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.maxpool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.maxpool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128*64*19, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dr = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, classes)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.maxpool1(out)
        out = self.bn1(out)
        
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.maxpool2(out)
        out = self.bn2(out)
        
        out = out.view(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.bn3(out)
        out = self.dr(out)
        out = self.fc2(out)
        
        out = torch.softmax(out, dim = 1)
        
        return out
        

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

def convert_pytorch2onnx(model, input_samples, path_onnx, mode='float32bit', device='cuda'):
    if mode == 'float16bit':
        print("Converting model and inputs to float16")
        model = model.half()  # Convert model to float16
        input_samples = input_samples.half()  # Convert input samples to float16
    elif mode == 'float32bit':
        print("Converting model and inputs to float32")
        model = model.float()  # Convert model to float32
        input_samples = input_samples.float()  # Convert input samples to float32
    
    model.to(device)
    model.eval()
    input_samples = input_samples.to(device)
    
    torch.onnx.export(
        model,  # The model
        input_samples,  # Input tensor with desired size
        path_onnx,  # Path to save the ONNX file
        verbose=False,  # Whether to print the process
        opset_version=12,  # ONNX opset version
        do_constant_folding=True,  # Whether to do constant folding optimization
        input_names=['images'],  # Model input names
        output_names=['output'],  # Model output names
    )

# Load the model
model = torch.load("models/model_FP32.pth", map_location=device)
input_samples = torch.randn(1, 3, 270, 90)  # Example input tensor
path_onnx = "models/model_FP32.onnx"

# Convert the model to ONNX
convert_pytorch2onnx(model, input_samples, path_onnx, mode='float32bit', device=device)
