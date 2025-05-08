import torch
import onnx
import onnxscript
from models import Wav2Lip

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print("Loading checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

checkpoint_path = 'checkpoints/wav2lip_gan.pth'
model = load_model(checkpoint_path)


input1 = torch.randn(128, 1, 80, 16).to(device)  
input2 = torch.randn(128, 6, 96, 96).to(device)

torch.onnx.export(model,
                 (input1, input2),
                 "model_convert/wav2lip_gan.onnx",
                 verbose=False,
                 input_names=["input1", "input2"],
                 output_names=["output"],
                 export_params=True,
                 dynamic_axes={
                     'input1': {0: 'batch_size'},  
                     'input2': {0: 'batch_size'}, 
                     'output': {0: 'batch_size'}   
                 },
                 opset_version=12 
                 )

print("Success")
