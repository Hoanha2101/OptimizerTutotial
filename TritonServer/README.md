# Triton Server Tutorial

Now, I don't have Ubuntu engine to run linux kernel, I just have a window computer. 

So I set up environment in `WSL2`.

## Server

#### 1. Pull Triton Server Docker images

```bash
docker pull nvcr.io/nvidia/tritonserver:24.09-py3
```

#### 2. Set up models folder 

I have a model classification, classify 2 object

[`good_pepsi`, `error_pepsi`]

Link public model: [model](https://drive.google.com/file/d/1E91K1ILMNb2AQ4LN5woIOrhQhWnL9SCw/view?usp=sharing)

```bash
cd /mnt/d 
mkdir model_triton
cd model_triton
mkdir -p ./models/pepsi_onnx/1
```
#### 3. Set up model, config

```bash
mv model.onnx ./models/pepsi_onnx/1
```
Create config.pbtxt

Check  `input name` and `output name` in [netron.app](https://netron.app/)
```bash
name: "pepsi_onnx"
backend: "onnxruntime"
max_batch_size: 0
input: [
  {
    name: "images",
    data_type: TYPE_FP32,
    dims: [ 1, 3, 270, 90]
  }
]
output: [
  {
    name: "output",
    data_type: TYPE_FP32,
    dims: [ 1, 2 ]
  }
]
```

We have folder map:
```bash
model_triton
    |----models
        |-----pepsi_onnx
                |-----1
                |    |-----model.onnx
                |-----config.pbtxt
        |...
```
Note: 
+ pepsi_onnx: is name model, that verify model used.
+ 1: version, in project, there are many versions: 1, 2,..
+ config.pbtxt: config for each model.

#### 4. Run Triton Server

Run container
```bash
docker run --gpus all --rm -it \
  --network bridge \
  -v $PWD:/mnt \
  --name triton-server \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  nvcr.io/nvidia/tritonserver:24.09-py3
```

Run model infer server
```bash
tritonserver --model-repository=/mnt/models
```
If successful run, the screen will display:

```bash
root@4c7e7d3027a0:/opt/tritonserver# tritonserver --model-repository=/mnt/model_triton/models
I1004 15:52:15.206197 195 pinned_memory_manager.cc:277] "Pinned memory pool is created at '0x204c00000' with size 268435456"
I1004 15:52:15.206313 195 cuda_memory_manager.cc:107] "CUDA memory pool is created on device 0 with size 67108864"
I1004 15:52:15.262215 195 model_lifecycle.cc:472] "loading: pepsi_onnx:1"
I1004 15:52:15.304265 195 onnxruntime.cc:2875] "TRITONBACKEND_Initialize: onnxruntime"
I1004 15:52:15.304854 195 onnxruntime.cc:2885] "Triton TRITONBACKEND API version: 1.19"
I1004 15:52:15.304918 195 onnxruntime.cc:2891] "'onnxruntime' TRITONBACKEND API version: 1.19"
I1004 15:52:15.304936 195 onnxruntime.cc:2921] "backend configuration:\n{\"cmdline\":{\"auto-complete-config\":\"true\",\"backend-directory\":\"/opt/tritonserver/backends\",\"min-compute-capability\":\"6.000000\",\"default-max-batch-size\":\"4\"}}"
I1004 15:52:15.335609 195 onnxruntime.cc:2986] "TRITONBACKEND_ModelInitialize: pepsi_onnx (version 1)"
I1004 15:52:15.339013 195 onnxruntime.cc:984] "skipping model configuration auto-complete for 'pepsi_onnx': inputs and outputs already specified"
I1004 15:52:15.342682 195 onnxruntime.cc:3051] "TRITONBACKEND_ModelInstanceInitialize: pepsi_onnx_0 (GPU device 0)"
I1004 15:52:17.657312 195 model_lifecycle.cc:839] "successfully loaded 'pepsi_onnx'"
I1004 15:52:17.657497 195 server.cc:604]
+------------------+------+
| Repository Agent | Path |
+------------------+------+
+------------------+------+

I1004 15:52:17.657561 195 server.cc:631]
+-------------+--------------------------------------------------+--------------------------------------------------+
| Backend     | Path                                             | Config                                           |
+-------------+--------------------------------------------------+--------------------------------------------------+
| onnxruntime | /opt/tritonserver/backends/onnxruntime/libtriton | {"cmdline":{"auto-complete-config":"true","backe |
|             | _onnxruntime.so                                  | nd-directory":"/opt/tritonserver/backends","min- |
|             |                                                  | compute-capability":"6.000000","default-max-batc |
|             |                                                  | h-size":"4"}}                                    |
|             |                                                  |                                                  |
+-------------+--------------------------------------------------+--------------------------------------------------+

I1004 15:52:17.657661 195 server.cc:674]
+------------+---------+--------+
| Model      | Version | Status |
+------------+---------+--------+
| pepsi_onnx | 1       | READY  |
+------------+---------+--------+

I1004 15:52:17.684909 195 metrics.cc:877] "Collecting metrics for GPU 0: NVIDIA GeForce RTX 3050 Laptop GPU"
I1004 15:52:17.688115 195 metrics.cc:770] "Collecting CPU metrics"
I1004 15:52:17.688321 195 tritonserver.cc:2598]
+----------------------------------+----------------------------------------------------------------------------------+
| Option                           | Value                                                                            |
+----------------------------------+----------------------------------------------------------------------------------+
| server_id                        | triton                                                                           |
| server_version                   | 2.50.0                                                                           |
| server_extensions                | classification sequence model_repository model_repository(unload_dependents) sch |
|                                  | edule_policy model_configuration system_shared_memory cuda_shared_memory binary_ |
|                                  | tensor_data parameters statistics trace logging                                  |
| model_repository_path[0]         | /mnt/model_triton/models                                                         |
| model_control_mode               | MODE_NONE                                                                        |
| strict_model_config              | 0                                                                                |
| model_config_name                |                                                                                  |
| rate_limit                       | OFF                                                                              |
| pinned_memory_pool_byte_size     | 268435456                                                                        |
| cuda_memory_pool_byte_size{0}    | 67108864                                                                         |
| min_supported_compute_capability | 6.0                                                                              |
| strict_readiness                 | 1                                                                                |
| exit_timeout                     | 30                                                                               |
| cache_enabled                    | 0                                                                                |
+----------------------------------+----------------------------------------------------------------------------------+

I1004 15:52:17.708600 195 grpc_server.cc:2558] "Started GRPCInferenceService at 0.0.0.0:8001"
I1004 15:52:17.708914 195 http_server.cc:4704] "Started HTTPService at 0.0.0.0:8000"
I1004 15:52:17.751774 195 http_server.cc:362] "Started Metrics Service at 0.0.0.0:8002"
W1004 15:52:18.689780 195 metrics.cc:631] "Unable to get power limit for GPU 0. Status:Success, value:0.000000"
W1004 15:52:19.693602 195 metrics.cc:631] "Unable to get power limit for GPU 0. Status:Success, value:0.000000"
W1004 15:52:20.694133 195 metrics.cc:631] "Unable to get power limit for GPU 0. Status:Success, value:0.000000"
```

## Client

Install library

```bash
pip install nvidia-pyindex
pip install tritonclient[all]
```

Check Server Status
```bash
# Check server status
curl -v http://localhost:8000/v2/health/live

# Check model information pepsi_onnx
curl -v http://localhost:8000/v2/models/pepsi_onnx

# Check all models
curl -v http://localhost:8000/v2/models

```

Run test
```bash
python Infer_simple.py
```

## Python Backend

The Triton backend for Python. The goal of Python backend is to let you serve models written in Python by Triton Inference Server without having to write any C++ code.

Model Directory Structure


