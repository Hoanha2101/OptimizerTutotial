from __init__ import TensorrtBase
import pycuda.driver as cuda
import pycuda.autoinit

input_names = ['input1', 'input2']
output_names = ['output']
batch = 128



def INFER_TRT(ip1, ip2):
    
    net = TensorrtBase("model_convert/wav2lip_gan.trt",
                input_names=input_names,
                output_names=output_names,
                max_batch_size=ip1.shape[0],
                )


    net.cuda_ctx.push()
    
    inf_in_list = [np.ascontiguousarray(ip1).astype(np.float32), np.ascontiguousarray(ip2).astype(np.float32)]
    inputs, outputs, bindings, stream = net.buffers
    
    binding_shape_map = {
    "input1": ip1.shape,
    "input2": ip2.shape,
    }
    
    if binding_shape_map:
        net.context.set_optimization_profile_async(0, stream.handle)
        for binding_name, shape in binding_shape_map.items():
            net.context.set_input_shape(binding_name, shape)
        for i in range(len(inputs)):
            inputs[i].host = inf_in_list[i]
            cuda.memcpy_htod_async(inputs[i].device, inputs[i].host, stream)
        stream.synchronize()
        net.context.execute_async_v2(
            bindings=bindings,
            stream_handle=stream.handle)  
        for i in range(len(outputs)):
            cuda.memcpy_dtoh_async(outputs[i].host, outputs[i].device, stream)
        stream.synchronize()
        trt_outputs = [out.host.copy() for out in outputs]
    out = trt_outputs[0].reshape((ip1.shape[0],3, 96, 96, ))
    net.cuda_ctx.pop()
    return out