import os
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

#3 * 1024 * 1024 * 1024

def initialize_builder(use_fp16=False, use_int8=False, workspace_size=(1<<32)):
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_tactic_sources(trt.TacticSource.CUBLAS_LT)
    config.max_workspace_size = workspace_size

    if builder.platform_has_fast_fp16 and use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if builder.platform_has_fast_int8 and use_int8:
        config.set_flag(trt.BuilderFlag.INT8)

    return builder, config

def parse_onnx_model(builder, onnx_file_path):
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('❌ Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("✅ Completed parsing ONNX file")
    return network

def set_dynamic_shapes(builder, config, dynamic_shapes):
    if dynamic_shapes:
        print(f"===> Using dynamic shapes: {str(dynamic_shapes)}")
        profile = builder.create_optimization_profile()

        for binding_name, (min_shape, opt_shape, max_shape) in dynamic_shapes.items():
            profile.set_shape(binding_name, min_shape, opt_shape, max_shape)

        config.add_optimization_profile(profile)

def build_and_save_engine(builder, network, config, engine_file_path):
    if os.path.isfile(engine_file_path):
        try:
            os.remove(engine_file_path)
        except Exception as e:
            print(f"Cannot remove existing file: {engine_file_path}. Error: {e}")

    print("Creating TensorRT Engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine:
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)
        print(f"===> Serialized Engine Saved at: {engine_file_path}")
    else:
        print("❌ Failed to build engine")

def main():
    onnx_file_path = "model_convert/wav2lip_gan.onnx"
    engine_file_path = "model_convert/wav2lip_gan_FP32_200_4GB.trt"
    dynamic_shapes = {
        "input1": ((1, 1, 80, 16), (2, 1, 80, 16), (200, 1, 80, 16)), 
        "input2": ((1, 6, 96, 96), (2, 6, 96, 96), (200, 6, 96, 96)) 
    }

    builder, config = initialize_builder(use_fp16=False, use_int8=False)
    network = parse_onnx_model(builder, onnx_file_path)
    if network:
        set_dynamic_shapes(builder, config, dynamic_shapes)
        build_and_save_engine(builder, network, config, engine_file_path)

if __name__ == "__main__":
    main()
