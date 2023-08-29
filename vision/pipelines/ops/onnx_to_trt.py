import os
import tensorrt as trt

model_path = "/home/yotam/FruitSpec/weights/yolov8_aug23/best.onnx"

def convert_onnx_to_trt(model_path, output_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(model_path)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        raise Exception

    config = builder.create_builder_config()
    serialized_engine = builder.build_serialized_network(network, config)

    with open(os.path.join(output_path, "model.engine"), "wb") as f:
        f.write(serialized_engine)


def load_engine(engine_path, logger=None):

    if logger is None:
        logger = trt.Logger(trt.Logger.WARNING)

    with open(engine_path, "rb") as f:
        serialized_engine = f.read()

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    return engine




