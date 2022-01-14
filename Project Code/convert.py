# coding=utf-8
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.keras.models import load_model


def h5_to_pb(h5_load_path, pb_save_path):
    model = load_model(h5_load_path, compile=False)
    model.summary()
    full_model = tf.function(lambda Input: model(Input))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./",
                      name=pb_save_path,
                      as_text=False)


h5_load_path = './models-l/mobilenet_v3.hdf5'    # 原来hdf5文件保存路径
pb_save_path = './models-l/mobilenet_v3.pb'      # 转换后的pb文件保存路径

h5_to_pb(h5_load_path, pb_save_path)
