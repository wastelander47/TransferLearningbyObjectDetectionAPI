import tensorflow as tf
from tensorflow.python.platform import gfile

def load_graph(frozen_graph_filename):
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        return graph_def
tf.import_graph_def(load_graph("frozen_inference_graph.pb"))