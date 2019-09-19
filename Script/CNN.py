import tensorflow
import os

from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

    
def Cria_Grafo_Modelo():
    model_info={
        'bottleneck_tensor_name':  'pool_3/_reshape:0',
        'bottleneck_tensor_size': 2048,
        'input_width': 299,
        'input_height': 299,
        'input_depth': 3,
        'resized_input_tensor_name':  'Mul:0',
        'name_input_layer':'Mul',
        'name_output_layer':'Tensor_Final',
        'path_modelo_arquitetura': 'inception_v3.pb',
        'input_mean': 128,
        'input_std': 128,
    }  
    
    with tensorflow.Graph().as_default() as graph:   
        with gfile.FastGFile(model_info['path_modelo_arquitetura'], 'rb') as modelo:    
            graph_def = tensorflow.GraphDef()
            graph_def.ParseFromString(modelo.read())
            
            bottleneck_tensor, resized_input_tensor = (tensorflow.import_graph_def(
                graph_def,
                name='',
                return_elements=[
                    model_info['bottleneck_tensor_name'],
                    model_info['resized_input_tensor_name'],
                ]))
   
    return graph, bottleneck_tensor, resized_input_tensor, model_info
    