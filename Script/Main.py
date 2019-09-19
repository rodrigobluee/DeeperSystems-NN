import tensorflow
import os
import sys

from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

import CNN
import Treinamento
import Softmax
import Bottleneck


def Adiciona_Decodificadores_jpeg(model_info) :
    """
    Adiciona operacoes que executam decodificacao JPEG e redimensionam para o grafico.
    
    Returns:
        Tensores para o noh para alimentar dados JPEG, e a saida do etapas de pre-processamento
    """
    
    jpeg_data = tensorflow.placeholder(tensorflow.string, name='DecodeJPGInput')
    
    decoded_image = tensorflow.image.decode_jpeg(jpeg_data, channels=model_info['input_depth'])
    decoded_image_as_float = tensorflow.cast(decoded_image, dtype=tensorflow.float32)
    decoded_image_4d = tensorflow.expand_dims(decoded_image_as_float, 0)
    
    resize_shape = tensorflow.stack([model_info['input_height'], model_info['input_width']])
    resize_shape_as_int = tensorflow.cast(resize_shape, dtype=tensorflow.int32)
    
    resized_image = tensorflow.image.resize_bilinear(decoded_image_4d,
                                                                     resize_shape_as_int)
                                                                     
    offset_image = tensorflow.subtract(resized_image, model_info['input_mean'])
    mul_image = tensorflow.multiply(offset_image, 1.0 / model_info['input_std'])
    
    return jpeg_data, mul_image  

if __name__ == '__main__':
    # Criando o grafo modelo
    graph, bottleneck_tensor, tensor_entrada_redimensionado,model_info=CNN.Cria_Grafo_Modelo()
    

    #Pega a lista de imagens para treinamento. 
    lista_imagens, ground_truths=Treinamento.Cria_Lista_Treinamento()   
   
    # Iniciando a sessao TensorFlow com o grafo criado
    with tensorflow.Session(graph=graph) as sessao: 
        
        # Criando Decodificadores JPEG
        tensor_dados_jpeg, tensor_decodificador_imagem = Adiciona_Decodificadores_jpeg(model_info)
        
        #Criacao do cache de bottleneck
        if sys.argv[1] == 'rebuild_bottleneck':
            Bottleneck.Refaz_Todo_Bottleneck(sessao,lista_imagens,tensor_dados_jpeg, tensor_decodificador_imagem,
                                                           tensor_entrada_redimensionado,bottleneck_tensor)
        
        #Criacao da camada Softmax
        entropia_cruzada_mean, entradas_bottleneck, ground_truth_input, tensor_final, evaluation_step, prediction = Softmax.Cria_Softmax(sessao, lista_imagens, bottleneck_tensor, model_info)

   
        # Salva os sumarios
        merged = tensorflow.summary.merge_all()
        escritor_treinamento = tensorflow.summary.FileWriter('summaries', sessao.graph)
        
        init=tensorflow.global_variables_initializer()
        sessao.run(init)
           
        # Chama o retreinamento. 
        if sys.argv[1] == 'retrain':
            sumarios_treinamento = Treinamento.Retreinamento_Por_BatchSize(sessao, lista_imagens, ground_truths, entradas_bottleneck, ground_truth_input, tensor_dados_jpeg, tensor_decodificador_imagem, 
                                                                                                   tensor_entrada_redimensionado, bottleneck_tensor, entropia_cruzada_mean, merged, evaluation_step, prediction)

       
            # Salva os sumarios de treinamento e validacao.
            for indice in range(len(sumarios_treinamento)-1):
                escritor_treinamento.add_summary(sumarios_treinamento[indice], indice)
            
            # Salva o grafo do modelo treinado.
            modelo_treinado = graph_util.convert_variables_to_constants(sessao, graph.as_graph_def(), ['Tensor_Final'])
            with gfile.FastGFile('trained_model.pb', 'wb') as escritor:
                escritor.write(modelo_treinado.SerializeToString())

            # Salva os rotulos
            with gfile.FastGFile('labels.txt','w') as escritor:
                escritor.write('\n'.join(['upright', 'rotated_left', 'rotated_right', 'upside_down'])+'\n')
        
            # Escreve no TensorBoard a estrutura do arquitetura. 
            writer = tensorflow.summary.FileWriter('summaries', sessao.graph)
            writer.close()