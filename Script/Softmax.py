import os
import tensorflow

def Cria_Softmax(sessao, lista_imagens, bottleneck_tensor, model_info):   
    with tensorflow.name_scope('input'):
        entradas_bottleneck=tensorflow.placeholder_with_default(bottleneck_tensor,
                                                                                      shape=[None, model_info['bottleneck_tensor_size']],
                                                                                      name='BottleneckInputPlaceholder')
        
        ground_truth_input = tensorflow.placeholder(tensorflow.float32,[None, 4], name='GroundTruthInput')

    nome_layer='Camada_Final'
    
    with tensorflow.name_scope(nome_layer):
    
        with tensorflow.name_scope('weights'):
            valores_iniciais=tensorflow.truncated_normal([model_info['bottleneck_tensor_size'],4], stddev=0.001)
            pesos_layer=tensorflow.Variable(valores_iniciais, name='Pesos_Finais')
            #variaveis_sumario(pesos_layer)
    
        with tensorflow.name_scope('biases'):
            vieses_layer=tensorflow.Variable(tensorflow.zeros(4), name='Vieses_Finais')
            #variaveis_sumario(vieses_layer)
            
        with tensorflow.name_scope('Wx_plus_b'):
            logits=tensorflow.matmul(entradas_bottleneck,pesos_layer) + vieses_layer
            tensorflow.summary.histogram('Pre Ativacoes', logits)
    
    tensor_final=tensorflow.nn.softmax(logits, name=model_info['name_output_layer'])
    tensorflow.summary.histogram('Ativacoes', tensor_final)
    
    with tensorflow.name_scope('cross_entropy'):
        entropia_cruzada =  tensorflow.nn.softmax_cross_entropy_with_logits(
                                    labels=ground_truth_input, logits=logits)
                                    
        with tensorflow.name_scope('total'):
            entropia_cruzada_mean=tensorflow.reduce_mean(entropia_cruzada)
    
    tensorflow.summary.scalar('cross_entropy',entropia_cruzada_mean)
    
    with tensorflow.name_scope('accuracy'):
        with tensorflow.name_scope('correct_prediction'):
            prediction=tensorflow.argmax(tensor_final,1)
            correct_prediction=tensorflow.equal(prediction, tensorflow.argmax(ground_truth_input, 1))
        with tensorflow.name_scope('accuracy'):
            evaluation_step = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))
            
    tensorflow.summary.scalar('accuracy', evaluation_step)
 
    return (entropia_cruzada_mean, entradas_bottleneck, ground_truth_input, tensor_final, evaluation_step, prediction)
   