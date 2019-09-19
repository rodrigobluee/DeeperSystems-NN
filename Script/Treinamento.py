import os
import tensorflow
import numpy as np

from random import randrange
from Bottleneck import Pega_Sumarios_Imagem
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat


labels = ['upright', 'rotated_left', 'rotated_right', 'upside_down']
n_epochs = 10000
batch_size = 180

def Adiciona_Parametros_Treinamento(entropia_cruzada_mean):
    with tensorflow.name_scope('train'):
        otimizador=tensorflow.train.GradientDescentOptimizer(0.1)
        passo_treinamento=otimizador.minimize(entropia_cruzada_mean)
    return passo_treinamento
    
    
def Cria_Lista_Treinamento():
    from csv import reader
    
    train_images_path = []
    ground_truths = []
    
    with open('train.truth.csv') as csvfile:
        readCSV = list(reader(csvfile, delimiter=','))
   
    for row in readCSV[1:]:
        train_images_path.append(os.path.join('train',row[0]))
        
        ground_truth = np.zeros(4, dtype=np.float32)
        ground_truth[labels.index(row[1])] = 1.0
        ground_truths.append(ground_truth)
    
    return train_images_path, ground_truths


def Pega_Sumarios_Imagens_Randomicas(lista_imagens, ground_truths):
    sumarios_imagens=[]
    grounds=[]

    total = len(lista_imagens)
    
    for i in range(batch_size):
        indice = randrange(total)
        
        sumarios_imagens.append(Pega_Sumarios_Imagem(os.path.basename(lista_imagens[indice])))
        grounds.append(ground_truths[indice])
               
    return sumarios_imagens, grounds
    
    
def Retreinamento_Por_BatchSize(sessao, lista_imagens, ground_truths, entradas_bottleneck, ground_truth_input, tensor_dados_jpeg, tensor_decodificador_imagem, 
                                                   tensor_entrada_redimensionado, bottleneck_tensor, entropia_cruzada_mean, merged, evaluation_step, prediction):
    
    passo_treinamento = Adiciona_Parametros_Treinamento(entropia_cruzada_mean)
    sumarios_treinamento=[]
    
    for epoca in range(n_epochs-1):
        sumarios_imagens, grounds = Pega_Sumarios_Imagens_Randomicas(lista_imagens, ground_truths)
        
        sumario_epoca,_=sessao.run([merged, passo_treinamento], 
                                                 feed_dict={entradas_bottleneck: sumarios_imagens,
                                                                 ground_truth_input: grounds})
        
        sumarios_treinamento.append(sumario_epoca)
    
    return sumarios_treinamento 
