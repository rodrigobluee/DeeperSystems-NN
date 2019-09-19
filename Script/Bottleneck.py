import numpy
import os

from tensorflow.python.platform import gfile


diretorio_bottleneck= 'Bottleneck'


def Cria_Novo_Arquivo(sessao, path_imagem, tensor_dados_jpeg, tensor_decodificador_imagem,
                                   tensor_entrada_redimensionado,bottleneck_tensor):

    dados_imagem=gfile.FastGFile(path_imagem,'rb').read()
    
    # Decodifica a imagem JPEG e a redimensiona.
    valores_redimensionados = sessao.run(tensor_decodificador_imagem,{tensor_dados_jpeg:dados_imagem})
    
    # Execucao pela rede de reconhecimento.
    valores_bottleneck=sessao.run(bottleneck_tensor,{tensor_entrada_redimensionado:valores_redimensionados})
    valores_bottleneck=numpy.squeeze(valores_bottleneck)
          
    # Escreve os valores bottleneck no arquivo bottleneck
    valores_bottleneck_str = ','.join(str(valor) for valor in valores_bottleneck)
    path_arquivo_bottleneck = os.path.join(diretorio_bottleneck,str(os.path.basename(path_imagem))+'.txt')
           
    with open(path_arquivo_bottleneck,'w') as arquivo_bottleneck:
        arquivo_bottleneck.write(valores_bottleneck_str)
        
                                       
def Refaz_Todo_Bottleneck(sessao,lista_imagens,tensor_dados_jpeg, tensor_decodificador_imagem,
                                         tensor_entrada_redimensionado,bottleneck_tensor):
                                         
    if not os.path.exists(diretorio_bottleneck):
        os.makedirs(diretorio_bottleneck)
    
    for path_imagem in lista_imagens:
        Cria_Novo_Arquivo(sessao, path_imagem, tensor_dados_jpeg, tensor_decodificador_imagem,
                                   tensor_entrada_redimensionado,bottleneck_tensor)
                                   

def Pega_Sumarios_Imagem(image_name):
    with open(os.path.join(diretorio_bottleneck, image_name+'.txt'),'r') as arquivo_bottleneck:
            valores_bottleneck_str = arquivo_bottleneck.read()
    
    return [float(valores) for valores in valores_bottleneck_str.split(',')]
    