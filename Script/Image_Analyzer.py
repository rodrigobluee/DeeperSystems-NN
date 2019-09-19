import cv2
import numpy as np
import os
import tensorflow
import sys

from tensorflow.python.platform import gfile

    
def Carrega_Grafo_e_Rotulos(model_info):
    # Carrega o grafo
    graph=tensorflow.Graph()
    graph_def=tensorflow.GraphDef()
    path_graph='trained_model.pb'
    
    with open(path_graph, "rb") as leitor:
        graph_def.ParseFromString(leitor.read())
        
    with graph.as_default():
        tensorflow.import_graph_def(graph_def)
    
    operacao_entrada=  graph.get_operation_by_name('import/'+model_info['name_input_layer'])
    operacao_saida = graph.get_operation_by_name('import/'+model_info['name_output_layer'])
    
    # Carrega os rotulos
    rotulos = []
    path_rotulos = 'labels.txt'
    linhas=tensorflow.gfile.GFile(path_rotulos).readlines()
    
    for rotulo in linhas:
        rotulos.append(rotulo.rstrip())
        
    return graph, operacao_entrada, operacao_saida, rotulos
    

def Rotate_Image(image, indice_top):
    #labels = ['upright', 'rotated_left', 'rotated_right', 'upside_down']
    if indice_top == 0:
        return image
        
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    if indice_top == 1:
        M = cv2.getRotationMatrix2D(center, 270, 1.0)
        image = cv2.warpAffine(image, M, (h, w))
    elif indice_top == 2:
        M = cv2.getRotationMatrix2D(center, 90, 1.0)
        image = cv2.warpAffine(image, M, (h, w))
    elif indice_top == 3:
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        image = cv2.warpAffine(image, M, (h, w))
    
    return image


def Create_Preds_Output(grounds_predicts):
    import csv
    
    name_file = 'test.preds.csv'
    
    with open(name_file, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(grounds_predicts)
        writeFile.close()


def Save_Image(image, path_imagem):
    # Salva Imagem
    name=os.path.basename(path_imagem)
    path_img='TruthImages/'+name
    tensor=tensorflow.placeholder(tensorflow.uint8)
    encode_jpeg = tensorflow.image.encode_jpeg(tensor,format='rgb',quality=100)
    jpeg_bytes = sessao.run(encode_jpeg, feed_dict={tensor: image})
    with open(path_img,'wb') as arq_img:
        arq_img.write(jpeg_bytes)


def Save_NumpyFile(matriz_numpy):
    m = np.array(matriz_numpy)
    np.save('Numpy_file.npy', m) 


def Create_ZIPFile():
    import shutil
    shutil.make_archive('TruthImages', 'zip', 'TruthImages')
    

if __name__ == '__main__':
    model_info={
        'bottleneck_tensor_name':  'pool_3/_reshape:0',
        'bottleneck_tensor_size': 2048,
        'input_width': 299,
        'input_height': 299,
        'input_depth': 3,
        'resized_input_tensor_name': 'Mul:0',
        'name_input_layer':'Mul',
        'name_output_layer':'Tensor_Final',
        'path_modelo_arquitetura': 'inception_v3.pb',
        'input_mean': 128,
        'input_std': 128,
    }  
    
    graph, operacao_entrada, operacao_saida, rotulos = Carrega_Grafo_e_Rotulos(model_info)
   
    # Inicia a sessao
    with tensorflow.Session(graph=graph) as sessao:
        
        grounds_predicts = [['fn','label']]
        labels = ['upright', 'rotated_left', 'rotated_right', 'upside_down']
        matriz_numpy=[]
        
        for path_imagem in gfile.Glob('../test/*'):
        #for path_imagem in range(1):
            #image = cv2.imread('../test/90-68090_1904-08-21_1955.jpg')
            
            face_dims_expander=tensorflow.expand_dims(image,0)
            face_resized=tensorflow.image.resize_bilinear(face_dims_expander,[model_info['input_width'], model_info['input_height']])
            face_normalizada=tensorflow.divide(tensorflow.subtract(face_resized, model_info['input_mean']), model_info['input_std'])
            
            resultado=sessao.run(operacao_saida.outputs[0], {operacao_entrada.outputs[0]:sessao.run(face_normalizada)})
            resultado=tuple(np.squeeze(resultado))
            
            indice_top=int(resultado.index(max(resultado)))

            #Create Ouput           
            image = Rotate_Image(image, indice_top) # imagem numpy
            
            if sys.argv[1] == 'all' or sys.argv[1] == 'numpy_file':
                matriz_numpy.append(np.array(image, dtype=float))
            
            if sys.argv[1] == 'all' or  sys.argv[1] == 'truth_images':
                Save_Image(image, path_imagem)
            
            if sys.argv[1] == 'all' or sys.argv[1] == 'truth_predicts':
                grounds_predicts.append([os.path.basename(path_imagem), labels[indice_top]])
        
        
        if sys.argv[1] == 'all' or sys.argv[1] == 'truth_predicts':        
            Create_Preds_Output(grounds_predicts)
        
        if sys.argv[1] == 'all' or sys.argv[1] == 'numpy_file':
            Save_NumpyFile(matriz_numpy)
         
        if sys.argv[1] == 'all' or  sys.argv[1] == 'truth_images':         
            Create_ZIPFile()