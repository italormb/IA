# Save model
from keras.models import load_model
input('\n\n                  Projeto de IA: Indentificação de Pneumonia a partir de radiografia usando Redes Neurais Artificiais (RNA)\n\n                Este programa foi desenvolvido por alunos da graduação da UnB a fim de verificar o quão bom foi o treino através da matriz de confusão.\n\n\nAlunos: \nBreno Alencar\nIgor Aragão\nÍtalo Rodrigo\nMatheus Avena\n\nPressione "Enter" para continuar...')
model_path = input('Nome do modelo: ')

# identical to the previous one
classifier = load_model(model_path + '.h5')

# Testing results

import os
import numpy as np
from keras.preprocessing import image

y_pred = []
y_true = []
path = 'val/NORMAL'


for filename in os.listdir(path):
    if filename.endswith(".jpeg"): 
        img = image.load_img(path+'/'+filename, target_size = (64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x/225
        images = np.vstack([x])
        classes = classifier.predict_classes(images, batch_size=10)
        y_pred.append(int(classes))
        y_true.append(0)

path = 'val/PNEUMONIA'
for filename in os.listdir(path):
    if filename.endswith(".jpeg"): 
        img = image.load_img(path+'/'+filename, target_size = (64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x/225
        images = np.vstack([x])
        classes = classifier.predict_classes(images, batch_size=10)
        y_pred.append(int(classes))
        y_true.append(1)
        
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
print("Matriz de confusao")
print(cm)
print('Acertou %d imagens normais' % cm[0][0])
print('Acertou %d imagens com pneumonia' % cm[1][1])
print('Errou %d imagens normais' % cm[0][1])
print('Errou %d imagens com pneumonia' % cm[1][0])