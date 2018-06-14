# Save model
from keras.models import load_model
input('\n\n                  Pojeto de IA: Indentificação de Pneumonia a partir de radiografia usando Redes Neurais Artificiais (RNA)\n\n                Este programa foi desenvolvido por alunos da graduação da UnB a fim de diagnosticar Pneumonia através de radiografia, onde \nessas imagens foram adquiridas no Kaggle.\n\n\nAlunos: \nBreno Alencar\nIgor Aragão\nÍtalo Rodrigo\nMatheus Avena\n\nPressione "Enter" para continuar...')

model_path = input('Nome do modelo: ')

# identical to the previous one
classifier = load_model(model_path + '.h5')

# Testing results

import numpy as np
from keras.preprocessing import image

img_path = input('Nome da imagem: ')

# Prever em uma imagem
img = image.load_img(img_path, target_size = (64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x/225
images = np.vstack([x])
classes = classifier.predict_classes(images, batch_size=10)
if int(classes) == 0:
    print("\nResultado: Normal\n")
elif int(classes) == 1:
    print("\nResultado: Pneumonia\n")