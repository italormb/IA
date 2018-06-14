# Part 1 - Building the CNN
input('\n\n                  Pojeto de IA: Indentificação de Pneumonia a partir de radiografia usando Redes Neurais Artificiais (RNA)\n\n                Este programa foi desenvolvido por alunos da graduação na UnB a fim de treinar e testar a rede neural, onde \nessas imagens foram adquiridas no Kaggle.\n\n\nAlunos: \nBreno Alencar\nIgor Aragão\nÍtalo Rodrigo\nMatheus Avena\n\nPressione "Enter" para continuar...')
lr = float(input('Learning rate: '))
epoch = int(input('Épocas: '))
n_train = int(input('Número de imagens para treinar: '))
n_val =int(input('Número de imagens para teste: '))
n_hidden = input('Número de camadas ocultas: ')
hidden = []
for i in range(1, int(n_hidden)+1):
    hidden.append(int(input("Número de nós camada oculta %d: " % i)))
model_path = input('Nome do modelo: ')
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
for i in range(int(n_hidden)):
    classifier.add(Dense(output_dim = hidden[i], activation = 'relu'))
    
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

from keras import optimizers
sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
classifier.compile(loss='mean_squared_error', optimizer=sgd, metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = False)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

progress = classifier.fit_generator(training_set,
                         samples_per_epoch = n_train,
                         nb_epoch = epoch,
                         validation_data = test_set,
                         nb_val_samples = n_val)

# Save model
from keras.models import load_model

classifier.save(model_path + '.h5')  # creates a HDF5 file 'my_model.h5'

# summarize history for accuracy
import matplotlib.pyplot as plt
from matplotlib import interactive
plt.figure(1)
plt.plot(progress.history['acc'])
plt.plot(progress.history['test_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
interactive(True)
plt.show()
# summarize history for loss
plt.figure(2)
plt.plot(progress.history['loss'])
plt.plot(progress.history['test_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
interactive(True)
plt.show()