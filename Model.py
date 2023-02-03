import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import keras
import tensorflow as tf




initial_count = 0
dir = "/home/roberto/images/Todas"
X_train = []
os.listdir()
for file in os.listdir():
    # Leer la imagen y convertirla a una forma compatible con el modelo
    img = cv2.imread('/home/roberto/images/Todas/'+ file)
    print(img)
    # img = cv2.resize(img, (128, 128))
    # img = np.array(img) / 255.0
    # X_train.append(img)

X_train = np.array(X_train)


# Leer el archivo CSV
df = pd.read_csv('./Etiquetas_ISIC.csv')

# Obtener las etiquetas
y_train = df['bening'].values

# Convertir las etiquetas a una codificación one-hot
y_train = keras.utils.to_categorical(y_train, num_classes=3)

print(y_train)