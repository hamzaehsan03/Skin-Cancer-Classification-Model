import tensorflow as tf
from keras.applications import VGG19
from tensorflow.python.keras.models import Model 
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator

# Load VGG19 model trained on ImageNet
# Set include top layer as false to customise the output layer for the classification task
# Freeze the weights of the model during initial training

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False


x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(base_model.input, x)