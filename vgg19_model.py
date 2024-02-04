import tensorflow as tf
import datetime
from keras.applications import VGG19
from keras.models import Model 
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator

# Load VGG19 model trained on ImageNet
# Set include top layer as false to customise the output layer for the classification task
# Freeze the weights of the model during initial training

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.summary()
for layer in base_model.layers:
    layer.trainable = False

#import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Add custom layers to the model
# Flatten the model to a single vector
# Add a connected layer with relu activation
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(base_model.input, x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_data = ImageDataGenerator(1./255)
# train_data = ImageDataGenerator(
#     1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

train_gen = train_data.flow_from_directory(
    'data/train',
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'binary'
)

validation_data = ImageDataGenerator(1./255)
validation_gen = validation_data.flow_from_directory(
    'data/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
callbacks = [
    EarlyStopping(patience=5, monitor='val_loss'),
    ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    TensorBoard(log_dir=log_dir, histogram_freq=1)
]

history = model.fit(
    train_gen,
    epochs=10, 
    validation_data=validation_gen,
    callbacks=callbacks
)
