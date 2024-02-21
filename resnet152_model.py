import tensorflow as tf
import datetime
from keras.applications import ResNet152
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import numpy as np


# Load default resnet model
base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.summary()
# for layer in base_model.layers:
#     layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)



model = Model(inputs=base_model.input, outputs=predictions)
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
    batch_size = 16,
    class_mode = 'binary'
)

validation_data = ImageDataGenerator(1./255)
validation_gen = validation_data.flow_from_directory(
    'data/validation',
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)

class_weight_dict = dict(enumerate(class_weights))

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
callbacks = [
    EarlyStopping(patience=5, monitor='val_loss'),
    ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    TensorBoard(log_dir=log_dir, histogram_freq=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
]

history = model.fit(
    train_gen,
    epochs=10, 
    validation_data=validation_gen,
    callbacks=callbacks,
    class_weight=class_weight_dict
)
