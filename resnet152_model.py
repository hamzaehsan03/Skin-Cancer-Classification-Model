import tensorflow as tf
import datetime
from keras.applications import ResNet152
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator

# Load default resnet model
base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.summary()
for layer in base_model.layers:
    layer.trainable = False


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)


model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


train_data = ImageDataGenerator(1./255)


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
