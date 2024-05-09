import tensorflow as tf
import datetime
from keras.applications import VGG19
from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, recall_score, precision_score, accuracy_score
import seaborn as sns
import pandas as pd

#tf.config.set_visible_devices([], 'GPU')
# Load VGG19 model trained on ImageNet
# Set include top layer as false to customise the output layer for the classification task
# Freeze the weights of the model during initial training

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.summary()
for layer in base_model.layers:
    layer.trainable = False

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
train_data = ImageDataGenerator(
    1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

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
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, verbose=1, min_lr=0.00001)
]

history = model.fit(
    train_gen,
    epochs=400, 
    validation_data=validation_gen,
    class_weight = class_weight_dict,
    callbacks=callbacks
    
)

# checkpoint_path = 'model.11-0.12.h5'
# model = load_model(checkpoint_path)

def evaluate_model(model, generator):
    true_labels = []
    predictions = []
    
    for _ in range(len(generator)):
        imgs, labels = next(generator)
        preds = model.predict(imgs)

        true_labels.extend(labels)
        predictions.extend(preds)
    
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    
    return true_labels, predictions

# Evaluate the model on the entire validation set
true_labels, predictions = evaluate_model(model, validation_gen)

predictions = (predictions > 0.5).astype(int)

# ROC Curve
fpr, tpr, thresholds = roc_curve(true_labels, predictions)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Heatmap for Classification Report
report = classification_report(true_labels, predictions, output_dict=True)
df_report = pd.DataFrame(report).transpose()
sns.heatmap(df_report.iloc[:-1, :].drop(['support'], axis=1), annot=True, cmap="Blues")
plt.title('Classification Report Heatmap')
plt.show()

# Performance Metrics
sensitivity = recall_score(true_labels, predictions)
specificity = recall_score(true_labels, predictions, pos_label=0)
precision = precision_score(true_labels, predictions)
accuracy = accuracy_score(true_labels, predictions)

print("Sensitivity: {:.2f}".format(sensitivity))
print("Specificity: {:.2f}".format(specificity))
print("Precision: {:.2f}".format(precision))
print("Accuracy: {:.2f}".format(accuracy))