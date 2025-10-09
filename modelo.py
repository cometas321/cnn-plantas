import sys
print(sys.executable)

import tensorflow as tf

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
DATA_DIR = "flores"  # con subcarpetas train, val, test

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR + '/train',
    labels='inferred',
    label_mode='int',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR + '/val',
    labels='inferred',
    label_mode='int',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR + '/test',
    labels='inferred',
    label_mode='int',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Clases:", class_names)


# Normalización
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data augmentation

from tensorflow.keras import layers

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

## Construcción del modelo CNN desde cero

from tensorflow.keras import layers, models

def build_cnn(input_shape=(128,128,3), num_classes=5, dropout_rate=0.5):
    inputs = layers.Input(shape=input_shape)

    # Normalización simple (0-255 -> 0-1)
    x = layers.Rescaling(1./255)(inputs)

    # Aumento de datos (solo activo en entrenamiento)
    x = data_augmentation(x)

    # Bloque Conv 1
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Bloque Conv 2
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Bloque Conv 3
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Red densa final
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='flower_cnn')
    return model

model = build_cnn(input_shape=(128,128,3), num_classes=num_classes)
model.summary()


# Compilación del modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

callbacks = [
    ModelCheckpoint("best_flower_model.h5", save_best_only=True, monitor='val_loss'),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

EPOCHS = 30
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Gráficos de pérdida y precisión (train vs validation)
import matplotlib.pyplot as plt

def plot_history(history):
    hist = history.history
    epochs = range(1, len(hist['loss'])+1)

    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, hist['loss'], label='train_loss')
    plt.plot(epochs, hist['val_loss'], label='val_loss')
    plt.title('Pérdida')
    plt.xlabel('Época')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, hist['accuracy'], label='train_acc')
    plt.plot(epochs, hist['val_accuracy'], label='val_acc')
    plt.title('Precisión')
    plt.xlabel('Época')
    plt.legend()
    plt.show()

plot_history(history)


# ##### Evaluación final en test set
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

#Matriz de confusión y reporte por clase

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Obtener etiquetas verdaderas y predichas
y_true = []
y_pred = []
for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred)
print("Classification report:\n", classification_report(y_true, y_pred, target_names=class_names))
print("Confusion matrix:\n", cm)

# Mostrar matriz como imagen
plt.figure(figsize=(8,6))
plt.imshow(cm, interpolation='nearest')
plt.title('Matriz de confusión')
plt.colorbar()
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
