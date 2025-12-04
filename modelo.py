import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing import image
from sklearn.utils import class_weight

# ============ CONFIGURACIÓN ============
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
DATA_DIR = "flores"

# ============ DATA AUGMENTATION (aplicado al dataset) ============
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.15, 0.15),
    layers.RandomBrightness(0.2),
])

def augment(image, label):
    return data_augmentation(image, training=True), label

# ============ CARGAR DATOS ============
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

# ============ CALCULAR PESOS DE CLASE ============
train_labels = []
for _, labels in train_ds:
    train_labels.extend(labels.numpy())
train_labels = np.array(train_labels)

class_weights_array = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

# ============ APLICAR AUGMENTATION Y OPTIMIZACIÓN ============
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ============ MODELO (SIN data augmentation integrado) ============
def build_model(input_shape=(224,224,3), num_classes=5):
    inputs = layers.Input(shape=input_shape)
    
    # Normalización
    x = layers.Rescaling(1./127.5, offset=-1)(inputs)
    
    # Modelo preentrenado
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model, base_model

# Crear modelo
model, base_model = build_model(input_shape=(224,224,3), num_classes=num_classes)

# ============ COMPILAR ============
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# ============ ENTRENAR FASE 1 ============
print("\n=== ENTRENANDO FASE 1 (Transfer Learning) ===")

checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-7, verbose=1)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weights
)

# ============ FINE-TUNING FASE 2 ============
print("\n=== ENTRENANDO FASE 2 (Fine-Tuning) ===")
base_model.trainable = True

# Congelar las primeras capas
for layer in base_model.layers[:-30]:
    layer.trainable = False

print(f"Capas entrenables: {sum([1 for layer in base_model.layers if layer.trainable])}")

# Recompilar con learning rate más bajo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

checkpoint2 = ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
early_stop2 = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr2 = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-7, verbose=1)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[checkpoint2, early_stop2, reduce_lr2],
    class_weight=class_weights
)

# ============ GUARDAR MODELO ============
model.save("flower_model_final.h5")
print("\n✓ Modelo guardado: flower_model_final.h5")

# ============ GRÁFICAS ============
# Combinar historiales
for key in history1.history.keys():
    history1.history[key].extend(history2.history[key])

hist = history1.history
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
plt.tight_layout()
plt.show()

# ============ EVALUACIÓN ============
print("\n=== EVALUACIÓN TEST SET ===")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}")

# ============ MATRIZ DE CONFUSIÓN ============
y_true = []
y_pred = []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Matriz de Confusión')
plt.colorbar()
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)
plt.ylabel('True label')
plt.xlabel('Predicted label')

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.tight_layout()
plt.show()

# ============ FUNCIÓN DE PREDICCIÓN ============
def predict_image(model, img_path, class_names, image_size=(224,224)):
    img = image.load_img(img_path, target_size=image_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    preds = model.predict(x, verbose=0)[0]
    top_idx = np.argmax(preds)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Predicción: {class_names[top_idx]}\nConfianza: {preds[top_idx]:.2%}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    sorted_idx = np.argsort(preds)[::-1]
    colors = ['green' if i == top_idx else 'skyblue' for i in sorted_idx]
    plt.barh([class_names[i] for i in sorted_idx], [preds[i] for i in sorted_idx], color=colors)
    plt.xlabel('Probabilidad')
    plt.title('Probabilidades')
    plt.xlim([0, 1])
    plt.tight_layout()
    plt.show()
    
    return top_idx, preds[top_idx], preds

# ============ CARGAR Y PROBAR ============
print("\n=== CARGANDO MODELO ===")
inference_model = tf.keras.models.load_model("flower_model_final.h5")

print("\n=== PREDICCIÓN ===")
idx, prob, all_probs = predict_image(inference_model, "probar.jpg", class_names)

print(f"\nClase predicha: {class_names[idx]}")
print(f"Confianza: {prob:.2%}")
print("\nTodas las probabilidades:")
for cls, p in zip(class_names, all_probs):
    print(f"  {cls}: {p:.2%}")