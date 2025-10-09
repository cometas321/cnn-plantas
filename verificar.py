import tensorflow as tf
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs disponibles:", gpus)
    # Opcional: evitar que TF consuma toda la memoria
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No se detectó GPU. Se ejecutará en CPU.")