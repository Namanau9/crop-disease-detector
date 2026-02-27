import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

IMG_SIZE = 224
BATCH_SIZE = 32

def load_data():
    """Reads and splits the PlantVillage dataset."""
    print("Loading PlantVillage dataset...")
    # PlantVillage is available in TFDS
    (train_ds, val_ds, test_ds), ds_info = tfds.load(
        'plant_village',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )
    
    num_classes = ds_info.features['label'].num_classes
    class_names = ds_info.features['label'].names
    
    return train_ds, val_ds, test_ds, num_classes, class_names

def preprocess_image(image, label):
    """Resizes and normalizes images for EfficientNetB0."""
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    # EfficientNetB0 expects input in range [0, 255] if using the pre-built model's preprocessing,
    # or the preprocessing can be included in the model itself.
    # For simplicity, we'll keep it as float32 but keep the values in [0, 255] as EfficientNet's internal 
    # Rescaling layer usually handles it if included, otherwise we'll add it in the model.
    return tf.cast(image, tf.float32), label

def get_data_augmentation():
    """Returns a sequential model for data augmentation."""
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])
    return data_augmentation

def prepare_datasets(train_ds, val_ds, test_ds):
    """Optimizes datasets for performance."""
    
    # Preprocessing
    train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle, batch, and augment
    train_ds = train_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Add augmentation as a map function to the training set
    data_aug = get_data_augmentation()
    train_ds = train_ds.map(lambda x, y: (data_aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    val_ds = val_ds.batch(BATCH_SIZE).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    # Test loading and visualization
    train_ds, val_ds, test_ds, num_classes, class_names = load_data()
    train_ds_ready, val_ds_ready, test_ds_ready = prepare_datasets(train_ds, val_ds, test_ds)
    
    print(f"Number of classes: {num_classes}")
    
    # Visualize samples
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds_ready.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.tight_layout()
    plt.show()
