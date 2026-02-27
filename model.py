import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0

def build_model(num_classes):
    """Builds the EfficientNetB0 model with custom layers for classification."""
    
    # EfficientNetB0 includes its own rescaling if we use it, 
    # but we resized images to 224x224 in data_loader.py.
    # Note: Keras Applications EfficientNetB0 expects inputs to be floated uint8 [0, 255].
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # Base model with pre-trained weights
    # include_top=False means we exclude the final classification layers
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
    
    # Freeze the pre-trained weights initially
    base_model.trainable = False
    
    # Add custom head
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Final classification layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs, name="EfficientNetB0_PlantVillage")
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Test model building
    NUM_CLASSES = 38  # Default for PlantVillage
    model = build_model(NUM_CLASSES)
    model.summary()
