import tensorflow as tf
from data_loader import load_data, prepare_datasets
from model import build_model
import os

# Hyperparameters
EPOCHS_INITIAL = 10
EPOCHS_FINE_TUNE = 10
LEARNING_RATE_INITIAL = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5

def train():
    # 1. Load and prepare data
    train_ds, val_ds, test_ds, num_classes, class_names = load_data()
    train_ds, val_ds, test_ds = prepare_datasets(train_ds, val_ds, test_ds)
    
    # 2. Build model
    model = build_model(num_classes)
    
    # 3. Callbacks
    checkpoint_path = "checkpoints/crop_classifier.keras"
    os.makedirs("checkpoints", exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
    ]
    
    # 4. Initial training (frozen base)
    print("Starting initial training (frozen base)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_INITIAL,
        callbacks=callbacks
    )
    
    # 5. Fine-tuning (unfreeze top layers of base model)
    print("Starting fine-tuning...")
    # Get the base model (which is the first layer after input since we used Functional API)
    base_model = model.get_layer(index=1)
    base_model.trainable = True
    
    # Freeze all layers except the last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False
        
    # Recompile after changes to trainable status
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE_TUNE,
        callbacks=callbacks
    )
    
    # 6. Evaluation
    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # 7. Save final model
    model.save("crop_disease_classifier_final.keras")
    print("Model saved to crop_disease_classifier_final.keras")

if __name__ == "__main__":
    train()
