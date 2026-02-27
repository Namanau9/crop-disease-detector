import tensorflow as tf
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

def load_model(model_path):
    """Loads a pre-trained Keras model."""
    print(f"Loading model from {model_path}...")
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    """Loads and preprocesses an image for inference."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0) # Batch size 1
    # Note: Preprocessing layers (if any) are part of the model usually,
    # or we handle scaling here. From our model.py, we only scaled in data_loader.
    # We should add a scaling layer to the model or scale it here.
    return img

def predict_crop(image_path, model_path='crop_disease_classifier_final.keras'):
    """Predicts the disease of a crop from an image."""
    model = load_model(model_path)
    
    # Needs class names - we can extract them from tfds info or provide them.
    # For now, let's assume the user has the class names or we output the index.
    
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    print(f"Predicted Class Index: {predicted_class_idx}")
    print(f"Confidence: {confidence:.2f}")
    
    # Visualization
    img_display = cv2.imread(image_path)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    plt.imshow(img_display)
    plt.title(f"Prediction: Index {predicted_class_idx} ({confidence:.2f})")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [model_path]")
    else:
        img_p = sys.argv[1]
        model_p = sys.argv[2] if len(sys.argv) > 2 else 'crop_disease_classifier_final.keras'
        try:
            predict_crop(img_p, model_p)
        except Exception as e:
            print(f"Error: {e}")
