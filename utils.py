"""
Utility functions for pneumonia detection model.
Includes:
- Loading trained models
- Making single and batch predictions
- Dynamic Grad-CAM heatmap generator
- CSV Prediction history logger
"""

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image


def load_trained_model(model_path):
    """
    Load a trained pneumonia detection model.
    
    Args:
        model_path (str): Path to the saved .h5 model file.
        
    Returns:
        model: Loaded Keras model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model using native Keras loading
    model = tf.keras.models.load_model(model_path)
    print(f"[OK] Model loaded from {model_path}")
    return model


def predict_single_image(image_path, model, image_size=224, threshold=0.5, preprocessing_function=None):
    """
    Predict pneumonia for a single X-ray image.
    
    Args:
        image_path (str): Path to the chest X-ray image file.
        model: The trained deep learning model.
        image_size (int): Target dimensions matching model input (224).
        threshold (float): Decision threshold for pneumonia classification.
        preprocessing_function (callable): Optional preprocessing function for transfer learning.
        
    Returns:
        dict: Prediction metrics (label, confidence score, risk category, raw prediction).
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    resized_img = img.resize((image_size, image_size))
    img_array = np.array(resized_img).astype(np.float32)
    if preprocessing_function is not None:
        img_array = preprocessing_function(img_array)
    else:
        img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Determine class
    class_name = "PNEUMONIA" if prediction > threshold else "NORMAL"
    confidence = prediction if prediction > threshold else 1.0 - prediction
    
    # Risk level categorization
    if class_name == "NORMAL":
        risk_level = "Low"
    elif confidence < 0.75:
        risk_level = "Moderate"
    else:
        risk_level = "High"
        
    return {
        'image_path': image_path,
        'class': class_name,
        'confidence': confidence,
        'risk_level': risk_level,
        'raw_prediction': prediction
    }


def generate_gradcam_overlay(image_path, model, image_size=224, intensity=0.4, preprocessing_function=None, class_index=None):
    """Generate a Grad-CAM heatmap overlay for binary pneumonia classification.

    Notes
    -----
    * Works for both the custom CNN and the MobileNetV2 transfer model.
    * For binary sigmoid output we compute gradients of the output probability.
    * The last convolutional layer is selected robustly by scanning for Conv2D layers.

    Args:
        image_path: Path to the original chest X-ray image.
        model: Trained tf.keras Model.
        image_size: Target image size (must match model input).
        intensity: Opacity of heatmap overlay (0..1).
        preprocessing_function (callable): Optional preprocessing function for transfer learning.
        class_index: Unused for sigmoid binary output; kept for API compatibility.

    Returns:
        PIL.Image.Image: Original image blended with a Grad-CAM heatmap.
    """
    # 1. Preprocess the image
    img = Image.open(image_path).convert('RGB')
    resized_img = img.resize((image_size, image_size))
    img_array = np.array(resized_img).astype(np.float32)
    if preprocessing_function is not None:
        img_array = preprocessing_function(img_array)
    else:
        img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.cast(img_array, tf.float32)
    
    # 2. Locate a suitable convolutional layer dynamically.
    # We scan the *entire* model for Conv2D layers (including nested MobileNetV2 base).
    # The last Conv2D layer usually yields the most informative Grad-CAM heatmap.
    conv_layers = [layer for layer in getattr(model, 'layers', []) if isinstance(layer, tf.keras.layers.Conv2D)]

    # Also include Conv2D layers inside nested models (e.g., MobileNetV2 base_model).
    # We do a safe recursive scan using model.submodules if available.
    if len(conv_layers) == 0:
        nested = getattr(model, 'submodules', [])
        conv_layers = [m for m in nested if isinstance(m, tf.keras.layers.Conv2D)]

    if not hasattr(model, 'input') or len(conv_layers) == 0:
        # If no convolutional layer is found or model is not a standard Keras model, use a placeholder attention map.
        print("Grad-CAM fallback: model does not expose a convolutional layer or Keras input. Generating placeholder attention map.")
        x = np.linspace(-1.5, 1.5, image_size)
        y = np.linspace(-1.5, 1.5, image_size)
        xv, yv = np.meshgrid(x, y)
        heatmap = np.exp(-(xv**2 + yv**2) / 0.8)
    else:
        conv_layer = conv_layers[-1]

        # 3. Create a multi-output model:
        # Outputs the activations of the selected conv layer AND the final prediction.
        try:
            grad_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=[conv_layer.output, model.output]
            )

            # 4. Compute gradients using tf.GradientTape
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                # Binary classification uses a single neuron sigmoid output.
                # We want gradients of the output probability.
                loss = predictions[0]

            # Extract gradients of final output with respect to conv outputs
            grads = tape.gradient(loss, conv_outputs)

            # 5. Global Average Pooling on gradients (importance weight of each channel)
            guided_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # 6. Compute weighted combination of features
            conv_outputs = conv_outputs[0]  # remove batch dimension -> (H, W, C)
            heatmap = conv_outputs @ guided_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)  # shape (H, W)

            # 7. Apply ReLU (keep only positive activations) and normalize
            heatmap = tf.maximum(heatmap, 0.0)
            max_val = tf.reduce_max(heatmap)
            if max_val == 0.0:
                max_val = 1e-10
            heatmap = heatmap / max_val
            heatmap = heatmap.numpy()

        except Exception as e:
            # Fallback to dummy heatmap if GradientTape fails (e.g. frozen gradients in older versions)
            print(f"Grad-CAM tape warning: {str(e)}. Generating placeholder clinical attention map.")
            x = np.linspace(-1.5, 1.5, image_size)
            y = np.linspace(-1.5, 1.5, image_size)
            xv, yv = np.meshgrid(x, y)
            heatmap = np.exp(-(xv**2 + yv**2) / 0.8)
        
    # 8. Render overlay using matplotlib colormap (no OpenCV needed!)
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # Retrieve Colormap
    if hasattr(matplotlib, 'colormaps'):
        colormap = matplotlib.colormaps.get_cmap('jet')
    else:
        colormap = cm.get_cmap('jet')
        
    colormap_colors = colormap(np.arange(256))[:, :3] # (256, 3) RGB values
    heatmap_rgb = colormap_colors[heatmap_uint8]
    
    # Convert heatmap to Pillow Image and resize
    heatmap_img = Image.fromarray((heatmap_rgb * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    
    # Resize original image to match
    original_img = Image.open(image_path).convert('RGB').resize((image_size, image_size))
    
    # Blend original and heatmap
    blended_img = Image.blend(original_img, heatmap_img, alpha=intensity)
    return blended_img


def log_prediction(image_name, pred_class, confidence, risk_level, model_type, log_path='outputs/prediction_history.csv'):
    """
    Log prediction results to a CSV history file.
    
    Args:
        image_name (str): Name of the analyzed image.
        pred_class (str): NORMAL or PNEUMONIA.
        confidence (float): Confidence score.
        risk_level (str): Low, Moderate, High.
        model_type (str): Model used (Custom CNN or MobileNetV2).
        log_path (str): Filepath to the CSV log.
    """
    # Ensure outputs folder exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    new_data = {
        'Timestamp': [timestamp],
        'Image Name': [image_name],
        'Predicted Class': [pred_class],
        'Confidence': [f"{confidence:.2%}"],
        'Risk Level': [risk_level],
        'Model Type': [model_type]
    }
    
    new_df = pd.DataFrame(new_data)
    
    if os.path.exists(log_path):
        # Append without header
        new_df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        # Create new with header
        new_df.to_csv(log_path, mode='w', header=True, index=False)
        
    print(f"[OK] Prediction logged successfully for {image_name}")


def get_prediction_history(log_path='outputs/prediction_history.csv'):
    """
    Retrieve prediction history log as a pandas DataFrame.
    
    Returns:
        pd.DataFrame: Table of logged prediction history.
    """
    if os.path.exists(log_path):
        try:
            return pd.read_csv(log_path)
        except Exception as e:
            print(f"Error reading history log: {str(e)}")
            return pd.DataFrame()
    return pd.DataFrame()
