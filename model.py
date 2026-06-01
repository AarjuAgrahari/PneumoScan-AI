"""
Model architectures for chest X-ray pneumonia detection.
Includes:
1. PneumoniaCNN: A custom 4-block Convolutional Neural Network.
2. PneumoniaMobileNetV2: A transfer learning model based on MobileNetV2.
"""

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0


class PneumoniaCNN:
    """
    Custom 4-block Convolutional Neural Network for pneumonia detection.
    Includes convolutional layers, pooling, dropout, and dense layers.
    """
    
    def __init__(self, input_shape=(224, 224, 3)):
        """
        Initialize Custom CNN.
        
        Args:
            input_shape (tuple): Shape of input images (height, width, channels)
        """
        self.input_shape = input_shape
        self.model = None
    
    def build_model(self):
        """
        Build the CNN architecture.
        
        Returns:
            model: Compiled Keras model
        """
        model = models.Sequential()
        
        # Block 1: Conv -> MaxPool -> Dropout
        # 32 filters, extracts initial edges and low-level textures
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Block 2: Conv -> MaxPool -> Dropout
        # 64 filters, extracts intermediate shapes
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Block 3: Conv -> MaxPool -> Dropout
        # 128 filters, extracts complex spatial combinations
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Block 4: Conv -> MaxPool -> Dropout
        # 128 filters, extracts deep structural features
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Flatten and Dense Layers
        model.add(layers.Flatten())
        
        # Dense classification head
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        
        # Output layer (Sigmoid for binary classification: 0=NORMAL, 1=PNEUMONIA)
        model.add(layers.Dense(1, activation='sigmoid'))
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with Adam optimizer and binary crossentropy.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        optimizer = __import__('tensorflow.keras.optimizers', fromlist=['Adam']).Adam(
            learning_rate=learning_rate
        )
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return self.model
    
    def get_model(self):
        return self.model
    
    def print_summary(self):
        if self.model is None:
            raise ValueError("Model not built.")
        self.model.summary()


class PneumoniaEfficientNetB0:
    """
    Transfer learning model based on EfficientNetB0 for high-performance pneumonia detection.
    """

    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        self.base_model = None

    def build_model(self, fine_tune=False, fine_tune_layers=20):
        base_model = EfficientNetB0(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(1, activation='sigmoid')
        ])

        self.model = model
        self.base_model = base_model

        if fine_tune:
            self.unfreeze_top_layers(fine_tune_layers)

        return model

    def unfreeze_top_layers(self, num_layers=20):
        if not hasattr(self, 'base_model') or self.base_model is None:
            raise ValueError("Build the model before unfreezing layers.")

        total_layers = len(self.base_model.layers)
        num_layers = max(1, min(num_layers, total_layers))

        self.base_model.trainable = True
        for layer in self.base_model.layers[:-num_layers]:
            layer.trainable = False
        for layer in self.base_model.layers[-num_layers:]:
            layer.trainable = True

        print(f" [+] Enabled fine-tuning for the last {num_layers} EfficientNetB0 layers.")
        return self.base_model

    def compile_model(self, learning_rate=0.001):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        optimizer = __import__('tensorflow.keras.optimizers', fromlist=['Adam']).Adam(
            learning_rate=learning_rate
        )
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return self.model

    def get_model(self):
        return self.model

    def print_summary(self):
        if self.model is None:
            raise ValueError("Model not built.")
        self.model.summary()


class PneumoniaMobileNetV2:
    """
    Transfer learning model based on MobileNetV2 for fast and high-accuracy training.
    """
    
    def __init__(self, input_shape=(224, 224, 3)):
        """
        Initialize MobileNetV2 transfer learning model.
        """
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self, fine_tune=False, fine_tune_layers=30):
        """
        Build the MobileNetV2 architecture with a frozen base and custom classifier head.

        Args:
            fine_tune (bool): If True, unfreeze the top MobileNet layers for fine-tuning.
            fine_tune_layers (int): Number of MobileNet layers to unfreeze.
        """
        # Load pre-trained MobileNetV2 base (trained on ImageNet)
        # We exclude the top classification layer so we can add a binary classifier.
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the pre-trained weights by default
        base_model.trainable = False
        
        # Build the transfer learning model head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),    # Reduces 2D features to 1D vector
            layers.Dropout(0.5),                # Prevent overfitting on the new classification head
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(1, activation='sigmoid') # Binary classification (0=NORMAL, 1=PNEUMONIA)
        ])
        
        self.model = model
        self.base_model = base_model

        if fine_tune:
            self.unfreeze_top_layers(fine_tune_layers)

        return model

    def unfreeze_top_layers(self, num_layers=30):
        """
        Unfreeze the top layers of the MobileNetV2 base model for fine-tuning.

        Args:
            num_layers (int): Number of layers from the top of the base model to unfreeze.
        """
        if not hasattr(self, 'base_model') or self.base_model is None:
            raise ValueError("Build the model before unfreezing layers.")

        total_layers = len(self.base_model.layers)
        num_layers = max(1, min(num_layers, total_layers))

        self.base_model.trainable = True
        for layer in self.base_model.layers[:-num_layers]:
            layer.trainable = False
        for layer in self.base_model.layers[-num_layers:]:
            layer.trainable = True

        print(f" [+] Enabled fine-tuning for the last {num_layers} MobileNetV2 layers.")
        return self.base_model
        
    def compile_model(self, learning_rate=0.001):
        """
        Compile the transfer learning model.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        optimizer = __import__('tensorflow.keras.optimizers', fromlist=['Adam']).Adam(
            learning_rate=learning_rate
        )
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return self.model
        
    def get_model(self):
        return self.model
        
    def print_summary(self):
        if self.model is None:
            raise ValueError("Model not built.")
        self.model.summary()