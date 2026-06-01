"""
Data loading and preprocessing module for chest X-ray pneumonia detection.
Handles dataset loading, preprocessing, and augmentation.
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image


class DataLoader:
    """
    Loads and preprocesses chest X-ray images for pneumonia detection.
    Supports both training and testing datasets.
    """
    
    def __init__(self, dataset_path, image_size=224, batch_size=16, preprocessing_function=None):
        """
        Initialize DataLoader.

        Args:
            dataset_path (str): Path to the dataset directory
            image_size (int): Target image size (224x224)
            batch_size (int): Batch size for data loading
            preprocessing_function (callable): Optional image preprocessing function for transfer learning models.
        """
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.preprocessing_function = preprocessing_function
        self.train_path = os.path.join(dataset_path, "train")
        self.test_path = os.path.join(dataset_path, "test")
        self.val_path = os.path.join(dataset_path, "val")
        
    def _folder_image_counts(self, path):
        counts = {}
        if not os.path.exists(path):
            return counts
        for class_name in os.listdir(path):
            class_path = os.path.join(path, class_name)
            if os.path.isdir(class_path):
                counts[class_name] = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
        return counts

    def _has_sufficient_validation(self, min_per_class=20):
        counts = self._folder_image_counts(self.val_path)
        return len(counts) >= 2 and all(count >= min_per_class for count in counts.values())

    def check_dataset(self):
        """Check if dataset paths exist and display structure."""
        print("Checking dataset structure...")
        print(f"Dataset path: {self.dataset_path}")
        print(f"Dataset exists: {os.path.exists(self.dataset_path)}")

        if os.path.exists(self.train_path):
            print(f"\nTraining data:")
            for class_name, count in self._folder_image_counts(self.train_path).items():
                print(f"  {class_name}: {count} images")

        if os.path.exists(self.test_path):
            print(f"\nTest data:")
            for class_name, count in self._folder_image_counts(self.test_path).items():
                print(f"  {class_name}: {count} images")

        if os.path.exists(self.val_path):
            print(f"\nValidation data:")
            for class_name, count in self._folder_image_counts(self.val_path).items():
                print(f"  {class_name}: {count} images")

        if os.path.exists(self.val_path) and not self._has_sufficient_validation():
            print("\n[Warning] Validation directory contains few images. Training will use a validation split from the training set for better generalization.")
    
    def get_train_data(self, validation_split=0.15, seed=42):
        """
        Load and augment training data.

        If the validation directory is too small, split the training set for validation.
        Returns:
            train_generator: ImageDataGenerator for training data with augmentation
        """
        train_datagen = ImageDataGenerator(
            rescale=None if self.preprocessing_function else 1.0/255.0,
            preprocessing_function=self.preprocessing_function,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            zoom_range=0.15,
            shear_range=0.12,
            fill_mode='nearest',
            validation_split=validation_split
        )

        if os.path.exists(self.val_path) and self._has_sufficient_validation():
            train_generator = train_datagen.flow_from_directory(
                self.train_path,
                target_size=(self.image_size, self.image_size),
                batch_size=self.batch_size,
                class_mode='binary',
                shuffle=True,
                seed=seed
            )
        else:
            train_generator = train_datagen.flow_from_directory(
                self.train_path,
                target_size=(self.image_size, self.image_size),
                batch_size=self.batch_size,
                class_mode='binary',
                shuffle=True,
                subset='training',
                seed=seed
            )

        return train_generator

    def get_val_data(self, validation_split=0.15, seed=42):
        """
        Load validation data without augmentation.

        If the validation directory is too small, use a validation subset from the training directory.
        Returns:
            val_generator: ImageDataGenerator for validation data
        """
        val_rescale = None if self.preprocessing_function else 1.0/255.0
        if os.path.exists(self.val_path) and self._has_sufficient_validation():
            val_datagen = ImageDataGenerator(
                rescale=val_rescale,
                preprocessing_function=self.preprocessing_function
            )
            val_generator = val_datagen.flow_from_directory(
                self.val_path,
                target_size=(self.image_size, self.image_size),
                batch_size=self.batch_size,
                class_mode='binary',
                shuffle=False
            )
        else:
            val_datagen = ImageDataGenerator(
                rescale=val_rescale,
                preprocessing_function=self.preprocessing_function,
                validation_split=validation_split
            )
            val_generator = val_datagen.flow_from_directory(
                self.train_path,
                target_size=(self.image_size, self.image_size),
                batch_size=self.batch_size,
                class_mode='binary',
                shuffle=False,
                subset='validation',
                seed=seed
            )

        return val_generator
    
    def get_test_data(self):
        """
        Load test data without augmentation.
        
        Returns:
            test_generator: ImageDataGenerator for test data
        """
        test_datagen = ImageDataGenerator(
            rescale=None if self.preprocessing_function else 1.0/255.0,
            preprocessing_function=self.preprocessing_function
        )
        
        test_generator = test_datagen.flow_from_directory(
            self.test_path,
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        return test_generator
    
    def get_class_indices(self):
        """
        Get class to index mapping.
        
        Returns:
            dict: Mapping of class names to indices
        """
        # Get class indices from training directory
        classes = sorted(os.listdir(self.train_path))
        class_indices = {cls: idx for idx, cls in enumerate(classes)}
        return class_indices, classes
