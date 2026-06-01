"""
Training pipeline for NeuroScan AI.
Loads chest X-ray data, builds the selected model architecture (Custom CNN or MobileNetV2),
trains the model using data augmentation, early stopping, and checkpointing,
and generates performance evaluations (Confusion Matrix, Metrics, Loss/Accuracy Curves).
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

try:
    import config
    from data import DataLoader
    from model import PneumoniaCNN, PneumoniaMobileNetV2, PneumoniaEfficientNetB0
except ImportError:
    from . import config
    from .data import DataLoader
    from .model import PneumoniaCNN, PneumoniaMobileNetV2, PneumoniaEfficientNetB0


def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="NeuroScan AI Training Pipeline")
    parser.add_argument(
        '--model', 
        type=str, 
        default='custom', 
        choices=['custom', 'mobilenet', 'efficientnet'],
        help="Model architecture to train: 'custom' for 4-block CNN, 'mobilenet' for MobileNetV2, 'efficientnet' for EfficientNetB0"
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        help="Optional explicit dataset directory path. Overrides config.DATASET_PATH if provided."
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=config.MAX_EPOCHS, 
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=config.BATCH_SIZE, 
        help="Batch size for training"
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=config.LEARNING_RATE, 
        help="Learning rate for Adam optimizer"
    )
    parser.add_argument(
        '--fine_tune',
        action='store_true',
        help="Enable MobileNetV2 fine-tuning by unfreezing the top layers"
    )
    parser.add_argument(
        '--fine_tune_layers',
        type=int,
        default=30,
        help="Number of MobileNetV2 layers to unfreeze when fine-tuning"
    )
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Create directories if they do not exist
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('assets', exist_ok=True)
    
    print("=" * 70)
    print(f" NEUROSCAN AI - TRAINING PIPELINE")
    print(f" Target Model: {args.model.upper()}")
    print("=" * 70)
    
    # 1. Load Dataset
    print(f"\n[1/7] Initializing DataLoader...")
    # Prefer explicit CLI dataset path, then environment variable, then config.
    if args.dataset_path and os.path.exists(args.dataset_path):
        dataset_path = args.dataset_path
    else:
        env_dataset = os.environ.get('DATASET_PATH')
        if env_dataset and os.path.exists(env_dataset):
            dataset_path = env_dataset
        elif os.path.exists('/workspace/chest_xray'):
            dataset_path = '/workspace/chest_xray'
        else:
            dataset_path = config.DATASET_PATH

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    print(f" Dataset Path: {dataset_path}")
    
    # 3. Build Model Architecture
    print(f"\n[3/7] Building Model Architecture...")
    preprocessing_function = None
    if args.model == 'custom':
        model_builder = PneumoniaCNN(input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
        model_name = "Custom 4-Block CNN"
        save_filename = 'pneumonia_model.h5'
        model = model_builder.build_model()
    elif args.model == 'mobilenet':
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
        preprocessing_function = mobilenet_preprocess
        model_builder = PneumoniaMobileNetV2(input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
        model_name = "MobileNetV2 Transfer Learning"
        save_filename = 'mobilenet_pneumonia_model.h5'
        model = model_builder.build_model(
            fine_tune=args.fine_tune,
            fine_tune_layers=args.fine_tune_layers
        )
        if args.fine_tune:
            print(f" [+] MobileNet fine-tuning enabled: last {args.fine_tune_layers} layers will be trainable.")
        else:
            print(" [+] MobileNet base is frozen; only the classifier head will train.")
    else:
        from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
        preprocessing_function = efficientnet_preprocess
        model_builder = PneumoniaEfficientNetB0(input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
        model_name = "EfficientNetB0 Transfer Learning"
        save_filename = 'efficientnet_pneumonia_model.h5'
        model = model_builder.build_model(
            fine_tune=args.fine_tune,
            fine_tune_layers=args.fine_tune_layers
        )
        if args.fine_tune:
            print(f" [+] EfficientNet fine-tuning enabled: last {args.fine_tune_layers} layers will be trainable.")
        else:
            print(" [+] EfficientNetB0 base is frozen; only the classifier head will train.")

    loader = DataLoader(
        dataset_path=dataset_path,
        image_size=config.IMAGE_SIZE,
        batch_size=args.batch_size,
        preprocessing_function=preprocessing_function
    )

    # Display dataset details
    loader.check_dataset()
    
    # 2. Prepare Data Generators
    print(f"\n[2/7] Preparing Data Generators...")
    print(" Loading Training Generator (with Data Augmentation)...")
    train_generator = loader.get_train_data()
    print(f" [+] Training Samples: {train_generator.samples}")
    
    print(" Loading Validation Generator...")
    val_generator = loader.get_val_data()
    print(f" [+] Validation Samples: {val_generator.samples}")
    
    print(" Loading Test Generator...")
    test_generator = loader.get_test_data()
    print(f" [+] Test Samples: {test_generator.samples}")

    model_builder.compile_model(learning_rate=args.lr)
    print(f" [+] {model_name} initialized and compiled with Adam (LR={args.lr})")
    
    model.summary()
    
    # 4. Set Callbacks (Early Stopping & Checkpoint)
    print(f"\n[4/7] Setting up Callbacks...")
    
    # Stop training if validation loss doesn't improve for N epochs
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # Checkpoint to save best weights only
    model_checkpoint_local = ModelCheckpoint(
        filepath=os.path.join('saved_models', save_filename),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Root level checkpoint to satisfy user's filename request
    model_checkpoint_root = ModelCheckpoint(
        filepath=save_filename,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    callbacks = [early_stopping, reduce_lr, model_checkpoint_local, model_checkpoint_root]
    
    # Build class weights to compensate for imbalance
    class_weights = None
    try:
        class_weights_values = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_generator.classes),
            y=train_generator.classes
        )
        class_weights = dict(enumerate(class_weights_values))
        print(f" [+] Computed class weights: {class_weights}")
    except Exception as e:
        print(f" [!] Warning: could not compute class weights: {e}")

    # 5. Train Model
    print(f"\n[5/7] Training deep learning model...")
    print(f" Max Epochs: {args.epochs}")
    print(f" Batch Size: {args.batch_size}")
    
    fit_kwargs = {
        'epochs': args.epochs,
        'validation_data': val_generator,
        'callbacks': callbacks,
        'verbose': 1
    }
    if class_weights is not None:
        fit_kwargs['class_weight'] = class_weights

    history = model.fit(
        train_generator,
        **fit_kwargs
    )
    
    print("\n[SUCCESS] Model training completed successfully.")
    
    # 6. Plot & Save Accuracy/Loss Curves
    print(f"\n[6/7] Generating training history plots...")
    plt.figure(figsize=config.PLOT_FIGSIZE_METRICS)
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='#00C2FF', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#FF5733', linewidth=2)
    plt.title('Model Classification Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='#00C2FF', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='#FF5733', linewidth=2)
    plt.title('Model Cross-Entropy Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    chart_path = os.path.join('outputs', config.ACCURACY_GRAPH_PATH)
    plt.savefig(chart_path, dpi=config.PLOT_DPI)
    plt.close()
    print(f" [+] Training metrics chart saved: {chart_path}")
    
    # 7. Evaluate on Test Set
    print(f"\n[7/7] Evaluating trained model on Test Set...")
    
    test_generator.reset()
    test_predictions = []
    test_labels = []
    
    for i in range(len(test_generator)):
        x, y = test_generator[i]
        preds = model.predict(x, verbose=0)
        test_predictions.extend(preds)
        test_labels.extend(y)
        
    test_predictions = np.array(test_predictions).flatten()
    test_labels = np.array(test_labels)
    
    # Convert probabilities to binary predictions
    test_predictions_binary = (test_predictions > 0.5).astype(int)
    
    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_predictions_binary, average='binary'
    )
    test_accuracy = np.mean(test_predictions_binary == test_labels)
    
    print("\n" + "=" * 50)
    print(f" TEST EVALUATION REPORT ({args.model.upper()})")
    print("=" * 50)
    print(f" Test Accuracy:  {test_accuracy:.2%}")
    print(f" Test Precision: {precision:.2%}")
    print(f" Test Recall:    {recall:.2%}")
    print(f" Test F1-Score:  {f1:.4f}")
    print("=" * 50)
    
    print("\nDetailed Classification Report:")
    print(classification_report(
        test_labels,
        test_predictions_binary,
        target_names=config.CLASS_NAMES
    ))
    
    # Generate Confusion Matrix
    print(" Generating confusion matrix...")
    cm = confusion_matrix(test_labels, test_predictions_binary)
    plt.figure(figsize=config.PLOT_FIGSIZE_CM)
    
    # Cyber colors for the confusion matrix heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        cbar_kws={'label': 'Count'},
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    
    plt.title(f'Confusion Matrix: {model_name}', fontsize=12, fontweight='bold', pad=15)
    plt.ylabel('Ground Truth Label', fontsize=10)
    plt.xlabel('AI Predicted Label', fontsize=10)
    plt.tight_layout()
    
    cm_path = os.path.join('outputs', config.CONFUSION_MATRIX_PATH)
    plt.savefig(cm_path, dpi=config.PLOT_DPI)
    plt.close()
    print(f" [+] Confusion matrix chart saved: {cm_path}")
    print("\n[SUCCESS] Pipeline execution complete! Model trained and saved.")
    print("=" * 70)


if __name__ == '__main__':
    main()
