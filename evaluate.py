"""
Evaluation utilities for NeuroScan models.
Loads a trained model, runs inference on the test dataset, computes metrics
(accuracy, precision, recall, F1, ROC AUC), finds best thresholds, and
saves ROC/PR plots to `outputs/`.

Usage:
    python -m NeuroScan.evaluate --model_path saved_models/pneumonia_model.h5
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, precision_score, recall_score
)

try:
    import config
    from data import DataLoader
except ImportError:
    from . import config
    from .data import DataLoader

import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NeuroScan model on test set")
    parser.add_argument('--model_path', type=str, default=None, help='Path to .h5 model file')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size for test generator')
    parser.add_argument('--image_size', type=int, default=config.IMAGE_SIZE, help='Image size')
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['custom', 'mobilenet', 'efficientnet'],
        default=None,
        help='Optional model type for correct preprocessing.'
    )
    return parser.parse_args()


def evaluate_model(model_path, image_size, batch_size, model_type=None):
    if model_path is None:
        # choose available model
        cand = [config.MODEL_SAVE_PATH, os.path.join('saved_models', config.MODEL_SAVE_PATH), 'mobilenet_pneumonia_model.h5', os.path.join('saved_models','mobilenet_pneumonia_model.h5')]
        for p in cand:
            if p and os.path.exists(p):
                model_path = p
                break
    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found. Tried: {model_path}")

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Prepare preprocessing function based on model type or file heuristic
    preprocessing_function = None
    if model_type == 'mobilenet' or (model_type is None and 'mobilenet' in os.path.basename(model_path).lower()):
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
        preprocessing_function = mobilenet_preprocess
    elif model_type == 'efficientnet' or (model_type is None and 'efficientnet' in os.path.basename(model_path).lower()):
        from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
        preprocessing_function = efficientnet_preprocess

    loader = DataLoader(
        dataset_path=config.DATASET_PATH,
        image_size=image_size,
        batch_size=batch_size,
        preprocessing_function=preprocessing_function
    )
    test_gen = loader.get_test_data()

    # Predict on entire test set
    y_true = []
    y_score = []

    test_gen.reset()
    for i in range(len(test_gen)):
        x, y = test_gen[i]
        preds = model.predict(x, verbose=0).flatten()
        y_score.extend(preds.tolist())
        y_true.extend(y.tolist())

    y_true = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)

    # Default threshold 0.5
    y_pred_05 = (y_score > 0.5).astype(int)

    acc = np.mean(y_pred_05 == y_true)
    prec = precision_score(y_true, y_pred_05, zero_division=0)
    rec = recall_score(y_true, y_pred_05, zero_division=0)
    f1 = f1_score(y_true, y_pred_05, zero_division=0)

    print('\nEvaluation at threshold 0.5:')
    print(f' Accuracy: {acc:.4f}')
    print(f' Precision: {prec:.4f}')
    print(f' Recall: {rec:.4f}')
    print(f' F1: {f1:.4f}')
    print('\nClassification report (0.5):')
    print(classification_report(y_true, y_pred_05, target_names=config.CLASS_NAMES, zero_division=0))

    # ROC AUC
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float('nan')
    print(f'ROC AUC: {auc:.4f}')

    # ROC Curve
    fpr, tpr, thr_roc = roc_curve(y_true, y_score)

    # Precision-Recall
    precision, recall, pr_thr = precision_recall_curve(y_true, y_score)

    # Find threshold maximizing F1 from PR curve
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
    best_idx = np.nanargmax(f1_scores)
    best_thr = pr_thr[best_idx] if best_idx < len(pr_thr) else 0.5
    best_f1 = f1_scores[best_idx]

    # Find threshold maximizing accuracy explicitly
    threshold_candidates = np.linspace(0.0, 1.0, 1001)
    best_accuracy = 0.0
    best_accuracy_threshold = 0.5
    for thr in threshold_candidates:
        y_pred_candidate = (y_score > thr).astype(int)
        acc_candidate = np.mean(y_pred_candidate == y_true)
        if acc_candidate > best_accuracy:
            best_accuracy = acc_candidate
            best_accuracy_threshold = thr

    y_pred_best = (y_score > best_thr).astype(int)
    accuracy_best = np.mean(y_pred_best == y_true)
    precision_best = precision_score(y_true, y_pred_best, zero_division=0)
    recall_best = recall_score(y_true, y_pred_best, zero_division=0)

    print(f'Best threshold by PR-F1: {best_thr:.4f} (F1={best_f1:.4f})')
    print(f'Accuracy @PR-F1 threshold: {accuracy_best:.4f}')
    print(f'Precision @PR-F1 threshold: {precision_best:.4f}')
    print(f'Recall @PR-F1 threshold: {recall_best:.4f}')
    print(f'Best threshold by accuracy: {best_accuracy_threshold:.4f} (Accuracy={best_accuracy:.4f})')

    # Print the same threshold information again for clarity
    print(f'Best threshold candidates: PR-F1={best_thr:.4f}, Accuracy={best_accuracy_threshold:.4f}')

    # Find threshold that maximizes specificity while keeping recall >= 0.7 (user can tune)
    specs = 1 - fpr
    # find thresholds with recall >= 0.7
    valid = np.where(tpr >= 0.70)[0]
    if len(valid) > 0:
        spec_vals = specs[valid]
        chosen_idx = valid[np.argmax(spec_vals)]
        thr_spec = thr_roc[chosen_idx]
        spec_val = specs[chosen_idx]
        rec_val = tpr[chosen_idx]
        print(f"Threshold with recall>=0.70 maximizing specificity: thr={thr_spec:.4f}, specificity={spec_val:.4f}, recall={rec_val:.4f}")
    else:
        thr_spec = 0.5

    # Save plots
    os.makedirs('outputs', exist_ok=True)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
    plt.plot([0,1],[0,1],'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    roc_path = os.path.join('outputs','roc_curve.png')
    plt.savefig(roc_path, dpi=150)
    plt.close()

    plt.figure(figsize=(6,6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    pr_path = os.path.join('outputs','pr_curve.png')
    plt.savefig(pr_path, dpi=150)
    plt.close()

    # Print confusion matrices at 0.5 and best_thr
    cm_05 = confusion_matrix(y_true, y_pred_05)
    y_pred_best = (y_score > best_thr).astype(int)
    cm_best = confusion_matrix(y_true, y_pred_best)

    print('\nConfusion matrix @0.5')
    print(cm_05)
    print('\nConfusion matrix @best_thr')
    print(cm_best)

    # Save a small summary
    summary_path = os.path.join('outputs','evaluation_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f'Model: {model_path}\n')
        f.write(f'Num test samples: {len(y_true)}\n')
        f.write(f'AUC: {auc:.4f}\n')
        f.write(f'Accuracy @0.5: {acc:.4f}\n')
        f.write(f'Precision @0.5: {prec:.4f}\n')
        f.write(f'Recall @0.5: {rec:.4f}\n')
        f.write(f'F1 @0.5: {f1:.4f}\n')
        f.write(f'Best thr (PR F1): {best_thr:.4f}\n')
        f.write(f'Best F1: {best_f1:.4f}\n')
        f.write(f'Accuracy @PR-F1 threshold: {accuracy_best:.4f}\n')
        f.write(f'Precision @PR-F1 threshold: {precision_best:.4f}\n')
        f.write(f'Recall @PR-F1 threshold: {recall_best:.4f}\n')
        f.write(f'Best thr by accuracy: {best_accuracy_threshold:.4f}\n')
        f.write(f'Best accuracy: {best_accuracy:.4f}\n')
        f.write(f'ROC curve: {roc_path}\n')
        f.write(f'PR curve: {pr_path}\n')

    print(f'Plots and summary saved in outputs/')


if __name__ == '__main__':
    args = parse_args()
    evaluate_model(args.model_path, args.image_size, args.batch_size, model_type=args.model_type)
