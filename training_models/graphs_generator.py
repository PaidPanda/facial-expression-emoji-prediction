from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
import numpy as np

def generate_graphs(time_stamp, history, y_pred_proba, y_pred, y_true, emotions=None):
    if emotions is None:
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # ROC Curve
    n_classes = 7
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"../Graphs/{time_stamp}/roc_curve.png")
    plt.close()

    # Precision-Recall Curve
    precision = dict()
    recall = dict()
    avg_prec = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
        avg_prec[i] = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], lw=2, label=f'Class {i} (AP={avg_prec[i]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(f"../Graphs/{time_stamp}/precision_recall_curve.png")
    plt.close()

    # Calibration plot
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true_bin.ravel(), y_pred_proba.ravel(), n_bins=10)

    plt.figure()
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.savefig(f"../Graphs/{time_stamp}/calibration_plot.png")
    plt.close()

    # Report
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()

    # Per-class metrics
    plt.figure(figsize=(8, 6))
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(emotions))
    width = 0.25
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, df[metric][:7], width, label=metric)
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Per-Class Metrics')
    plt.xticks(x + width, emotions)
    plt.legend()
    plt.savefig(f"../Graphs/{time_stamp}/per_class_metrics.png")
    plt.close()

    # Saving the report
    report = classification_report(y_true, y_pred, target_names=emotions, output_dict=True)
    csv_path = f"../Graphs/{time_stamp}/report.csv"
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(csv_path, index=True)

    report = classification_report(y_true, y_pred, target_names=emotions, output_dict=False)
    print("\n" + "=" * 50)
    print(report)
    print("=" * 50)

    # Confusion Matrix
    print(history.history.keys())
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotions, yticklabels=emotions)
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"../Graphs/{time_stamp}/confusion_matrix.png")
    plt.close()

    # Learning curve
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../Graphs/{time_stamp}/learning_curve.png")
    plt.close()
