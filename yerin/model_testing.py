import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def model_testing(time_stamp, model, test_gen, model_name, emotions=None):
    if emotions is None:
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # Evaluate
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f"\nFINAL TEST ACCURACY: {test_acc*100:.2f}%")

    # Predictions
    Y_pred = model.predict(test_gen)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = test_gen.classes

    # Report
    print("\n" + "="*50)
    print(classification_report(y_true, y_pred, target_names=emotions))
    print("="*50)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotions, yticklabels=emotions)
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"../Graphs/{time_stamp}/confusion_matrix.png")

    # Save final model
    model.save(f'../Models/{time_stamp}/testing_{model_name}.keras')
    print(f"Model saved as 'testing_{model_name}.keras'")