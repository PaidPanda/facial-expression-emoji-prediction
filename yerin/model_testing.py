import numpy as np

def model_testing(time_stamp, model, test_gen, model_name, emotions=None):
    # Evaluate
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f"\nFINAL TEST ACCURACY: {test_acc*100:.2f}%")

    # Predictions
    y_pred_proba = model.predict(test_gen)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_gen.classes

    # Save final model
    model.save(f'../Models/{time_stamp}/testing_{model_name}.keras')
    print(f"Model saved as 'testing_{model_name}.keras'")
    return y_pred_proba, y_pred, y_true