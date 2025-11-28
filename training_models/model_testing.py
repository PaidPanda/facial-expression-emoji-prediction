import numpy as np
import pandas as pd


def model_testing(time_stamp, model, test_gen, model_name, emotions=None, csv_name='y_values'):
    if emotions is None:
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    csv_path = f"../Graphs/{time_stamp}/{csv_name}.csv"
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

    df = pd.DataFrame({
        "filename": test_gen.filenames,
        "y_true": y_true,
        "y_pred": y_pred
    })

    proba_df = pd.DataFrame(
        y_pred_proba,
        columns=[f"proba_{emo}" for emo in emotions]
    )

    df = pd.concat([df, proba_df], axis=1)
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    return y_pred_proba, y_pred, y_true