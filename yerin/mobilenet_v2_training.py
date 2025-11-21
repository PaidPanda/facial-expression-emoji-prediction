from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import LearningRateScheduler

# low-power devices, small GPUs, real-time inference, 30-80 FPS
def mobilenet_v2_training(
        time_stamp,
        train_gen,
        val_gen,
        class_weight_dict,
        initial_epochs=0,
        epochs=30,
        input_shape=(128, 128, 3),
        training_learning_rate=5e-4,
        fine_tuning_learning_rate=5e-5,
        csv_name='training_log',
        fine_tuning_epochs=120,
        is_load_model=True,
        model_name="mobilenet_v2"
        ):
    model_path = f"../Models/{time_stamp}/training_{model_name}.keras"
    csv_path = f"../Graphs/{time_stamp}/{csv_name}.csv"

    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(7, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    if is_load_model:
        print("Loading best model from checkpoint...")
        model = load_model(model_path)

    model.summary()
    training_optimizer = Adam(learning_rate=training_learning_rate)

    model.compile(
        optimizer=training_optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    def lr_schedule(epoch):
        initial = training_learning_rate
        if epoch < 50:
            return initial
        else:
            decay = 0.96 ** (epoch - 50)
            return initial * decay

    callbacks_training = [
        EarlyStopping(monitor='val_accuracy', patience=35, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        CSVLogger(csv_path, append=True),
        LearningRateScheduler(lr_schedule, verbose=1)
    ]


    print("Training started...")
    history_1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        initial_epoch=initial_epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks_training
    )

    print("Fine-tuning efficientnet b0")
    fine_tune_at = int(len(base_model.layers) * 0.7)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    def lr_schedule(epoch):
        # gentle decay starting after epoch 50
        initial = fine_tuning_learning_rate
        if epoch < 50:
            return float(initial)
        else:
            decay = 0.96 ** (epoch - 50)
            return float(initial * decay)

    training_optimizer = Adam(learning_rate=training_learning_rate)

    model.compile(
        optimizer=training_optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_finetuning = [
        EarlyStopping(monitor='val_accuracy', patience=35, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        CSVLogger(csv_path, append=True),
        LearningRateScheduler(lr_schedule, verbose=1)
    ]

    history_2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=fine_tuning_epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks_finetuning
    )

    model.save(model_path)

    full_history = {}
    for key in history_1.history.keys():
        full_history[key] = history_1.history[key] + history_2.history.get(key, [])

    class History:
        def __init__(self, history):
            self.history = history

    return model, History(full_history)