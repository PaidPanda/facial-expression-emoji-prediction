from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger


def cnn_training(
        time_stamp,
        train_gen,
        val_gen,
        class_weight_dict,
        initial_epochs=0,
        epochs=150,
        model_name="cnn",
        is_load_model=True,
        learning_rate=5e-4,
        csv_name='training_log',
        input_shape=(128, 128, 3)
):
    model_path = f"../Models/{time_stamp}/training_{model_name}.keras"
    csv_path = f"../Graphs/{time_stamp}/{csv_name}.csv"
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='same', kernel_regularizer=l2(1e-5)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-5)),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-5)),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-5)),
        MaxPooling2D(2, 2),
        Dropout(0.35),

        Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-5)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.40),

        GlobalAveragePooling2D(),
        Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ], name="EmotionNet")

    if is_load_model:
        print("Loading best model from checkpoint...")
        model = load_model(model_path)

    model.summary()
    optimizer = Adam(learning_rate=learning_rate)

    # recompile — does NOT erase weights, only resets optimizer
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    def lr_schedule(epoch):
        # gentle decay starting after epoch 50
        initial = 5e-4
        if epoch < 50:
            return initial
        else:
            decay = 0.96 ** (epoch - 50)
            return initial * decay

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=35, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        CSVLogger(csv_path, append=True),
        LearningRateScheduler(lr_schedule, verbose=1)
    ]

    print("TRAINING...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        initial_epoch=initial_epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    return model, history