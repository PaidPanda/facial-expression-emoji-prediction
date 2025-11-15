import os

import keras.models
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam


def mobile_net_training(
        train_gen,
        val_gen,
        class_weight_dict,
        input_shape=(96,96,3),
        lr=1e-5,
        epochs=30,
        fine_tuning_epochs=10,
        model_name="mobilenet_v2"
        ):
    model_path = f"../Models/training_{model_name}.keras"
    os.makedirs("../Models", exist_ok=True)
    if not os.path.exists(model_path):
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(7, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=outputs)

        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary()

        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
            ModelCheckpoint(f'../Models/training_{model_name}.keras', save_best_only=True)
        ]

        print("Training started...")
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            class_weight=class_weight_dict,
            callbacks=callbacks
        )

        base_model.trainable = True
        for layer in base_model.layers[:100]:
            layer.trainable = False
    else:
        model=keras.models.load_model(model_path)

        model.trainable = True
        for layer in model.layers[:100]:
            layer.trainable = False

        print("Fine-tuning mobile net v2")
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
            ModelCheckpoint(f'../Models/training_{model_name}.keras', save_best_only=True)
        ]

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=fine_tuning_epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    return model