import os
from datetime import datetime

from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

def efficientnet_b0_training(
        train_gen,
        val_gen,
        class_weight_dict,
        input_shape=(224,224,3),
        epochs=50,
        fine_tuning_epochs=50,
        model_name="efficientnet_b0"
        ):
    os.makedirs("../Models", exist_ok=True)
    model_path = f"../Models/training_{model_name}_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".keras"
    base_model = EfficientNetB0(
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
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    # Callbacks
    callbacks_training = [
        EarlyStopping(monitor='val_loss',
                      patience=8,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss',
                          factor=0.5, patience=5,
                          verbose=1),
        ModelCheckpoint(model_path,
                        monitor='val_accuracy',
                        save_best_only=True,
                        mode='max')
    ]

    print("Training started...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks_training
    )

    print("Fine-tuning efficientnet b0")
    fine_tune_at = int(len(base_model.layers) * 0.7)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_finetuning = [
        EarlyStopping(monitor='val_loss',
                      patience=8,
                      restore_best_weights=True),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=fine_tuning_epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks_finetuning
    )
    model.save(model_path)
    return model