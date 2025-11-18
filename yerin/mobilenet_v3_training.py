import os
from datetime import datetime

from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.optimizers import Adam

# low-power devices, small GPUs, real-time inference, 30-80 FPS
def mobilenet_v3_small_training(
        train_gen,
        val_gen,
        class_weight_dict,
        input_shape=(224,224,3),
        epochs=10,
        fine_tuning_epochs=10,
        model_name="mobilenet_v3_small"
        ):
    os.makedirs("../Models", exist_ok=True)
    model_path = f"../Models/training_{model_name}_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".keras"
    base_model = MobileNetV3Small(
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

    print("Fine-tuning mobilenet v3 small")
    for layer in base_model.layers[:60]:
        layer.trainable = False
    for layer in base_model.layers[60:]:
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