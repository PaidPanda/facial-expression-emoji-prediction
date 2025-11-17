import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import compute_class_weight
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input



def data_generators(
        model_name='cnn',
        train_dir='../DataSets/train',
        val_dir='../DataSets/val',
        test_dir='../DataSets/test',
        emotions=None,
):
    if emotions is None:
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    BATCH_SIZE = 64

    # Data transformation
    if model_name=='cnn':
        image_size = (48, 48),
        color_mode = 'grayscale'
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        val_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
    else:
        image_size = (224, 224)
        color_mode = 'rgb'
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

    # Data labeling
    train_gen = train_datagen.flow_from_directory(
        f'{train_dir}/',
        target_size=image_size,
        color_mode=color_mode,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        f'{val_dir}/',
        target_size=image_size,
        color_mode=color_mode,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_gen = test_datagen.flow_from_directory(
        f'{test_dir}/',
        target_size=image_size,
        color_mode=color_mode,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )

    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print("Class weights applied:")
    for i, emo in enumerate(emotions):
        print(f"  {emo:8} → {class_weight_dict[i]:.2f}x")
    return train_gen, val_gen, test_gen, class_weight_dict