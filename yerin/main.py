from yerin.cnn_training import cnn_training
from yerin.data_generators import data_generators
from yerin.data_injection import data_injection
from yerin.mobilenet_v2_training import mobile_net_training
from yerin.prepare_dataset import prepare_dataset
from yerin.model_testing import model_testing

if __name__ == "__main__":
    # Set-up
    # data_injection()
    # prepare_dataset()

    # CNN
    train_gen, val_gen, test_gen, class_weight_dict =data_generators()
    cnn_model = cnn_training(train_gen, val_gen, class_weight_dict)
    model_testing(cnn_model, test_gen, model_name="cnn")

    # MobileNetV2
    train_gen, val_gen, test_gen, class_weight_dict =data_generators(image_size=(96,96), color_mode='rgb')
    mobilenet_model = mobile_net_training(train_gen, val_gen, class_weight_dict)
    model_testing(mobilenet_model, test_gen, model_name="mobilenet_v2")