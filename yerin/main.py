from yerin.cnn_training import cnn_training
from yerin.data_generators import data_generators
from yerin.data_injection import data_injection
from yerin.efficientnet_b0 import efficientnet_b0_training
from yerin.folders_creator import create_folders
from yerin.mobilenet_v2_training import mobilenet_v2_training
from yerin.mobilenet_v3_training import mobilenet_v3_small_training
from yerin.prepare_dataset import prepare_dataset
from yerin.model_testing import model_testing

if __name__ == "__main__":
    # Set-up
    # data_injection()
    # prepare_dataset()

    # Create folders
    time_stamp = create_folders()

    # CNN
    # train_gen, val_gen, test_gen, class_weight_dict =data_generators()
    # cnn_model = cnn_training(time_stamp=time_stamp, train_gen=train_gen, val_gen=val_gen,
    #                          class_weight_dict=class_weight_dict)
    # model_testing(time_stamp=time_stamp, model=cnn_model, test_gen=test_gen, model_name="cnn")

    # MobileNet V2
    train_gen, val_gen, test_gen, class_weight_dict =data_generators(model_name="mobilenet_v2")
    mobilenet_model = mobilenet_v2_training(time_stamp=time_stamp, train_gen=train_gen, val_gen=val_gen,
                                            class_weight_dict=class_weight_dict, epochs=100, fine_tuning_epochs=100)
    model_testing(time_stamp=time_stamp, model=mobilenet_model, test_gen=test_gen, model_name="mobilenet_v2")

    # MobileNet V3 small
    # train_gen, val_gen, test_gen, class_weight_dict =data_generators(model_name='mobilenet_v3_small')
    # mobilenet_model = mobilenet_v3_small_training(time_stamp=time_stamp, train_gen=train_gen, val_gen=val_gen,
    #                                               class_weight_dict=class_weight_dict, epochs=100, fine_tuning_epochs=100)
    # model_testing(time_stamp=time_stamp, model=mobilenet_model, test_gen=test_gen, model_name="mobilenet_v3_small")

    # Efficientnet B0
    # train_gen, val_gen, test_gen, class_weight_dict = data_generators(model_name="efficientnet_b0")
    # efficientnet_model = efficientnet_b0_training(time_stamp=time_stamp, train_gen=train_gen, val_gen=val_gen,
    #                                               class_weight_dict=class_weight_dict, epochs=100, fine_tuning_epochs=100)
    # model_testing(time_stamp=time_stamp, model=efficientnet_model, test_gen=test_gen, model_name="efficientnet_b0")