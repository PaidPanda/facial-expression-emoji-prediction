from yerin.cnn_training import cnn_training
from yerin.data_generators import data_generators
from yerin.data_injection import data_injection
from yerin.efficientnet_b0_training import efficientnet_b0_training
from yerin.folders_creator import create_folders
from yerin.graphs_generator import generate_graphs
from yerin.mobilenet_v2_training import mobilenet_v2_training
from yerin.mobilenet_v3_training import mobilenet_v3_small_training
from yerin.prepare_dataset import prepare_dataset
from yerin.model_testing import model_testing
from yerin.webcam import webcam_demo

if __name__ == "__main__":
    # Set-up
    # data_injection()
    # prepare_dataset()

    # Create folders
    time_stamp = create_folders()

    # CNN
    train_gen, val_gen, test_gen, class_weight_dict =data_generators(image_size=(48,48))
    cnn_model, history = cnn_training(time_stamp=time_stamp, train_gen=train_gen, val_gen=val_gen,
                             class_weight_dict=class_weight_dict, is_load_model=False, epochs=1, input_shape=(48,48,3))
    y_pred_proba, y_pred, y_true = model_testing(time_stamp=time_stamp, model=cnn_model, test_gen=test_gen, model_name="cnn")

    # MobileNet V2
    # train_gen, val_gen, test_gen, class_weight_dict =data_generators(model_name="mobilenet_v2")
    # mobilenet_v2_model, history = mobilenet_v2_training(time_stamp=time_stamp, train_gen=train_gen, val_gen=val_gen,
    #                                         class_weight_dict=class_weight_dict, epochs=100, fine_tuning_epochs=100)
    # y_pred_proba, y_pred, y_true = model_testing(time_stamp=time_stamp, model=mobilenet_v2_model, test_gen=test_gen, model_name="mobilenet_v2")

    # MobileNet V3 small
    # train_gen, val_gen, test_gen, class_weight_dict =data_generators(model_name='mobilenet_v3_small')
    # mobilenet_v3_model, history = mobilenet_v3_small_training(time_stamp=time_stamp, train_gen=train_gen, val_gen=val_gen,
    #                                               class_weight_dict=class_weight_dict, epochs=100, fine_tuning_epochs=100)
    # y_pred_proba, y_pred, y_true = model_testing(time_stamp=time_stamp, model=mobilenet_v3_model, test_gen=test_gen, model_name="mobilenet_v3_small")

    # Efficientnet B0
    # train_gen, val_gen, test_gen, class_weight_dict = data_generators(model_name="efficientnet_b0")
    # efficientnet_model, history = efficientnet_b0_training(time_stamp=time_stamp, train_gen=train_gen, val_gen=val_gen,
    #                                               class_weight_dict=class_weight_dict, epochs=100, fine_tuning_epochs=100)
    # y_pred_proba, y_pred, y_true  = model_testing(time_stamp=time_stamp, model=efficientnet_model, test_gen=test_gen, model_name="efficientnet_b0")


    generate_graphs(time_stamp=time_stamp, history=history, y_pred_proba=y_pred_proba, y_pred=y_pred, y_true=y_true)

    # webcam_demo('../Models/2025_11_17_21_53_14/training_mobilenet_v2.keras')
