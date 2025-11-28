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

def run_model(option):
    time_stamp = create_folders()

    match option:
        case 2:
            # CNN
            train_gen, val_gen, test_gen, class_weight_dict =data_generators()
            cnn_model, history = cnn_training(time_stamp=time_stamp, train_gen=train_gen, val_gen=val_gen,
                                     class_weight_dict=class_weight_dict)
            y_pred_proba, y_pred, y_true = model_testing(time_stamp=time_stamp, model=cnn_model, test_gen=test_gen, model_name="cnn")
        case 3:
            # MobileNet V2
            train_gen, val_gen, test_gen, class_weight_dict =data_generators(model_name="mobilenet_v2")
            mobilenet_v2_model, history = mobilenet_v2_training(time_stamp=time_stamp, train_gen=train_gen,
                                                                val_gen=val_gen, class_weight_dict=class_weight_dict)
            y_pred_proba, y_pred, y_true = model_testing(time_stamp=time_stamp, model=mobilenet_v2_model, test_gen=test_gen, model_name="mobilenet_v2")
        case 4:
            # MobileNet V3 small
            train_gen, val_gen, test_gen, class_weight_dict = data_generators(model_name='mobilenet_v3_small')
            mobilenet_v3_model, history = mobilenet_v3_small_training(time_stamp=time_stamp, train_gen=train_gen,
                                                                      val_gen=val_gen, class_weight_dict=class_weight_dict)
            y_pred_proba, y_pred, y_true = model_testing(time_stamp=time_stamp, model=mobilenet_v3_model, test_gen=test_gen,
                                                         model_name="mobilenet_v3_small")
        case 5:
            # Efficientnet B0
            train_gen, val_gen, test_gen, class_weight_dict = data_generators(model_name="efficientnet_b0")
            efficientnet_model, history = efficientnet_b0_training(time_stamp=time_stamp, train_gen=train_gen, val_gen=val_gen,
                                                                   class_weight_dict=class_weight_dict)
            y_pred_proba, y_pred, y_true  = model_testing(time_stamp=time_stamp, model=efficientnet_model, test_gen=test_gen, model_name="efficientnet_b0")

    generate_graphs(time_stamp=time_stamp, history=history, y_pred_proba=y_pred_proba, y_pred=y_pred, y_true=y_true)

def run_menu():
    while True:
        print(f'--- Main Menu ---')
        print('1: Setup\n2: CNN\n3: Mobile Net V2\n4: Mobile Net V3\n5: Efficient Net B0\n6: Exit')
        print('-' * 8)
        try:
            user_input = int(input('Select option: '))
        except ValueError:
            user_input = -1
        match user_input:
            case 1:
                data_injection()
                prepare_dataset()
            case 2:
                run_model(2)
            case 3:
                run_model(3)
            case 4:
                run_model(4)
            case 5:
                run_model(5)
            case 6:
                break
            case _:
                print("Invalid input.")

if __name__ == "__main__":
    run_menu()