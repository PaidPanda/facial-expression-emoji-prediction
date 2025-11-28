import os
import shutil
import kagglehub


def data_injection(target_dir:str ="../DataFiles", remove_original_dir:bool = False):
    # Download facial emotion recognition dataset
    path = kagglehub.dataset_download("fahadullaha/facial-emotion-recognition-dataset")

    # Directories
    src_dir = os.path.join(path, "processed_data")
    os.makedirs(target_dir, exist_ok=True)

    # Copy into data into ../DataFiles
    for root, dirs, files in os.walk(src_dir):
        rel_path = os.path.relpath(root, src_dir)
        dest_dir = os.path.join(target_dir, rel_path)
        os.makedirs(dest_dir, exist_ok=True)
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_dir, file)
            shutil.copy2(src_file, dest_file)

    # Remove directory where the data is originally downloaded
    if remove_original_dir:
        shutil.rmtree(path)

    print("Path to dataset files:", os.path.abspath(target_dir))
