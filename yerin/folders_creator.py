import os
from datetime import datetime


def create_folders():
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    os.makedirs("../Models", exist_ok=True)
    os.makedirs("../Graphs", exist_ok=True)
    os.makedirs(f"../Models/{time_stamp}", exist_ok=True)
    os.makedirs(f"../Graphs/{time_stamp}", exist_ok=True)
    return time_stamp