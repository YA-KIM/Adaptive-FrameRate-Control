import os
import shutil

base_path = "/home/hyhy/Datasets/FR_Dataset/LaSOT2"
train_txt = os.path.join(base_path, "training_set.txt")
val_txt = os.path.join(base_path, "testing_set.txt")

def move_folders(txt_path, target_folder):
    with open(txt_path, "r") as f:
        lines = f.read().splitlines()

    for line in lines:
        if '-' not in line:
            continue
        category, folder = line.split('-')[0], line.strip()
        src = os.path.join(base_path, category, folder)
        dst = os.path.join(base_path, target_folder, category, folder)

        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Moved: {src} -> {dst}")
        else:
            print(f"Not found: {src}")

# 실행
move_folders(train_txt, "training")
move_folders(val_txt, "val")
