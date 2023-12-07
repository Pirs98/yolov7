import os
import shutil

def reduce_dataset(source_dir, ratio):

    dest_dir = source_dir + "_small"
    os.makedirs(dest_dir, exist_ok=True)

    for split in ["train2017", "val2017", "test"]:
        source_img_dir = os.path.join(source_dir, "images", split)
        dest_img_dir = os.path.join(dest_dir, "images", split)
        source_lab_dir = os.path.join(source_dir, "labels", split)
        dest_lab_dir = os.path.join(dest_dir, "labels", split)

        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_lab_dir, exist_ok=True)

        files = os.listdir(source_img_dir)
        num_files_to_keep = int(len(files)*ratio)

        for file_name in files[:num_files_to_keep]:
            source_img_path = os.path.join(source_img_dir, file_name)
            dest_img_path = os.path.join(dest_img_dir, file_name)
            source_lab_path = os.path.join(source_lab_dir, file_name.replace(".jpg", ".txt"))
            dest_lab_path = os.path.join(dest_lab_dir, file_name.replace(".jpg", ".txt"))

            try:
                shutil.copy(source_img_path, dest_img_path)
            except FileNotFoundError:
                print(f"Image {source_img_path} not found in source folder.")
            try:
                shutil.copy(source_lab_path, dest_lab_path)
            except FileNotFoundError:
                print(f"Image {source_lab_path} not found in source folder.")

    print("Dataset ridotto creato in: ", dest_dir)



ratio = 0.1
source_dir = "/storage/coco_dataset/coco/"

reduce_dataset(source_dir, ratio)