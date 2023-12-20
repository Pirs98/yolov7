import os
import shutil

dataset_txt_split_dir = "/storage/mtsd_v2_fully_annotated/splits/"
output_dataset = "/storage/mapillary_dataset"

os.makedirs(output_dataset, exist_ok=True)
os.makedirs(os.path.join(output_dataset, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dataset, "val"), exist_ok=True)
os.makedirs(os.path.join(output_dataset, "test"), exist_ok=True)

def copy_images(image_list, source_folder, dest_folder):
    with open(image_list, 'r') as file:
        for line in file:
            image_name = line.strip() + ".jpg"
            source_path = os.path.join(source_folder, image_name)
            dest_path = os.path.join(output_dataset, dest_folder, image_name)
            try:
                shutil.copyfile(source_path, dest_path)
            except FileNotFoundError:
                print(f"Image {image_name} not found in source folder.")

copy_images(os.path.join(dataset_txt_split_dir, "val.txt"), "/storage/images", "val")
copy_images(os.path.join(dataset_txt_split_dir, "test.txt"), "/storage/images", "test")
copy_images(os.path.join(dataset_txt_split_dir, "train.txt"), "/storage/images", "train")