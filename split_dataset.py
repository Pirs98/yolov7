
import os
import shutil
import random

dataset_path = "/storage/DATASET_NO_UAV/"
output_dataset_path = "/storage/DATASET_NO_UAV_SPLITTED_medium/"

train_ratio = 0.3
val_ratio = 0.15
test_ratio = 0.55

annotations_path = os.path.join(dataset_path, "labels")
images_path = os.path.join(dataset_path, "images")

out_annotations_path = os.path.join(output_dataset_path, "labels")
out_images_path = os.path.join(output_dataset_path, "images")

ann_train_path = os.path.join(out_annotations_path, "train")
ann_val_path = os.path.join(out_annotations_path, "val")
ann_test_path = os.path.join(out_annotations_path, "test")

im_train_path = os.path.join(out_images_path, "train")
im_val_path = os.path.join(out_images_path, "val")
im_test_path = os.path.join(out_images_path, "test")

os.makedirs(ann_train_path, exist_ok=True)
os.makedirs(ann_val_path, exist_ok=True)
os.makedirs(ann_test_path, exist_ok=True)

os.makedirs(im_train_path, exist_ok=True)
os.makedirs(im_val_path, exist_ok=True)
os.makedirs(im_test_path, exist_ok=True)

annotations_files = os.listdir(annotations_path)
images_files = os.listdir(images_path)

num_train_samples = int(len(annotations_files) * train_ratio)
num_val_samples = int(len(annotations_files) * val_ratio)
num_test_samples = int(len(annotations_files) * test_ratio)

train_samples = random.sample(annotations_files, num_train_samples)
val_samples = random.sample(list(set(annotations_files) - set(train_samples)), num_val_samples)
test_samples = list(set(annotations_files) - set(train_samples) - set(val_samples))

for sample in train_samples:
    annotation_file = os.path.join(dataset_path, "labels", sample)
    image_file = os.path.join(dataset_path, "images", sample.replace(".txt", ".jpg"))
    if os.path.exists(image_file):
        shutil.copy(annotation_file, os.path.join(ann_train_path, sample))
        shutil.copy(image_file, os.path.join(im_train_path, sample.replace(".txt", ".jpg")))
    else:
        image_file = os.path.join(dataset_path, "images", sample.replace(".txt", ".png"))
        shutil.copy(annotation_file, os.path.join(ann_train_path, sample))
        shutil.copy(image_file, os.path.join(im_train_path, sample.replace(".txt", ".png")))

for sample in val_samples:
    annotation_file = os.path.join(dataset_path, "labels", sample)
    image_file = os.path.join(dataset_path, "images", sample.replace(".txt", ".jpg"))
    if os.path.exists(image_file):
        shutil.copy(annotation_file, os.path.join(ann_val_path, sample))
        shutil.copy(image_file, os.path.join(im_val_path, sample.replace(".txt", ".jpg")))
    else:
        image_file = os.path.join(dataset_path, "images", sample.replace(".txt", ".png"))
        shutil.copy(annotation_file, os.path.join(ann_val_path, sample))
        shutil.copy(image_file, os.path.join(im_val_path, sample.replace(".txt", ".png")))

for sample in test_samples:
    annotation_file = os.path.join(dataset_path, "labels", sample)
    image_file = os.path.join(dataset_path, "images", sample.replace(".txt", ".jpg"))
    if os.path.exists(image_file):
        shutil.copy(annotation_file, os.path.join(ann_test_path, sample))
        shutil.copy(image_file, os.path.join(im_test_path, sample.replace(".txt", ".jpg")))
    else:
        image_file = os.path.join(dataset_path, "images", sample.replace(".txt", ".png"))
        shutil.copy(annotation_file, os.path.join(ann_test_path, sample))
        shutil.copy(image_file, os.path.join(im_test_path, sample.replace(".txt", ".png")))
