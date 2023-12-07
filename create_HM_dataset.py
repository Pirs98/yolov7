import os
import random
import shutil
import sys

def copy_images(input_file, num_images, output_dir, use_random):

    with open(input_file, 'r') as f:
        lines = f.readlines()

    lines = [line.split() for line in lines]
    image_paths = [line[0] for line in lines]
    # values = [line.strip()[1] for line in lines]

    if use_random:
        selected_indices = random.sample(range(len(image_paths)), num_images)
    else:
        selected_indices = list(range(num_images))

    for index in selected_indices:
        image_path = image_paths[index]
        label_path = os.path.join(os.path.dirname(image_path).replace("images", "labels"), os.path.basename(os.path.splitext(image_path)[0]) + ".txt")
        # image_value = values[index]
        image_name = os.path.basename(image_path)
        dest_path = os.path.join(output_dir, "images/train/", image_name)
        label_dest_path = os.path.join(os.path.dirname(dest_path).replace("images", "labels"), os.path.basename(os.path.splitext(dest_path)[0]) + ".txt")
        shutil.copy(image_path, dest_path)
        shutil.copy(label_path, label_dest_path)
        print(f"Copied {image_path} in {dest_path}")


if __name__=="__main__":
    if len(sys.argv) < 4:
        print("Usage: python create_HM_dataset.py input_file.txt N dir [--random]")
        sys.exit(1)

    input_file = sys.argv[1]
    num_images = int(sys.argv[2])
    output_dir = sys.argv[3]
    use_random = "--random" in sys.argv

    image_train_dir = os.path.join(output_dir, "images", "train")
    os.makedirs(image_train_dir, exist_ok=True)
    label_train_dir = os.path.join(output_dir, "labels", "train")
    os.makedirs(label_train_dir, exist_ok=True)

    copy_images(input_file, num_images, output_dir, use_random)