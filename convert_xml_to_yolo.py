import os
import shutil
import xml.etree.ElementTree as ET

def xml_to_yolo(xml_file):
    label_map = {
        "bike": 0,
        "car": 1,
        "lorry": 2,
        "bus": 3,
        "pedestrian": 4
    }

    tree = ET.parse(xml_file)
    root = tree.getroot()

    yolo_annotations = []

    for obj in root.findall('object'):
        label=obj.find('name').text
        if label in label_map:
            label_id = label_map[label]
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            xmax = float(bbox.find('xmax').text)
            ymin = float(bbox.find('ymin').text)
            ymax = float(bbox.find('ymax').text)

            width = xmax - xmin
            height = ymax - ymin
            x_center = xmin + width / 2
            y_center = ymin + height / 2

            width /= float(root.find('size').find('width').text)
            height /= float(root.find('size').find('height').text)
            x_center /= float(root.find('size').find('width').text)
            y_center /= float(root.find('size').find('height').text)

            yolo_annotations.append([label_id, x_center, y_center, width, height])

    return  yolo_annotations

input_dir = "/storage/crossroad_datasets"
output_dir = "/storage/DATASET_NO_UAV/"

out_image_dir = os.path.join(output_dir, "images")
os.makedirs(out_image_dir, exist_ok=True)
out_label_dir = os.path.join(output_dir, "labels")
os.makedirs(out_label_dir, exist_ok=True)

counter = 0

for dataset_dir in os.listdir(input_dir):
    dataset_path = os.path.join(input_dir, dataset_dir)
    annotations_dir = os.path.join(dataset_path, "Annotations")
    images_dir = os.path.join(dataset_path, "JPEGImages")

    if os.path.exists(annotations_dir) and os.path.exists(images_dir):
        for xml_file in os.listdir(annotations_dir):
            xml_path = os.path.join(annotations_dir, xml_file)
            yolo_data = xml_to_yolo(xml_path)

            counter_formatted = '{:04d}'.format(counter)
            txt_path = os.path.join(out_label_dir, f"label_{counter_formatted}.txt")
            with open(txt_path, "w") as f:
                for gt in yolo_data:
                    f.write(' '.join(str(x) for x in gt) + "\n")

            if dataset_dir.startswith("other"):
                image_file = os.path.splitext(xml_file)[0] + ".png"
                image_path = os.path.join(images_dir, image_file)
                out_image_path = os.path.join(out_image_dir, f"label_{counter_formatted}.png")
                shutil.copy(image_path, out_image_path)
            else:
                image_file = os.path.splitext(xml_file)[0] + ".jpg"
                image_path = os.path.join(images_dir, image_file)
                out_image_path = os.path.join(out_image_dir, f"label_{counter_formatted}.jpg")
                shutil.copy(image_path, out_image_path)

            counter += 1

print("Operazione completata")
















