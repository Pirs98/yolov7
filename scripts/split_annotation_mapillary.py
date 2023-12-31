import json
import os
import shutil

dataset_txt_split_dir = "/storage/mtsd_v2_fully_annotated/splits/"
output_dataset = "/storage/mapillary_dataset/labels/"

class_to_id_map = {"other-sign": 0, "regulatory--keep-right--g1": 1, "regulatory--priority-over-oncoming-vehicles--g1": 2, "regulatory--height-limit--g1": 3, "regulatory--maximum-speed-limit-35--g2": 4, "warning--railroad-crossing-with-barriers--g1": 5, "warning--curve-left--g2": 6, "warning--falling-rocks-or-debris-right--g1": 7, "regulatory--keep-right--g4": 8, "warning--pedestrians-crossing--g4": 9, "complementary--go-right--g2": 10, "complementary--keep-left--g1": 11, "regulatory--maximum-speed-limit-45--g3": 12, "complementary--chevron-right--g3": 13, "regulatory--one-way-right--g2": 14, "regulatory--yield--g1": 15, "regulatory--one-way-straight--g1": 16, "warning--curve-right--g1": 17, "regulatory--pedestrians-only--g2": 18, "information--emergency-facility--g2": 19, "regulatory--no-entry--g1": 20, "warning--railroad-crossing--g3": 21, "warning--pedestrians-crossing--g5": 22, "warning--crossroads--g3": 23, "complementary--chevron-left--g5": 24, "information--motorway--g1": 25, "regulatory--no-stopping--g15": 26, "information--pedestrians-crossing--g1": 27, "warning--railroad-crossing-without-barriers--g3": 28, "regulatory--go-straight-or-turn-right--g1": 29, "complementary--go-right--g1": 30, "complementary--distance--g1": 31, "warning--slippery-road-surface--g1": 32, "warning--curve-left--g1": 33, "information--parking--g1": 34, "complementary--go-left--g1": 35, "information--tram-bus-stop--g2": 36, "warning--crossroads--g1": 37, "regulatory--no-overtaking--g2": 38, "warning--railroad-crossing-with-barriers--g2": 39, "complementary--one-direction-left--g1": 40, "regulatory--stop--g1": 41, "complementary--trucks-turn-right--g1": 42, "regulatory--maximum-speed-limit-30--g1": 43, "regulatory--priority-road--g4": 44, "regulatory--pedestrians-only--g1": 45, "warning--pedestrians-crossing--g9": 46, "warning--junction-with-a-side-road-acute-right--g1": 47, "regulatory--end-of-maximum-speed-limit-30--g2": 48, "information--end-of-living-street--g1": 49, "regulatory--one-way-right--g3": 50, "information--road-bump--g1": 51, "warning--height-restriction--g2": 52, "complementary--obstacle-delineator--g2": 53, "warning--double-curve-first-left--g2": 54, "regulatory--no-overtaking--g5": 55, "information--food--g2": 56, "warning--divided-highway-ends--g2": 57, "regulatory--turn-right--g1": 58, "complementary--chevron-left--g1": 59, "regulatory--turn-left--g1": 60, "regulatory--no-parking-or-no-stopping--g3": 61, "warning--roundabout--g1": 62, "regulatory--no-heavy-goods-vehicles--g1": 63, "regulatory--maximum-speed-limit-60--g1": 64, "complementary--maximum-speed-limit-70--g1": 65, "regulatory--maximum-speed-limit-40--g1": 66, "warning--road-widens--g1": 67, "complementary--chevron-right--g1": 68, "warning--road-bump--g1": 69, "warning--uneven-road--g6": 70, "regulatory--maximum-speed-limit-50--g1": 71, "regulatory--no-parking--g5": 72, "regulatory--turn-left--g3": 73, "warning--railroad-crossing-without-barriers--g1": 74, "warning--junction-with-a-side-road-perpendicular-right--g3": 75, "regulatory--maximum-speed-limit-100--g1": 76, "warning--double-curve-first-right--g1": 77, "regulatory--maximum-speed-limit-5--g1": 78, "complementary--extent-of-prohibition-area-both-direction--g1": 79, "warning--road-narrows-left--g2": 80, "warning--children--g2": 81, "information--parking--g5": 82, "regulatory--no-u-turn--g3": 83, "warning--y-roads--g1": 84, "warning--trail-crossing--g2": 85, "regulatory--maximum-speed-limit-40--g3": 86, "regulatory--go-straight-or-turn-left--g1": 87, "regulatory--bicycles-only--g1": 88, "warning--texts--g2": 89, "regulatory--one-way-left--g1": 90, "warning--road-narrows-right--g2": 91, "regulatory--one-way-left--g3": 92, "regulatory--give-way-to-oncoming-traffic--g1": 93, "warning--double-curve-first-right--g2": 94, "complementary--maximum-speed-limit-30--g1": 95, "regulatory--no-u-turn--g1": 96, "warning--narrow-bridge--g1": 97, "regulatory--turn-right-ahead--g1": 98, "information--parking--g3": 99, "regulatory--maximum-speed-limit-70--g1": 100, "warning--uneven-road--g2": 101, "regulatory--shared-path-pedestrians-and-bicycles--g1": 102, "regulatory--pass-on-either-side--g2": 103, "regulatory--no-bicycles--g2": 104, "regulatory--no-pedestrians--g2": 105, "regulatory--no-stopping--g2": 106, "complementary--maximum-speed-limit-15--g1": 107, "warning--roundabout--g25": 108, "regulatory--go-straight-or-turn-left--g2": 109, "regulatory--no-parking--g2": 110, "regulatory--u-turn--g1": 111, "regulatory--keep-left--g1": 112, "regulatory--go-straight--g1": 113, "regulatory--keep-right--g2": 114, "regulatory--no-overtaking--g1": 115, "regulatory--no-parking-or-no-stopping--g2": 116, "information--telephone--g2": 117, "regulatory--road-closed-to-vehicles--g3": 118, "regulatory--no-left-turn--g3": 119, "warning--other-danger--g3": 120, "information--airport--g1": 121, "regulatory--no-right-turn--g1": 122, "regulatory--no-left-turn--g1": 123, "warning--railroad-crossing-without-barriers--g4": 124, "warning--texts--g1": 125, "information--end-of-built-up-area--g1": 126, "warning--junction-with-a-side-road-acute-left--g1": 127, "warning--divided-highway-ends--g1": 128, "regulatory--maximum-speed-limit-90--g1": 129, "regulatory--maximum-speed-limit-110--g1": 130, "warning--junction-with-a-side-road-perpendicular-left--g4": 131, "warning--other-danger--g1": 132, "regulatory--no-parking--g1": 133, "warning--hairpin-curve-left--g3": 134, "information--bus-stop--g1": 135, "warning--winding-road-first-left--g1": 136, "warning--turn-right--g1": 137, "regulatory--no-bicycles--g1": 138, "regulatory--no-heavy-goods-vehicles--g4": 139, "regulatory--weight-limit--g1": 140, "regulatory--radar-enforced--g1": 141, "regulatory--lane-control--g1": 142, "regulatory--turn-right--g2": 143, "warning--traffic-signals--g3": 144, "warning--added-lane-right--g1": 145, "warning--emergency-vehicles--g1": 146, "complementary--keep-right--g1": 147, "complementary--distance--g3": 148, "warning--winding-road-first-right--g3": 149, "warning--traffic-signals--g1": 150, "complementary--both-directions--g1": 151, "warning--junction-with-a-side-road-perpendicular-right--g1": 152, "regulatory--stop--g10": 153, "regulatory--maximum-speed-limit-20--g1": 154, "regulatory--maximum-speed-limit-25--g2": 155, "regulatory--no-motor-vehicles-except-motorcycles--g2": 156, "complementary--maximum-speed-limit-25--g1": 157, "complementary--maximum-speed-limit-55--g1": 158, "warning--curve-right--g2": 159, "regulatory--no-pedestrians--g1": 160, "complementary--maximum-speed-limit-35--g1": 161, "complementary--chevron-left--g3": 162, "regulatory--wrong-way--g1": 163, "complementary--chevron-left--g2": 164, "warning--double-reverse-curve-right--g1": 165, "warning--double-curve-first-left--g1": 166, "regulatory--maximum-speed-limit-30--g3": 167, "regulatory--no-bicycles--g3": 168, "regulatory--no-heavy-goods-vehicles--g2": 169, "warning--traffic-merges-right--g1": 170, "information--limited-access-road--g1": 171, "regulatory--maximum-speed-limit-55--g2": 172, "complementary--maximum-speed-limit-45--g1": 173, "warning--junction-with-a-side-road-perpendicular-left--g3": 174, "warning--pass-left-or-right--g2": 175, "complementary--one-direction-right--g1": 176, "regulatory--turn-left--g2": 177, "regulatory--stop--g2": 178, "information--pedestrians-crossing--g2": 179, "regulatory--maximum-speed-limit-80--g1": 180, "complementary--trucks--g1": 181, "complementary--tow-away-zone--g1": 182, "warning--roadworks--g1": 183, "regulatory--turn-left-ahead--g1": 184, "warning--horizontal-alignment-right--g1": 185, "warning--trams-crossing--g1": 186, "warning--double-turn-first-right--g1": 187, "warning--narrow-bridge--g3": 188, "warning--children--g1": 189, "warning--domestic-animals--g3": 190, "warning--winding-road-first-right--g1": 191, "information--central-lane--g1": 192, "regulatory--road-closed--g2": 193, "regulatory--no-vehicles-carrying-dangerous-goods--g1": 194, "warning--t-roads--g2": 195, "information--minimum-speed-40--g1": 196, "warning--school-zone--g2": 197, "regulatory--reversible-lanes--g2": 198, "regulatory--no-parking-or-no-stopping--g1": 199, "warning--traffic-merges-right--g2": 200, "complementary--maximum-speed-limit-20--g1": 201, "warning--slippery-road-surface--g2": 202, "warning--traffic-signals--g2": 203, "regulatory--one-way-left--g2": 204, "warning--bus-stop-ahead--g3": 205, "regulatory--no-u-turn--g2": 206, "regulatory--no-overtaking--g4": 207, "regulatory--keep-left--g2": 208, "information--stairs--g1": 209, "warning--two-way-traffic--g1": 210, "regulatory--no-turn-on-red--g1": 211, "warning--turn-right--g2": 212, "warning--road-narrows-right--g1": 213, "complementary--turn-left--g2": 214, "warning--texts--g3": 215, "information--end-of-motorway--g1": 216, "regulatory--pass-on-either-side--g1": 217, "complementary--chevron-right--g4": 218, "regulatory--no-left-turn--g2": 219, "complementary--chevron-right--g5": 220, "warning--trucks-crossing--g1": 221, "regulatory--no-motor-vehicle-trailers--g1": 222, "warning--road-bump--g2": 223, "regulatory--no-stopping--g8": 224, "regulatory--maximum-speed-limit-led-100--g1": 225, "complementary--obstacle-delineator--g1": 226, "regulatory--maximum-speed-limit-10--g1": 227, "complementary--priority-route-at-intersection--g1": 228, "regulatory--maximum-speed-limit-40--g6": 229, "regulatory--maximum-speed-limit-45--g1": 230, "regulatory--one-way-right--g1": 231, "regulatory--end-of-bicycles-only--g1": 232, "regulatory--roundabout--g1": 233, "information--living-street--g1": 234, "complementary--except-bicycles--g1": 235, "warning--bicycles-crossing--g1": 236, "warning--pedestrian-stumble-train--g1": 237, "regulatory--no-turn-on-red--g2": 238, "complementary--maximum-speed-limit-75--g1": 239, "information--safety-area--g2": 240, "warning--turn-left--g1": 241, "regulatory--road-closed--g1": 242, "warning--stop-ahead--g9": 243, "regulatory--mopeds-and-bicycles-only--g1": 244, "regulatory--end-of-speed-limit-zone--g1": 245, "information--interstate-route--g1": 246, "complementary--distance--g2": 247, "warning--roadworks--g3": 248, "complementary--chevron-left--g4": 249, "regulatory--triple-lanes-turn-left-center-lane--g1": 250, "warning--roadworks--g4": 251, "information--highway-exit--g1": 252, "regulatory--turn-right--g3": 253, "warning--winding-road-first-left--g2": 254, "warning--flaggers-in-road--g1": 255, "regulatory--no-motor-vehicles--g1": 256, "regulatory--no-right-turn--g2": 257, "regulatory--left-turn-yield-on-green--g1": 258, "regulatory--dual-lanes-go-straight-on-right--g1": 259, "regulatory--no-overtaking-by-heavy-goods-vehicles--g1": 260, "warning--pedestrians-crossing--g1": 261, "regulatory--no-straight-through--g1": 262, "complementary--chevron-right-unsure--g6": 263, "warning--offset-roads--g3": 264, "regulatory--maximum-speed-limit-120--g1": 265, "regulatory--go-straight-or-turn-right--g3": 266, "information--disabled-persons--g1": 267, "information--parking--g6": 268, "warning--loop-270-degree--g1": 269, "regulatory--dual-path-bicycles-and-pedestrians--g1": 270, "regulatory--buses-only--g1": 271, "complementary--accident-area--g3": 272, "complementary--pass-right--g1": 273, "warning--dual-lanes-right-turn-or-go-straight--g1": 274, "warning--road-narrows--g1": 275, "information--children--g1": 276, "regulatory--end-of-prohibition--g1": 277, "information--bike-route--g1": 278, "information--end-of-limited-access-road--g1": 279, "regulatory--no-mopeds-or-bicycles--g1": 280, "warning--wombat-crossing--g1": 281, "warning--crossroads-with-priority-to-the-right--g1": 282, "regulatory--maximum-speed-limit-led-80--g1": 283, "information--highway-interstate-route--g2": 284, "regulatory--stop-here-on-red-or-flashing-light--g1": 285, "warning--traffic-merges-left--g1": 286, "warning--hairpin-curve-right--g1": 287, "warning--equestrians-crossing--g2": 288, "information--gas-station--g3": 289, "regulatory--keep-right--g6": 290, "warning--road-widens-right--g1": 291, "warning--wild-animals--g4": 292, "regulatory--turn-right-ahead--g2": 293, "information--trailer-camping--g1": 294, "warning--railroad-crossing--g1": 295, "warning--domestic-animals--g1": 296, "warning--playground--g1": 297, "regulatory--no-stopping--g5": 298, "regulatory--end-of-maximum-speed-limit-70--g2": 299, "warning--traffic-merges-left--g2": 300, "regulatory--no-motorcycles--g1": 301, "information--hospital--g1": 302, "regulatory--no-stopping--g4": 303, "warning--falling-rocks-or-debris-right--g4": 304, "regulatory--shared-path-bicycles-and-pedestrians--g1": 305, "warning--railroad-intersection--g3": 306, "regulatory--minimum-safe-distance--g1": 307, "warning--steep-ascent--g7": 308, "warning--kangaloo-crossing--g1": 309, "warning--hairpin-curve-left--g1": 310, "regulatory--go-straight--g3": 311, "information--dead-end--g1": 312, "complementary--turn-right--g2": 313, "regulatory--stop-signals--g1": 314, "warning--falling-rocks-or-debris-right--g2": 315, "regulatory--passing-lane-ahead--g1": 316, "information--airport--g2": 317, "regulatory--no-turn-on-red--g3": 318, "warning--junction-with-a-side-road-perpendicular-left--g1": 319, "regulatory--width-limit--g1": 320, "information--gas-station--g1": 321, "regulatory--go-straight-or-turn-left--g3": 322, "information--camp--g1": 323, "regulatory--no-motorcycles--g2": 324, "regulatory--stop-here-on-red-or-flashing-light--g2": 325, "regulatory--no-turns--g1": 326, "regulatory--maximum-speed-limit-15--g1": 327, "regulatory--no-straight-through--g2": 328, "regulatory--maximum-speed-limit-led-60--g1": 329, "regulatory--maximum-speed-limit-100--g3": 330, "warning--wild-animals--g1": 331, "regulatory--no-motor-vehicles-except-motorcycles--g1": 332, "complementary--buses--g1": 333, "regulatory--parking-restrictions--g2": 334, "regulatory--bicycles-only--g3": 335, "regulatory--end-of-buses-only--g1": 336, "warning--two-way-traffic--g2": 337, "regulatory--end-of-priority-road--g1": 338, "information--no-parking--g3": 339, "information--telephone--g1": 340, "regulatory--truck-speed-limit-60--g1": 341, "warning--horizontal-alignment-left--g1": 342, "warning--railroad-crossing--g4": 343, "information--parking--g2": 344, "warning--slippery-motorcycles--g1": 345, "regulatory--maximum-speed-limit-50--g6": 346, "warning--pedestrians-crossing--g12": 347, "regulatory--do-not-block-intersection--g1": 348, "regulatory--end-of-maximum-speed-limit-70--g1": 349, "complementary--maximum-speed-limit-40--g1": 350, "regulatory--dual-lanes-go-straight-on-left--g1": 351, "warning--horizontal-alignment-right--g3": 352, "regulatory--end-of-no-parking--g1": 353, "warning--pedestrians-crossing--g10": 354, "warning--t-roads--g1": 355, "regulatory--detour-left--g1": 356, "warning--road-narrows-left--g1": 357, "warning--bicycles-crossing--g2": 358, "regulatory--dual-lanes-turn-left-or-straight--g1": 359, "regulatory--do-not-stop-on-tracks--g1": 360, "warning--roadworks--g2": 361, "warning--dip--g2": 362, "regulatory--maximum-speed-limit-65--g2": 363, "warning--road-narrows--g2": 364, "regulatory--no-heavy-goods-vehicles--g5": 365, "regulatory--road-closed-to-vehicles--g1": 366, "warning--railroad-intersection--g4": 367, "warning--railroad-crossing-with-barriers--g4": 368, "regulatory--no-pedestrians--g3": 369, "regulatory--maximum-speed-limit-25--g1": 370, "regulatory--text-four-lines--g1": 371, "regulatory--no-buses--g3": 372, "regulatory--bicycles-only--g2": 373, "warning--bicycles-crossing--g3": 374, "warning--uneven-roads-ahead--g1": 375, "warning--traffic-signals--g4": 376, "regulatory--no-pedestrians-or-bicycles--g1": 377, "information--lodging--g1": 378, "warning--shared-lane-motorcycles-bicycles--g1": 379, "regulatory--dual-lanes-turn-left-no-u-turn--g1": 380, "regulatory--no-hawkers--g1": 381, "regulatory--roundabout--g2": 382, "regulatory--weight-limit-with-trucks--g1": 383, "information--parking--g45": 384, "regulatory--dual-path-pedestrians-and-bicycles--g1": 385, "regulatory--no-heavy-goods-vehicles-or-buses--g1": 386, "regulatory--no-motor-vehicles--g4": 387, "warning--pedestrians-crossing--g11": 388, "warning--hairpin-curve-right--g4": 389, "warning--accidental-area-unsure--g2": 390, "warning--pass-left-or-right--g1": 391, "warning--restricted-zone--g1": 392, "regulatory--turning-vehicles-yield-to-pedestrians--g1": 393, "information--end-of-pedestrians-only--g2": 394, "regulatory--no-right-turn--g3": 395, "regulatory--dual-lanes-turn-right-or-straight--g1": 396, "complementary--maximum-speed-limit-50--g1": 397, "warning--playground--g3": 398, "warning--roadworks--g6": 399, "information--dead-end-except-bicycles--g1": 400}

os.makedirs(output_dataset, exist_ok=True)
os.makedirs(os.path.join(output_dataset, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dataset, "val"), exist_ok=True)
os.makedirs(os.path.join(output_dataset, "test"), exist_ok=True)

def convert_to_yolo(info_dict, directory_path):
    """ Convert extracted json info into a txt file in yolo format.

    :param info_dict: a dictionary with the relevant json info
    :param directory_path: path to directory the txts will be saved in
    """
    # code adapted from https://blog.paperspace.com/train-yolov5-custom-data/

    print_buffer = []

    # loop through all identified objects in `info_dict`
    for object in info_dict['objects']:
        class_id = class_to_id_map[object['label']]

        # convert coordinates to yolo format
        center_x = (object['xmin'] + object['xmax']) / 2
        center_y = (object['ymin'] + object['ymax']) / 2
        width = object['xmax'] - object['xmin']
        height = object['ymax'] - object['ymin']

        # normalize new coordinates
        image_w, image_h = info_dict['image_size']
        center_x /= image_w
        center_y /= image_h
        width /= image_w
        height /= image_h

        print_buffer.append("{} {} {} {} {}".format(class_id, center_x, center_y, width, height))

    save_file_name = os.path.join(directory_path, info_dict["filename"].replace("json", "txt"))
    print("\n".join(print_buffer), file=open(save_file_name, "w"))

def extract_from_json(json_name, directory_path):
    """ Extract info from json into a dictionary and return it.

    :param json_name: name of the json file we want to extract info from
    :param directory_path: path to directory with the json file
    :returns: a dictionary with the json filename, image size, and all classified objects in the image
    along with info about their labels, xmins, ymins, xmaxs, and ymaxs
    """
    # code adapted from https://blog.paperspace.com/train-yolov5-custom-data/

    # open json file and load it
    f = open(directory_path + "/" + json_name)
    data = json.load(f)

    # copy relevant json info about the image into a dictionary
    info_dict = {}
    info_dict['filename'] = json_name
    info_dict['image_size'] = tuple([data['width'], data['height']])

    # make a list of all identified objects' info and append it to `info_dict`
    info_dict['objects'] = []
    for object in data['objects']:
        object_dict = {}
        object_dict['label'] = object['label']
        object_dict['xmin'] = object['bbox']['xmin']
        object_dict['ymin'] = object['bbox']['ymin']
        object_dict['xmax'] = object['bbox']['xmax']
        object_dict['ymax'] = object['bbox']['ymax']
        info_dict['objects'].append(object_dict)

    # close file
    f.close()

    return info_dict

def jsons_to_yolos(directory_path, save_dir):
    """ Convert the dataset labels from json to yolo format.

    :param directory_path: path to directory with the labels directory
    """
    # make a list of all files in `directory_path` that are jsons
    directory_path = str(directory_path)
    jsons = [f for f in os.listdir(directory_path) if f.endswith('.json')]

    # convert each json file to a txt file in yolo format
    for json_name in jsons:
        info_dict = extract_from_json(json_name, directory_path)
        convert_to_yolo(info_dict, save_dir)

    return jsons

def move_labels(label_list, source_folder, dest_folder, jsons):
    with open(label_list, 'r') as file:
        for line in file:
            label_name = line.strip() + ".txt"
            source_path = os.path.join(source_folder, label_name)
            dest_path = os.path.join(output_dataset, dest_folder, label_name)
            try:
                shutil.move(source_path, dest_path)
            except FileNotFoundError:
                print(f"Image {label_name} not found in source folder.")

# jsons = jsons_to_yolos("/storage/mtsd_v2_fully_annotated/annotations/", "/storage/mapillary_annotation_dataset")
jsons = [f for f in os.listdir("/storage/mtsd_v2_fully_annotated/annotations/") if f.endswith('.json')]
move_labels(os.path.join(dataset_txt_split_dir, "val.txt"), "/storage/mapillary_annotation_dataset", "val", jsons)
move_labels(os.path.join(dataset_txt_split_dir, "test.txt"), "/storage/mapillary_annotation_dataset", "test", jsons)
move_labels(os.path.join(dataset_txt_split_dir, "train.txt"), "/storage/mapillary_annotation_dataset", "train", jsons)