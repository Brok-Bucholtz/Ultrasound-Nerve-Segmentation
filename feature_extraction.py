import re
import glob

from PIL import Image
from tqdm import tqdm


def _get_file_root_name(file_path):
    file_root_name_expression = re.compile(r'/[^/]*\.', re.IGNORECASE)
    return file_root_name_expression.search(file_path).group(0)[1:-1]


def _get_feature_label_images():
    TRAIN_FILES = './data/train/*.tif'
    mask_expression = re.compile(r'_mask\.tif', re.IGNORECASE)
    features = {}
    labels = {}

    image_paths = glob.glob(TRAIN_FILES)
    with tqdm(desc='Reading Images from Disk', total=len(image_paths), unit='image') as progress_bar:
        for file_name in image_paths:
            progress_bar.update()
            with Image.open(file_name) as image:
                root_name = _get_file_root_name(file_name)

                if mask_expression.search(file_name):
                    labels[root_name[:-5]] = image.getdata()
                else:
                    features[root_name] = image.getdata()

    assert len(features) == len(labels)
    return [(features[root_name], labels[root_name]) for root_name, label in labels.items()]


def get_detection_data():
    features = []
    labels = []

    features_labels = _get_feature_label_images()
    with tqdm(desc='Extracting Features', total=len(features_labels), unit='image') as progress_bar:
        for feature, label in features_labels:
            progress_bar.update()
            features.append(list(feature)[:784])
            extracted_label = [1.0, 0.0] if 255 in label else [0.0, 1.0]
            labels.append(extracted_label)

    return features, labels


