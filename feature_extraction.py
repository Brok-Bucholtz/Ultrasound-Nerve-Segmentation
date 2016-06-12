import re
import glob
import numpy as np

from PIL import Image
from tqdm import tqdm
from os.path import basename, dirname, join, splitext


def _get_file_root_name(file_path):
    file_root_name_expression = re.compile(r'/[^/]*\.', re.IGNORECASE)
    return file_root_name_expression.search(file_path).group(0)[1:-1]


def _normalization_pixel(pixel):
    return pixel/255.0


def _get_rootname(filename):
    return splitext(basename(filename))[0]


def _get_feature_label_images():
    TRAIN_FILES = './data/train/*.tif'
    features = []
    labels = []

    image_paths = [filename for filename in glob.glob(TRAIN_FILES) if not _get_rootname(filename).endswith('_mask')]
    with tqdm(desc='Reading Images from Disk', total=len(image_paths), unit='image') as progress_bar:
        for filename in image_paths:
            progress_bar.update()
            with Image.open(filename) as image:
                features.append([_normalization_pixel(pixel) for pixel in image.getdata()])
            with Image.open(join(dirname(filename), _get_rootname(filename)+'_mask.tif')) as image:
                labels.append(list(image.getdata()))

    return features, labels


def get_detection_data():
    new_labels = []

    features, labels = _get_feature_label_images()
    with tqdm(desc='Extracting Labels', total=len(labels), unit='image') as progress_bar:
        for label in labels:
            progress_bar.update()
            new_labels.append(255 in label)

    return features, new_labels


def _find_coord_mask(mask, height, width):
    rectangel = [[0,0], [0,0]]
    mask_transpose = np.transpose(np.reshape(mask, (height, width))).flatten()

    rectangel[0][0] = next((i for i, x in enumerate(mask_transpose) if x), 0) / height
    rectangel[0][1] = next((i for i, x in enumerate(mask) if x), 0) / width
    rectangel[1][0] = width - (next((i for i, x in enumerate(reversed(mask_transpose)) if x), len(mask)) / height)
    rectangel[1][1] = height-(next((i for i, x in enumerate(reversed(mask)) if x), len(mask)) / width)

    return rectangel


def get_rectangle_masks():
    IMAGE_HEIGHT = 420
    IMAGE_WIDTH = 580
    rectangle_masks = []
    _, labels = _get_feature_label_images()

    with tqdm(desc='Extracting Labels', total=len(labels), unit='image') as progress_bar:
        for label in labels:
            progress_bar.update()
            rectangle_coords = _find_coord_mask(label, IMAGE_HEIGHT, IMAGE_WIDTH)
            if rectangle_coords[1][0] and rectangle_coords[1][1]:
                rectangle_masks.append(rectangle_coords)
    return rectangle_masks


def get_sub_samples(height, width):
    IMAGE_HEIGHT = 420
    IMAGE_WIDTH = 580
    features = []
    labels = []

    features_labels = zip(*_get_feature_label_images())
    with tqdm(desc='Extracting Features and Labels', total=len(features_labels), unit='image') as progress_bar:
        for feature, label in features_labels:
            feature = np.array(feature).reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
            label = np.array(label).reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
            progress_bar.update()

            # Get Inner Rectangles
            for height_i in range(0, IMAGE_HEIGHT-height/2, height/2):
                for width_i in range(0, IMAGE_WIDTH-width/2, width/2):
                    # Make sure the Rectangle size is always height by width
                    if IMAGE_HEIGHT-height_i < height:
                        height_i = IMAGE_HEIGHT-height
                    if IMAGE_WIDTH - width_i < width:
                        width_i = IMAGE_WIDTH - width

                    features.append(feature[height_i:height_i+height, width_i:width_i+width].flatten())
                    # Set the label as the percentage of masked positive pixels to number of pixels in the rectangle
                    unique_counts = dict(zip(*np.unique(
                        label[height_i:height_i+height, width_i:width_i+width],
                        return_counts=True)))
                    labels.append(float(unique_counts.get(255, 0))/(height*width))

    return features, labels
