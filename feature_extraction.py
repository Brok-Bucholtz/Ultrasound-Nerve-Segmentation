import re
import glob

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


def get_rectangle_masks():
    IMAGE_WIDTH = 580
    rectangle_masks = []
    _, labels = _get_feature_label_images()

    with tqdm(desc='Extracting Labels', total=len(labels), unit='image') as progress_bar:
        for label in labels:
            progress_bar.update()
            mask_coord = [(i-IMAGE_WIDTH*(i/IMAGE_WIDTH), i/IMAGE_WIDTH) for i, pixel in enumerate(label) if pixel != 0]
            if mask_coord:
                mask_xs, mask_ys = zip(*mask_coord)
                rectangle_masks.append(((min(mask_xs), mask_ys[0]), (max(mask_xs), mask_ys[len(mask_ys)-1])))
    return rectangle_masks
