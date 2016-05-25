from PIL import Image
import glob


def _get_masks():
    TRAIN_MASKS = './data/train/*_mask.tif'
    return [Image.open(file_name) for file_name in glob.glob(TRAIN_MASKS)]






def _get_mask_labels():
    mask_labels = []
    for image in _get_masks():
        mask_labels.append((image.filename, 255 in image.getdata()))

    return mask_labels
