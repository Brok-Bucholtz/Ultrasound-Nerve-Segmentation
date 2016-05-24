from PIL import Image
import glob


def _get_masks():
    TRAIN_MASKS = './data/train/*_mask.tif'
    return [Image.open(file_name) for file_name in glob.glob(TRAIN_MASKS)]


def _get_rectangle_masks():
    rectangle_masks = []
    for image in _get_masks():
        rectangle_mask = ((0,0), (0,0))
        mask_coord = [(i-image.width*(i/image.width), i/image.width) for i, pixel in enumerate(image.getdata()) if pixel != 0]

        if mask_coord:
            mask_xs, mask_ys = zip(*mask_coord)
            rectangle_mask = ((min(mask_xs), mask_ys[0]), (max(mask_xs), mask_ys[len(mask_ys)-1]))

        rectangle_masks.append(rectangle_mask)
    return rectangle_masks


def run():
    print _get_rectangle_masks()


if __name__ == '__main__':
    run()
