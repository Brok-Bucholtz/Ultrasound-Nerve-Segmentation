from feature_extraction import get_rectangle_masks
import matplotlib.pyplot as plt


def _plot_rectangle_masks():
    heights = []
    widths = []
    for cord_1, cord_2 in get_rectangle_masks():
        widths.append(cord_2[0]-cord_1[0])
        heights.append(cord_2[1] - cord_1[1])

    plt.plot(heights, widths, 'bo')
    plt.show()


if __name__ == '__main__':
    _plot_rectangle_masks()
