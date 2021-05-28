import cv2
import matplotlib.pyplot as plt
import numpy as np


def cv2_imshow(image, is_bgr=True, window_name="default", window_size=(1000, 1000)):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, *window_size)
    if not is_bgr_or_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)


def plt_imshow(image, is_bgr=False):
    if is_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()


def tile_images(images, shape=(5, 5)):
    '''
    :param images: a set of images. Must be in shape of NHWC
    :param shape: the shape of tiled images
    :return: tiled images
    '''
    try:
        assert np.shape(images)[0] == shape[0]*shape[1], "ERROR: Number of images don't match the given tile shape"
    except AssertionError as e:
        return

    canvas = []
    for row in shape[0]:
        curr_row = []
        for col in shape[1]:
            idx = row*shape[1] + col
            curr_image = images[idx]
            curr_row.append(curr_image)
        curr_row = np.hstack(curr_row)
        canvas.append(curr_row)
    canvas = np.vstack(canvas)
    return canvas
