import cv2

def resize(image, size):
    if isinstance(image, str):
        image = cv2.imread(image)

    # upscale
    if image.shape[0] * image.shape[1] < size[0] * size[1]:
        return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    # downscale
    else:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)