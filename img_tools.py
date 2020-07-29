import cv2
import numpy as np


def show_images(img1, img2, name="Images"):

    cv2.imshow(name, np.hstack([img1, img2]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_image(img1, winname="Image"):
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, 40, 30)
    cv2.imshow(winname, img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_img(img, scale=600.0):

    r = scale / img.shape[1]
    dim = (int(scale), int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img


def color_detect(img, lower, upper):

    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(hsv_image, lower, upper)

    return mask

def get_center(h, w, radius=10):
    half_h = int(h / 2)
    half_w = int(w / 2)

    ix = half_w - radius
    iy = half_h - radius

    jx = half_w + radius
    jy = half_h + radius
    # cv2.rectangle(img, (ix, iy), (jx, jy), (0, 255, 0), 1)
    return [(ix, iy), (jx, jy)]

def find_min_conts(img, width):
    blurred = cv2.pyrMeanShiftFiltering(
        img, 27, width)
    imgray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(
        imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def mean_color_in_rect(rect, img):

    roi = img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
    hsv_image = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v, _ = np.uint8(cv2.mean(hsv_image))
    return (h, s, v)

def take_center_cont(conts, h, w):
    x = w / 2
    y = h / 2
    for c in conts:
        res = cv2.pointPolygonTest(c, (x, y), measureDist=False)
        if res >= 0:
            return c
    return None


def specify_range(diff, color, isLower):
    h = color[0]
    s = color[1]
    v = color[2]

    if isLower:
        diff = [h - diff[0], s -
                diff[1], v - diff[2]]
    else:
        diff = [
            h + diff[0], s + diff[1], v + diff[2]]

    diff = np.clip(
        diff, 0, 255)
    return diff
