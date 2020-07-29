import numpy as np
import cv2
import sys
import img_tools
from operator import add, sub


class EqualMeanGreyValueCounter:

    def convert_to_grayscale(self):

        self.gray_img_soil = cv2.cvtColor(
            self.img_only_soil, cv2.COLOR_BGR2GRAY)
        self.gray_img_paper = cv2.cvtColor(
            self.paper, cv2.COLOR_BGR2GRAY)

    def __init__(self, img):
        self.init_vars()
        self.img = img_tools.resize_img(img, self.RESIZE_SCALE)
        self.h, self.w = self.img.shape[:2]
        self.detect_soil()
        self.detect_paper()
        self.convert_to_grayscale()
        self.divide_into_locations()
        self.count_equal_mean_gray_value()

    def init_vars(self):

        self.RESIZE_SCALE = 400.0
        self.WIDE_CONTOUR = 10

        self.LOWER_HSV_BOUND_SOIL = [5, 20, 25]
        self.UPPER_HSV_BOUND_SOIL = [35, 255, 160]
        self.LOWER_HSV_BOUND_PAPER = [0, 0, 60]
        self.UPPER_HSV_BOUND_PAPER = [180, 40, 255]

        self.LOWER_DIFF_HSV_SOIL = (15, 20, 60)
        self.UPPER_DIFF_HSV_SOIL = (15, 65, 55)

        self.DIFF_HSV_PAPER = (255, 35, 55)

        self.radius_of_sq = 20

        self.parts = list()

    def detect_soil(self):

        self.detect_soil_by_color()
        self.keep_only_soil()
        self.detect_soil_by_contour()

    def detect_paper(self):

        self.detect_paper_by_color(self.center_cont)
        self.keep_only_paper()

    def divide_into_locations(self):

        self.l = np.array(self.divide_into_4_sq(self.gray_img_paper))
        self.s = np.array(self.divide_into_4_sq(self.gray_img_soil))

    def detect_soil_by_color(self):

        rect = img_tools.get_center(self.h, self.w, self.radius_of_sq)

        color = img_tools.mean_color_in_rect(rect, self.img)
        l_diff = self.LOWER_DIFF_HSV_SOIL
        u_diff = self.UPPER_DIFF_HSV_SOIL
        self.LOWER_HSV_BOUND_SOIL = img_tools.specify_range(
            l_diff, color, True)
        self.UPPER_HSV_BOUND_SOIL = img_tools.specify_range(
            u_diff, color, False)

    def detect_soil_by_contour(self):

        contours_soil = img_tools.find_min_conts(
            self.img_only_soil, self.WIDE_CONTOUR)
        self.center_cont = img_tools.take_center_cont(
            contours_soil, self.h, self.w)
        self.mask_only_soil = np.zeros((self.h, self.w, 1), np.uint8)
        cv2.drawContours(self.mask_only_soil, [
                         self.center_cont], -1, (255, 255, 255), -1)
        self.img_only_soil = cv2.bitwise_and(
            self.img, self.img, mask=self.mask_only_soil)

    def detect_paper_by_color(self, cont):
        rect = self.get_rect_for_paper_color(cont)
        paper_color = img_tools.mean_color_in_rect(
            (rect[0], rect[1]), self.img)

        l_diff = self.DIFF_HSV_PAPER
        u_diff = self.DIFF_HSV_PAPER

        self.LOWER_HSV_BOUND_PAPER = img_tools.specify_range(
            l_diff, paper_color, True)
        self.UPPER_HSV_BOUND_PAPER = img_tools.specify_range(
            u_diff, paper_color, False)

    def get_rect_for_paper_color(self, cont):

        coord_right = cont[cont[:, :, 0].argmax()][0]
        coord_top = cont[cont[:, :, 1].argmin()][0]
        coord_bot = cont[cont[:, :, 1].argmax()][0]
        coord_i = np.array([0, 0])
        coord_j = np.array([0, 0])
        coord_i[0] = coord_right[0] + 10
        coord_i[1] = coord_top[1]
        coord_j[0] = coord_i[0] + (self.radius_of_sq * 0.80)
        coord_j[1] = coord_bot[1]
        return [tuple(coord_i), tuple(coord_j)]

    def keep_only_paper(self):

        lower = np.array(self.LOWER_HSV_BOUND_PAPER, dtype="uint8")
        upper = np.array(self.UPPER_HSV_BOUND_PAPER, dtype="uint8")
        grey = cv2.cvtColor(
            self.img_only_soil, cv2.COLOR_BGR2GRAY)
        mask = 255 - self.mask_only_soil
        mask2 = cv2.inRange(self.hsv_image, lower, upper)
        mask = cv2.bitwise_and(mask, mask2)
        self.paper = cv2.bitwise_and(self.img, self.img, mask=mask)

    def keep_only_soil(self):

        lower = np.array(self.LOWER_HSV_BOUND_SOIL, dtype="uint8")
        upper = np.array(self.UPPER_HSV_BOUND_SOIL, dtype="uint8")
        self.hsv_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(self.hsv_image, lower, upper)

        self.img_only_soil = cv2.bitwise_and(self.img, self.img, mask=mask)

    def count_equal_mean_gray_value(self):

        res = 255 - self.l
        self.eq_mean_grey_values = np.add(self.s, res)
        self.mean_of_eq_mean_grey_values = np.average(self.eq_mean_grey_values)
        # print('Equal Mean Gray Value', self.mean_of_eq_mean_grey_values, '\n')

    def count_mean_gray_value(self, img):

        nondarkpix = cv2.countNonZero(img)

        mean = np.sum(img) / nondarkpix
        return mean

    def convert_to_triangle_image(self, img, coord):

        pt1 = coord[0]
        pt2 = coord[1]
        pt3 = coord[2]

        cv2.circle(img, pt1, 0, (0, 0, 0), -1)
        cv2.circle(img, pt2, 0, (0, 0, 0), -1)
        cv2.circle(img, pt3, 0, (0, 0, 0), -1)

        triangle_cnt = np.array([pt1, pt2, pt3])

        cv2.drawContours(img, [triangle_cnt], 0, (0, 0, 0), -1)

        return img

    def count_diagonally_mean(self, sq_img, isPrincipalDiag):

        (h, w) = sq_img.shape[:2]

        upper_triangle = sq_img.copy()
        lower_triangle = sq_img.copy()

        if isPrincipalDiag:
            upper_triangle = self.convert_to_triangle_image(
                upper_triangle, [(0, 0), (w, h), (0, h)])
            lower_triangle = self.convert_to_triangle_image(
                lower_triangle, [(0, 0), (w, h), (w, 0)])

        else:
            upper_triangle = self.convert_to_triangle_image(
                upper_triangle, [(0, h), (w, 0), (w, h)])
            lower_triangle = self.convert_to_triangle_image(
                lower_triangle, [(0, h), (w, 0), (0, 0)])

        self.parts.append(upper_triangle)
        self.parts.append(lower_triangle)

        mean_upper = self.count_mean_gray_value(upper_triangle)
        mean_lower = self.count_mean_gray_value(lower_triangle)

        return (mean_upper, mean_lower)

    def divide_into_4_sq(self, image):
        mean_list = list()

        half_w = int(self.w / 2)
        half_h = int(self.h / 2)

        upper_left_sq = image[0: half_h, 0:half_w]
        lower_right_sq = image[half_h:, half_w:]
        mean_list.extend(self.count_diagonally_mean(
            upper_left_sq, isPrincipalDiag=True))
        mean_list.extend(self.count_diagonally_mean(
            lower_right_sq, isPrincipalDiag=True))

        upper_right_sq = image[0: half_h, half_w:]
        lower_left_sq = image[half_h:, 0:half_w]
        mean_list.extend(self.count_diagonally_mean(
            upper_right_sq, isPrincipalDiag=False))
        mean_list.extend(self.count_diagonally_mean(
            lower_left_sq, isPrincipalDiag=False))

        return mean_list

    def show_parts(self):
        for i, img in enumerate(self.parts, 0):

            cv2.imshow(str(i), img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @classmethod
    def input_image(cls):

        if len(sys.argv) < 1:

            print('Usage: python script.py input.png ')
            sys.exit()

        image = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)

        return cls(image)
