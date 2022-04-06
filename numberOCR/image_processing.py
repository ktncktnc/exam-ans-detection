import cv2
import numpy as np
import math


def image_to_binary(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_img, 20, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    return gray_img, binary_image


def crop_by_contour(gray_img):
    _, binary_image = cv2.threshold(gray_img, 15, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    contours = sorted(contours, key=cv2.contourArea)

    # Find bounding box and extract ROI
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        roi = gray_img[y:y + h, x:x + w]
        return roi


def find_rectangle(binary_image, max_rec):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_rec]

    bboxes = []
    for i in range(len(contours)):
        bbox = cv2.approxPolyDP(contours[i], 10, True)
        result = get_bbox_from_points(bbox)
        bboxes.append(result)

    return np.array(bboxes)


def draw_bboxes(img, bboxes, color=(0, 255, 0)):
    for bbox in bboxes:
        img = cv2.rectangle(img, bbox[0], bbox[1], color, 2)

    return img


def divide_bboxes(bboxes):
    result_bboxes = []

    for bbox in bboxes:
        n_part = calculate_ratio(abs(bbox[0][0] - bbox[1][0]) / abs(bbox[0][1] - bbox[1][1]))
        bbox = divide_bbox(bbox, n_part)

        for small_bbox in bbox:
            result_bboxes.append(small_bbox)

    return np.array(result_bboxes)


def crop_bbox(img, bbox):
    return img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]


def divide_bbox(bbox, n_parts=2):
    result_bboxes = np.zeros((n_parts, 2, 2))

    divided_width = math.floor((abs(bbox[1][0] - bbox[0][0])) / n_parts)

    for n in range(n_parts):
        result_bboxes[n] = np.array(
            [[bbox[0][0] + n * divided_width, bbox[0][1]], [bbox[0][0] + (n + 1) * divided_width, bbox[1][1]]])

    return result_bboxes.astype(np.uint32)


def display(img, framename="OpenCV Image", destroy_all=False):
    h, w = img.shape[0:2]
    new_width = 1200
    new_height = int(new_width * (h / w))
    img = cv2.resize(img, (new_width, new_height))
    cv2.imshow(framename, img)
    if destroy_all:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def calculate_ratio(ratio, threshold=0.5):
    for i in range(10, 1, -1):
        if ratio >= (i - 1 + threshold):
            return i

    return 1


def get_bbox_from_points(points):
    top = 100000
    left = 100000

    right = -1
    bot = -1
    for point in points:
        point = point[0]

        if point[0] < top:
            top = point[0]

        if point[0] > bot:
            bot = point[0]

        if point[1] < left:
            left = point[1]

        if point[1] > right:
            right = point[1]

    return np.array([[top, left], [bot, right]])


def normalize_filename(name: str):
    name = name.replace(" (", "_")
    name = name.replace(")", "")

    return name