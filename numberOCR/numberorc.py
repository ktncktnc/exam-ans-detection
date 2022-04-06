from numberOCR.deepmodel import *
from numberOCR.image_processing import *
import numpy as np
import cv2


def extract(predictor, img):
    result_dict = dict()
    md = ""
    sbd = ""
    md_conf = sbd_conf = 10.0

    gray_img, binary_img = image_to_binary(img)
    recs = np.flip(find_rectangle(binary_img, 2), 0)
    final_bboxes = divide_bboxes(recs)

    cropped_samples = []
    empty_sample = []
    for i, bbox in enumerate(final_bboxes):
        cropped = crop_bbox(gray_img, bbox)
        cropped = cropped[10:-10, 10:-10]
        kernel = np.ones((3, 3), np.uint8)
        cropped = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, kernel)

        maxvalue = np.max(cropped.astype('float32'))
        if maxvalue < 30:
            empty_sample.append(i)
            continue

        cropped[cropped >= 30] = 255
        cropped = cv2.resize(cropped, (28, 28))
        cropped = np.expand_dims(cropped.astype('float32') / 255, -1)
        cropped_samples.append(cropped)

    if len(cropped_samples) > 0:
        values, confidences = predict(predictor, np.array(cropped_samples))
    else:
        values = confidences = np.array([])

    n_empty = 0
    for i in range(len(empty_sample) + values.shape[0]):
        if i in empty_sample:
            n_empty += 1
            if i < 2:
                md = md + "-"
            else:
                sbd = sbd + "-"
            continue

        if i < 2:
            md = md + str(values[i - n_empty])
            if confidences[i - n_empty] < md_conf:
                md_conf = confidences[i - n_empty]

        else:
            sbd = sbd + str(values[i - n_empty])
            if confidences[i - n_empty] < sbd_conf:
                sbd_conf = confidences[i - n_empty]

    result_dict['MD'] = md
    result_dict['SBD'] = sbd
    result_dict['MD_conf'] = md_conf
    result_dict['SBD_conf'] = sbd_conf

    return result_dict
