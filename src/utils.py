import math
import numpy as np
from functools import cmp_to_key
import os
import cv2
import pandas as pd
from itertools import chain


######### Line funcs
def merge_lines(lines, length_thresh=90, y_low_thresh=200, y_high_thresh=800):
    tmp_lines = []
    for line in lines:
        if check_valid_line(line[0], y_low_thresh, y_high_thresh):
            tmp_lines.append(line[0])

    lines = tmp_lines
    lines = partition_lines(lines, length_thresh)
    lines = partition_lines(lines, length_thresh)
    lines = sort_lines(lines)
    # print(lines)
    lines = preprocess_lines(lines)
    clusters = clustering_lines(lines)
    lines = insert_missing_lines(clusters, use_avg_h=True)
    lines = sort_lines(lines)
    return lines


def partition_lines(lines, length_thresh=90):
    partitions = []
    duplicated_idx = []

    lines_len = len(lines)
    for i in range(lines_len):
        line_i = lines[i]
        partition = [i]
        if i in duplicated_idx:
            continue

        for j in range(i + 1, lines_len):
            line_j = lines[j]
            res = check_equal_lines(line_i, line_j)
            if res:
                duplicated_idx.append(j)
                partition.append(j)

        partitions.append(partition)

    result_lines = []
    for partition in partitions:
        max_x = -1
        min_x = 99999
        total_y = 0
        for idx in partition:
            line = lines[idx]
            min_x = min(min_x, line[0])
            max_x = max(max_x, line[2])

            total_y += line[1] + line[3]

        y = int(total_y / (len(partition) * 2))
        merged_line = [min_x, y, max_x, y]

        if check_length_line(merged_line, length_thresh):
            result_lines.append(merged_line)

    return result_lines


def check_valid_line(line, y_low_thresh=200, y_high_thresh=800, low_thresh=30, high_thresh=150):
    # Not in interested zone
    if line[1] < y_low_thresh or line[1] > y_high_thresh:
        return False

    # Not horizontal line
    v = np.arctan2(line[3] - line[1], line[2] - line[0]) * 180. / np.pi
    if low_thresh < abs(v) < high_thresh:
        return False

    return True


def check_length_line(line, thresh=50):
    if line_length(line) < thresh:
        return False

    return True


def check_equal_lines(l1, l2, low_thresh=4, high_thresh=174):
    l3 = [l1[0], l1[1], l2[2], l2[3]]

    if l3[0] > l3[2]:
        return False

    angle_l3 = np.arctan2(l3[3] - l3[1], l3[2] - l3[0]) * 180. / np.pi
    if low_thresh < abs(angle_l3) < high_thresh:
        return False

    length_l1 = line_length(l1)
    length_l2 = line_length(l2)
    length_l3 = line_length(l3)

    if length_l3 > 650:
        return False

    if length_l1 + length_l2 < length_l3 - 10:
        return False

    return True


def sort_lines(lines, confs):
    confs = [x for _, x in sorted(zip(lines, confs), key=cmp_to_key(compare_lines))]
    lines.sort(key=cmp_to_key(compare_lines))
    return lines, confs


def compare_lines(l1, l2, threshold=200):
    if type(l1) is tuple or type(l1) is tuple:
        l1, conf1 = l1
        l2, conf2 = l2

    if l1[0] - l2[0] > threshold:
        return 1

    if l1[2] - l2[2] > threshold:
        return 1

    if l2[0] - l1[0] > threshold:
        return -1

    if l2[2] - l1[2] > threshold:
        return -1

    if l1[1] - l2[1] > 5:
        return 1

    if l2[1] - l1[1] > 5:
        return -1

    return 0


def preprocess_lines(lines=None):
    if lines is None:
        lines = []
    min_y = 999999
    max_len = 0

    for line in lines:
        min_y = min(min_y, line[1])
        max_len = max(max_len, line_length(line))

    i = 0
    while True:
        if i >= len(lines):
            break

        line = lines[i]
        if (line[1] <= min_y + 30 or line[3] <= min_y + 30):
            del lines[i]

        else:
            i += 1

    return lines


def line_length(line):
    return math.sqrt(math.pow(line[2] - line[0], 2) + math.pow(line[3] - line[1], 2))


# ####### Img funcs
def crop_bboxes(img, lines, filename, savepath='2022\\crop_result', yolo_path=None, answers=None):
    answer_mapper = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4
    }

    assert (yolo_path is None or (answers is not None))
    name = os.path.splitext(filename)[0]

    write_i = 0
    break_flag = False

    height, width = img.shape[0], img.shape[1]

    if yolo_path is not None:
        txt_file = open(os.path.join(yolo_path, name + '.txt'), mode='w')

    for i in range(len(lines) - 1):
        line_1 = lines[i]
        line_2 = lines[i + 1]
        if line_1[1] > line_2[1]:
            continue

        n_split = 1

        # if len(heights) > 0:
        #     avg_height = np.average(heights)
        #     if line_2[1] - line_1[1] > avg_height + 40:
        #         n = (line_2[1] - line_1[1])/avg_height
        #         for divide_v in range(7, 1, -1):
        #             if n > divide_v - 0.5:
        #                 n_split = divide_v
        #                 break
        #
        #     else:
        #         heights.append(line_2[1] - line_1[1])
        # else:
        #     heights.append(line_2[1] - line_1[1])

        split_h = int((line_2[1] - line_1[1]) / n_split)
        for split in range(n_split):
            y_center = (line_1[1] + split_h * (split + 0.5)) / height
            x_center = (int((min(line_1[0], line_2[0]) + max(line_1[2], line_2[2])) / 2)) / width
            w = (max(line_1[2], line_2[2]) - min(line_1[0], line_2[0])) / width
            h = split_h / height

            bbox = img[line_1[1] + split_h * split: line_1[1] + split_h * (split + 1),
                   min(line_1[0], line_2[0]): max(line_1[2], line_2[2])]

            if bbox.shape[0] > 0 and bbox.shape[1] > 0:
                if savepath is not None:
                    cv2.imwrite(os.path.join(savepath, name + f"_{write_i}" + ".jpg"), bbox)

                if yolo_path is not None:
                    ans = answers.loc[answers['filename'] == name].iloc[0]
                    txt_file.write("{an} {x} {y} {w} {h}\n".format(x=x_center, y=y_center, w=w, h=h, an=answer_mapper[ans['No.' + str(write_i + 1)]]))

                write_i += 1
                if write_i == 25:
                    break_flag = True
                    break

        if break_flag:
            break

    if write_i != 25:
        print("check split: " + filename)

    if yolo_path is not None:
        txt_file.close()


def clustering_lines(lines):
    clusters = []
    cluster = []

    prev_y = -1
    for line in lines:
        if line[1] < prev_y:
            clusters.append(cluster)
            cluster = []

        prev_y = line[1]
        cluster.append(line)

    if len(cluster) > 0:
        clusters.append(cluster)

    return clusters


def avg_size(line_clusters):
    sum_w = sum_h = 0
    total_v = 0
    for cluster in line_clusters:
        total_v += len(cluster) - 1
        for i in range(len(cluster) - 1):
            sum_w += max(cluster[i][2], cluster[i + 1][2]) - min(cluster[i][0], cluster[i + 1][0])
            sum_h += cluster[i + 1][1] - cluster[i][1]

    return int(sum_h / total_v), int(sum_w / total_v)


def get_multiple(value, threshold=0.7):
    for i in range(10, -1, -1):
        if value > i - threshold:
            return i


def get_min_max_y(line_clusters):
    min_y = 9999999
    max_y = 0
    for cluster in line_clusters:
        min_y = min(min_y, cluster[0][1])
        max_y = max(max_y, cluster[-1][1])

    return min_y, max_y


def get_avg_min_max_x(cluster):
    min_x = 0
    max_x = 0
    for line in cluster:
        min_x += line[0]
        max_x += line[2]

    return int(min_x / len(cluster)), int(max_x / len(cluster))


def insert_missing_lines(line_clusters, n_line=16, use_avg_h=True):
    avg_h, avg_w = avg_size(line_clusters)
    min_y, max_y = get_min_max_y(line_clusters)
    result_clusters = []

    for cluster in line_clusters:
        avg_min_x, avg_max_x = get_avg_min_max_x(cluster)

        # Missing lines in the middle
        result_cluster = cluster.copy()
        if len(cluster) < n_line:
            for i in range(len(cluster) - 1):
                line_i = cluster[i]
                line_j = cluster[i + 1]

                mul = 0
                h = line_j[1] - line_i[1]
                if h > avg_h + 50:
                    mul = get_multiple(h / avg_h) - 1

                if mul > 0:
                    split_h = int(h / mul)
                else:
                    split_h = avg_h
                if use_avg_h:
                    split_h = avg_h

                for m in range(mul):
                    result_cluster.insert(i + 1 + m, [avg_min_x, line_i[1] + split_h * (m + 1), avg_max_x,
                                                      line_i[1] + split_h * (m + 1)])

        cluster = result_cluster

        # Missing in the beginning
        if len(cluster) < n_line and cluster[0][1] > min_y + 40:
            mul = get_multiple((cluster[0][1] - min_y) / avg_h)
            for m in range(mul):
                cluster.insert(m, [avg_min_x, min_y + avg_h * m, avg_max_x, min_y + avg_h * m])

        # Missing in the end
        # print(len(cluster))
        # print(cluster[-1][1])
        # print(max_y)
        if len(cluster) < n_line and cluster[-1][1] < max_y - 40:
            mul = get_multiple((max_y - cluster[-1][1]) / avg_h)
            for m in range(mul):
                cluster.append([avg_min_x, max_y + avg_h * m, avg_max_x, max_y + avg_h * m])

        result_clusters.append(cluster)

    return list(chain.from_iterable(result_clusters))


def nms(bounding_boxes, confidence_score, threshold=0.8):
    if len(bounding_boxes) == 0:
        return [], []

        # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score

def remove_overlapped_bbox(bounding_boxes, confidence_score, threshold=25):
    """
    :param bounding_boxes: sorted
    :param confidence_score: sorted based on bounding bboxes
    :return:
    """
    result_bboxes = []
    result_confs = []
    flag = False
    for i in range(len(bounding_boxes) - 1):
        if flag:
            flag = False
            continue

        box_1 = bounding_boxes[i]
        box_2 = bounding_boxes[i + 1]

        if threshold <= box_1[3] - box_2[1] < max(box_2[2] - box_2[0], box_1[2] - box_1[0]):
            flag = True

        result_bboxes.append(box_1)
        result_confs.append(confidence_score[i])

    result_bboxes.append(bounding_boxes[-1])
    result_confs.append(confidence_score[-1])
    return result_bboxes, result_confs
