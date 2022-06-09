from src.utils import *
import os
from tqdm import tqdm


def detect_line(filepath, with_morpho_close=True, save_path=None, yolo_path=None, answers=None):
    assert (yolo_path is None or (answers is not None))

    filename = os.path.basename(filepath)

    img = cv2.imread(filepath)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_img = np.uint8((cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) < 190) * 255)
    if with_morpho_close:
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, np.ones((3, 3)))

    lines = cv2.HoughLinesP(binary_img, 1, 10 * np.pi / 180, 30, None, 30, 20)
    if lines is not None:
        lines = merge_lines(lines, 350, 800, 2600)

        if len(lines) == 48:
            crop_bboxes(img, lines, filename, save_path, yolo_path, answers)

        for i in range(0, len(lines)):
            line = lines[i]
            cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 5, cv2.LINE_AA)
        cv2.putText(img, str(len(lines)), (150, 350), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imwrite(os.path.join("../all/result", filename), img)
    if len(lines) != 48:
        print(filepath)
        return True
    return False


if __name__ == '__main__':
    files = os.listdir("../all/skew_correction")
    error_files = [
        "20220313173503863_0042", "20220313173612205_0012", "20220313173755479_0006", "20220313173755479_0008",
        "20220313173755479_0036", "20220313173850433_0012", "20220313173919210_0028", "20220313173945957_0030",
        "20220313174009616_0014", "20220313174731646_0030", "20220313174807707_0032", "20220313174807707_0034",
        "20220313174832842_0014", "20220313174910489_0052", "20220313174947850_0042", "20220313175010879_0006",
        "20220313175010879_0012", "20220313175010879_0030", "20220313175050718_0006", "20220313175050718_0032",
        "20220313175050718_0052", "20220313175128214_0022", "20220313175128214_0026", "20220313175219011_0028",
        "20220313175219011_0030", "20220313175347348_0010", "20220313175418427_0022", "20220313175445287_0026",
        "de_01_0032", "de_04_0038", "de_04_0064"
    ]

    answers = pd.read_excel("../all/all_answer_result.xlsx").reset_index()

    total = 0
    for _filepath in tqdm(files):
        base_name = os.path.basename(_filepath)[:-5]
        if base_name in error_files:
            continue

        with_morpho = True
        total += detect_line(os.path.join("../all/skew_correction", _filepath), with_morpho, None,
                             "../all/yolo_data", answers)

    print("Total = " + str(total))

    #detect_line("all/skew_corrected_answer/de_04_0020.jpeg", True, None, "all/yolo_data", answers)
