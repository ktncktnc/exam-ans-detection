import json
import os.path
from src.utils import *
import re


def yolo_result_to_xlsx(answers_path='result_answer.json',
                        save_path='all/answer_result.xlsx',
                        img_path='all/skew_corrected_answer', n_ans=25, threshold=0.995, save_img=True,
                        save_img_path='temp/result'):
    if save_img and not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

    ans_df = pd.DataFrame()
    prob_df = pd.DataFrame()

    ans_df_keys = {"filename": ""}
    prob_df_keys = {"filename": ""}
    for i in range(n_ans):
        ans_df_keys["No." + str(i + 1)] = ""
        prob_df_keys["No." + str(i + 1)] = ""
    ans_df_keys["Check_All"] = ""

    ans_df = ans_df.join(pd.DataFrame(ans_df_keys, index=ans_df.index))
    prob_df = prob_df.join(pd.DataFrame(prob_df_keys, index=prob_df.index))

    answer_mapper = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E"
    }
    with open(answers_path, "r") as jsonfile:
        results = json.load(jsonfile)

    for result in results:
        basename = os.path.basename(result['filename'])
        filepath = os.path.join(img_path, basename)

        img = cv2.imread(filepath)
        height, width = img.shape[0], img.shape[1]

        bboxes = []
        confs = []

        for idx, ans in enumerate(result['objects']):
            coords = ans['relative_coordinates']
            x_1 = int((coords['center_x'] - coords['width'] / 2) * width)
            y_1 = int((coords['center_y'] - coords['height'] / 2) * height)

            x_2 = int((coords['center_x'] + coords['width'] / 2) * width)
            y_2 = int((coords['center_y'] + coords['height'] / 2) * height)

            bboxes.append([x_1, y_1, x_2, y_2, ans['class_id']])
            confs.append(ans['confidence'])

        bboxes, confs = nms(bboxes, confs)
        bboxes, confs = sort_lines(bboxes, confs)
        bboxes, confs = remove_overlapped_bbox(bboxes, confs)

        if save_img:
            for bbox in bboxes:
                x_1, y_1, x_2, y_2, class_id = bbox
                cv2.rectangle(img, (x_1, y_1), (x_2, y_2), (0, 0, 255), 5)
                cv2.putText(img, answer_mapper[class_id], (x_1 - 50, int((y_1 + y_2) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(save_img_path, basename), img)

        confs = np.array(confs)
        answer = np.array(bboxes)[:, -1]
        answer = np.vectorize(answer_mapper.get)(answer)

        check_all = False
        n_predicted = answer.shape[0]
        if n_predicted < n_ans:
            answer = np.append(answer, ["E"] * (n_ans - n_predicted))
            confs = np.append(confs, [0.0] * (n_ans - n_predicted))
            check_all = True

        elif n_predicted > n_ans:
            answer = answer[:n_ans]
            confs = confs[:n_ans]
            check_all = True

        if check_all:
            print("Please check: " + basename)

        answer = np.append(answer, str(check_all))

        ans_df = ans_df.append(pd.DataFrame(np.append(basename, answer).reshape(1, -1), columns=ans_df.columns),
                               ignore_index=True)
        prob_df = prob_df.append(pd.DataFrame(np.append(basename, confs).reshape(1, -1), columns=prob_df.columns),
                                 ignore_index=True)

    writer = pd.ExcelWriter(save_path, engine='xlsxwriter', options={'strings_to_numbers': True})
    ans_df.to_excel(writer, sheet_name='Ans')
    prob_df.to_excel(writer, sheet_name='Prob')
    workbook = writer.book
    worksheet = writer.sheets['Ans']

    condition_format = workbook.add_format({'bg_color': '#ffff00'})
    condition_format1 = workbook.add_format({'bg_color': '#eb4034'})
    worksheet.conditional_format('C2:AA2000', {'type': 'formula',
                                               'criteria': '=Prob!C2:AA514<{threshold}'.format(threshold=threshold),
                                               'format': condition_format})

    worksheet.conditional_format('AB2:AB2000', {'type': 'cell',
                                                'criteria': '=',
                                                'value': '"True"',
                                                'format': condition_format1})

    writer.save()


def answer_filename_to_number(name, answer_number_dif=-1, ended='_cut'):
    title, ext = os.path.splitext(os.path.basename(name))
    number = re.findall(r'[0-9]+', title)[0]

    n_digits = len(number)
    answer_number = str(int(number) + answer_number_dif)
    answer_number = "0" * (n_digits - len(answer_number)) + answer_number

    index = name.find(number)
    answer_name = name[:index] + answer_number + ended + ext

    return answer_name


def results_to_xlsx(answers_path='result_answer.json',
                    numbers_path='numbers.csv',
                    save_path='all/answer_result.xlsx',
                    img_path='all/skew_corrected_answer', n_ans=25, threshold=0.995, n_threshold=0.99998,
                    save_img=True,
                    save_img_path='temp/result'
                    ):
    if save_img and not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

    ans_df = pd.DataFrame()
    prob_df = pd.DataFrame()

    ans_df_keys = {"filename": ""}
    prob_df_keys = {"filename": ""}

    ans_df_keys["MD"] = ""
    ans_df_keys["SBD"] = ""
    prob_df_keys["MD"] = ""
    prob_df_keys["SBD"] = ""
    for i in range(n_ans):
        ans_df_keys["No." + str(i + 1)] = ""
        prob_df_keys["No." + str(i + 1)] = ""
    ans_df_keys["Check_All"] = ""

    ans_df = ans_df.join(pd.DataFrame(ans_df_keys, index=ans_df.index))
    prob_df = prob_df.join(pd.DataFrame(prob_df_keys, index=prob_df.index))

    answer_mapper = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E"
    }
    with open(answers_path, "r") as jsonfile:
        answers_results = json.load(jsonfile)

    with open(numbers_path, "r") as csvfile:
        numbers_results = pd.read_csv(csvfile)

    for result in answers_results:
        basename = os.path.basename(result['filename'])
        filepath = os.path.join(img_path, basename)

        img = cv2.imread(filepath)
        height, width = img.shape[0], img.shape[1]

        bboxes = []
        confs = []

        number_filename = answer_filename_to_number(basename)
        number_result = numbers_results[numbers_results['Filename'] == number_filename].iloc[0]

        for idx, ans in enumerate(result['objects']):
            coords = ans['relative_coordinates']
            x_1 = int((coords['center_x'] - coords['width'] / 2) * width)
            y_1 = int((coords['center_y'] - coords['height'] / 2) * height)

            x_2 = int((coords['center_x'] + coords['width'] / 2) * width)
            y_2 = int((coords['center_y'] + coords['height'] / 2) * height)

            bboxes.append([x_1, y_1, x_2, y_2, ans['class_id']])
            confs.append(ans['confidence'])

        bboxes, confs = nms(bboxes, confs)
        bboxes, confs = sort_lines(bboxes, confs)
        bboxes, confs = remove_overlapped_bbox(bboxes, confs)

        if save_img:
            for bbox in bboxes:
                x_1, y_1, x_2, y_2, class_id = bbox
                cv2.rectangle(img, (x_1, y_1), (x_2, y_2), (0, 0, 255), 5)
                cv2.putText(img, answer_mapper[class_id], (x_1 - 50, int((y_1 + y_2) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(save_img_path, basename), img)

        confs = np.array(confs)
        answer = np.array(bboxes)[:, -1]
        answer = np.vectorize(answer_mapper.get)(answer)

        check_all = False
        n_predicted = answer.shape[0]
        if n_predicted < n_ans:
            answer = np.append(answer, ["E"] * (n_ans - n_predicted))
            confs = np.append(confs, [0.0] * (n_ans - n_predicted))
            check_all = True

        elif n_predicted > n_ans:
            answer = answer[:n_ans]
            confs = confs[:n_ans]
            check_all = True

        if check_all:
            print("Please check: " + basename)

        answer = np.append(answer, str(check_all))
        answer = np.append(np.array([basename, number_result['MD'], number_result['SBD']]), answer)

        confs = np.append(np.array([basename, number_result['MD_conf'], number_result['SBD_conf']]), confs)

        ans_df = ans_df.append(pd.DataFrame(answer.reshape(1, -1), columns=ans_df.columns),
                               ignore_index=True)
        prob_df = prob_df.append(pd.DataFrame(confs.reshape(1, -1), columns=prob_df.columns),
                                 ignore_index=True)

    writer = pd.ExcelWriter(save_path, engine='xlsxwriter', options={'strings_to_numbers': True})
    ans_df.to_excel(writer, sheet_name='Ans')
    prob_df.to_excel(writer, sheet_name='Prob')
    workbook = writer.book
    worksheet = writer.sheets['Ans']

    condition_format = workbook.add_format({'bg_color': '#ffff00'})
    condition_format1 = workbook.add_format({'bg_color': '#eb4034'})

    n_rows = ans_df.shape[0] + 1
    worksheet.conditional_format('E2:AC{n_rows}.'.format(n_rows=n_rows),
                                 {
                                     'type': 'formula',
                                     'criteria': '=Prob!E2:AC{n_rows}<{thshd}'.format(thshd=threshold, n_rows=n_rows),
                                     'format': condition_format
                                 }
                                 )

    worksheet.conditional_format('C2:D{n_rows}.'.format(n_rows=n_rows),
                                 {
                                     'type': 'formula',
                                     'criteria': '=Prob!C2:D{n_rows}<{thshd}'.format(thshd=n_threshold, n_rows=n_rows),
                                     'format': condition_format
                                 }
                                 )

    worksheet.conditional_format('AD2:AD2000', {'type': 'cell',
                                                'criteria': '=',
                                                'value': '"True"',
                                                'format': condition_format1})

    writer.save()


if __name__ == '__main__':
    answer_filename_to_number("SCAN0010.JPG")
