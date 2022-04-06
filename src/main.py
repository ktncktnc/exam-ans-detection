from src.model import *
import pandas as pd
import cv2
import os
from tqdm import tqdm

answer_mapper = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E"
}


def main():
    # DF
    ans_df = pd.DataFrame()
    prob_df = pd.DataFrame()

    ans_df_keys = {"filename": ""}
    prob_df_keys = {"filename": ""}
    for i in range(25):
        ans_df_keys["No." + str(i + 1)] = ""
        prob_df_keys["No." + str(i + 1)] = ""

    ans_df = ans_df.join(pd.DataFrame(ans_df_keys, index=ans_df.index))
    prob_df = prob_df.join(pd.DataFrame(prob_df_keys, index=prob_df.index))

    # Model
    model = load_nnet_model("../all/trained_model/cp.ckpt")

    ### Data
    crop_path = "../all/crop_result/"
    file_paths = "../all/answer/"
    ans = os.listdir(file_paths)
    ans.sort()

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

    for f in tqdm(ans):
        base_name = os.path.basename(f)[:-5]
        if base_name in error_files:
            continue

        x = []
        for i in range(25):
            img = cv2.imread(crop_path + base_name + f"_{i}.jpg")
            img = cv2.resize(img, (128, 32))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x.append(img)

        x = np.array(x).astype("float32") / 255.0
        x = np.expand_dims(x, -1)

        y = model.predict(x, batch_size=16)
        prob_y = list(np.max(y, axis=1))
        y = np.argmax(y, axis=1)

        y = list(map(answer_mapper.get, y))
        y.insert(0, base_name)
        prob_y.insert(0, base_name)

        ans_df = ans_df.append(pd.DataFrame([y], columns=ans_df.columns), ignore_index=True)
        prob_df = prob_df.append(pd.DataFrame([prob_y], columns=prob_df.columns), ignore_index=True)

    writer = pd.ExcelWriter('../all/model_answer_result.xlsx', engine='xlsxwriter')
    ans_df.to_excel(writer, sheet_name='Ans')
    prob_df.to_excel(writer, sheet_name='Prob')
    workbook = writer.book
    worksheet = writer.sheets['Ans']

    condition_format = workbook.add_format({'bg_color': '#ffff00'})
    worksheet.conditional_format('B2:Z514', {'type': 'formula',
                                           'criteria': '=Prob!B2<0.99998',
                                           'format': condition_format})

    writer.save()

if __name__ == '__main__':
    main()