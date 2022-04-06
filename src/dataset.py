import os.path
import numpy as np
import pandas as pd
import cv2
from tensorflow import keras


def preprocess_result_load_dataset(path='2022/result.xlsx'):
    df = pd.read_excel(path, index_col=False)
    df.drop(["file name", "MD", "SBD", "Unnamed: 29"], axis=1, inplace=True)
    df.rename(columns={"file name.1": "filename"}, inplace=True)

    values = {
        "a": 0,
        "b": 1,
        "c": 2,
        "d": 3,
        "e": 4,
        "abcde": -1
    }

    result_df = pd.DataFrame()
    result_df['filename'] = ""
    result_df['value'] = ""

    for key in df.keys():
        if key != "filename":
            ans = df[key].map(values)
            df[key] = ans

    for _, row in df.iterrows():
        filename = row['filename']
        for key in df.keys():
            if key != "filename" and row[key] != -1:
                new_filename = filename + "_" + str(int(key[3:]) - 1) + ".jpg"
                result_df = result_df.append({"filename": new_filename, "value": row[key]}, ignore_index=True)

    e_files = os.listdir("2022/temp/")
    for e in e_files:
        result_df = result_df.append({"filename": e, "value": 4}, ignore_index=True)

    result_df.to_csv("2022/result.csv")


def preprocess_label_dataset(path = 'all/label.csv'):
    df = pd.read_csv(path)
    image_path = df['image']

    final_path = []
    for path in image_path:
        idx = path.find('-')
        final_path.append(path[idx + 1:])

    df = pd.DataFrame({'filename': final_path, 'value': df['choice']})
    df.dropna(inplace=True)

    df.to_csv('all/dataset.csv')


def load_dataset(path="2022/result.csv", img_path="2022/crop_result/"):
    df = pd.read_csv(path)
    df = df.sample(frac=1, axis=1).reset_index(drop=True)
    x = []
    y = df['value']

    for path in df['filename']:
        # print(path)
        #print(img_path + path)
        img = cv2.imread(img_path + path)
        img = cv2.resize(img, (128, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x.append(img)

    x = np.array(x).astype("float32")/255.0
    x = np.expand_dims(x, -1)
    y = keras.utils.to_categorical(y, 5)

    return x, y


if __name__ == '__main__':
    # preprocess_load_dataset()
    preprocess_label_dataset("../all/project-4-at-2022-03-16-15-15-4987bd97.csv")
    #load_dataset("all/dataset.csv", "all/crop_result/")