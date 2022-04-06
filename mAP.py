from sklearn.metrics import average_precision_score
import pandas as pd
import numpy as np
from tensorflow import keras

def main():
    mapper = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4
    }
    error_files = [
        "20220313173503863_0042.jpeg", "20220313173612205_0012.jpeg", "20220313173755479_0006.jpeg", "20220313173755479_0008.jpeg",
        "20220313173755479_0036.jpeg", "20220313173850433_0012.jpeg", "20220313173919210_0028.jpeg", "20220313173945957_0030.jpeg",
        "20220313174009616_0014.jpeg", "20220313174731646_0030.jpeg", "20220313174807707_0032.jpeg", "20220313174807707_0034.jpeg",
        "20220313174832842_0014.jpeg", "20220313174910489_0052.jpeg", "20220313174947850_0042.jpeg", "20220313175010879_0006.jpeg",
        "20220313175010879_0012.jpeg", "20220313175010879_0030.jpeg", "20220313175050718_0006.jpeg", "20220313175050718_0032.jpeg",
        "20220313175050718_0052.jpeg", "20220313175128214_0022.jpeg", "20220313175128214_0026.jpeg", "20220313175219011_0028.jpeg",
        "20220313175219011_0030.jpeg", "20220313175347348_0010.jpeg", "20220313175418427_0022.jpeg", "20220313175445287_0026.jpeg",
        "de_01_0032.jpeg", "de_04_0038.jpeg", "de_04_0064.jpeg"
    ]

    pd_true = pd.read_excel('all/model_answer_result.xlsx').reset_index()
    pd_pred = pd.read_excel('result.xlsx').reset_index()

    pd_true.drop(['index', 'Unnamed: 0'], axis = 1, inplace=True)
    pd_pred.drop(['index', 'Unnamed: 0', 'Check_All'], axis = 1, inplace=True)
    pd_pred = pd_pred[~pd_pred['filename'].isin(error_files)]

    pd_true.sort_values('filename', inplace=True)
    pd_pred.sort_values('filename', inplace=True)

    pd_true.drop(['filename'], axis=1, inplace=True)
    pd_pred.drop(['filename'], axis=1, inplace=True)
    y_true = pd_true.to_numpy().reshape((-1))
    y_pred = pd_pred.to_numpy().reshape((-1))

    y_true = np.vectorize(mapper.get)(y_true)
    y_pred = np.vectorize(mapper.get)(y_pred)
    y_true = keras.utils.to_categorical(y_true, 5)
    y_pred = keras.utils.to_categorical(y_pred, 5)
    print(y_true[10000])
    print(y_pred[10000])
    print(average_precision_score(y_true, y_pred))


if __name__ == "__main__":
    main()