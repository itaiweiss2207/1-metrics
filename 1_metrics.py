import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# constants
PATH = r"C:\Users\itaiw\Downloads\CANCER_TABLE.csv"
BASIC_PRED = "prediction"
DIA = "diameter (cm)"
CANCER = " cancer"
TP = "true_positive"
FP = "false_positive"
TN = "true_negative"
FN = "false_negative"


def read_data_from_path() -> pd.DataFrame:
    """
    Returns dataframe of the given path
    :return: DataFrame
    """

    df = pd.read_csv(PATH)
    print(df.info())
    return df


def add_basic_model_prediction(df: pd.DataFrame, threshold) -> pd.DataFrame:
    """
    The func gets a df and adds a bool column, T for positive prediction of the model (diameter greater than 7 cm), F otherwise
    :param df:
    :return: df with the predictin column
    """
    df[BASIC_PRED] = df[DIA] > threshold
    return df


def add_confusion_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    The func gets a df and adds a bool columns, for TP, TN, FP, FN
    :param df:
    :return: df with the confusion column
    """
    df[TP] = np.where(df[CANCER] & df[BASIC_PRED], 1, 0)
    df[FP] = np.where((~ df[CANCER]) & df[BASIC_PRED], 1, 0)
    df[TN] = np.where((~ df[CANCER]) & (~ df[BASIC_PRED]), 1, 0)
    df[FN] = np.where(df[CANCER] & (~ df[BASIC_PRED]), 1, 0)
    return df


def gen_confusion_mat(df: pd.DataFrame):
    """

    :param df:
    :return:
    """
    # answer for business meaning - TP - people that the models says they'd
    # had cancer, and so it is FN - people that the models says they
    # wouldn't have cancer, but they do etc..
    # TPR, or recall, is the metrics that check the model ability
    # to 'find' the about to be ill people FPR is the measurement of
    # the model to give false-positive, which means to tell a healthy person
    # that he would have cancer.

    tp_count: int = df[TP].sum()
    tn_count: int = df[TN].sum()
    fp_count: int = df[FP].sum()
    fn_count: int = df[FN].sum()
    pos: int = df[CANCER].sum()
    neg: int = df.shape[0] - pos
    confusion_mat = np.array([[tp_count, fp_count], [fn_count, tn_count]])
    # print(confusion_mat)
    tpr = tp_count / pos
    fpr = fp_count / neg
    accuracy = (tp_count + tn_count) / df.shape[0]
    precision = tp_count / (tp_count + fp_count)
    f1_score = 2*(precision * tpr) / (precision + tpr)
    # print(f"recall is: {tpr}, precision is {precision}, and f1_score is: {f1_score}")

    return confusion_mat, tpr, fpr


if __name__ == '__main__':
    data = read_data_from_path()
    best_threshold = 0
    best_fpr = 1
    best_dis = 0

    # y=x points to calc the best threshold
    p1 = np.array([0,0])
    p2 = np.array([1,1])
    norm = np.linalg.norm


    for i in range(1000):
        data1 = add_basic_model_prediction(data, i/100)
        data1 = add_confusion_cols(data1)
        confus_mat, tpr, fpr = gen_confusion_mat(data1)
        p3 = np.array([fpr, tpr])
        distance = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
        plt.plot(fpr, tpr, color='green', linestyle='solid', linewidth=3, marker='o')
        if distance > best_dis:
            best_dis = distance
            best_threshold = i/100
            best_fpr = fpr
    plt.show()
    print(f"best threshold is: {best_threshold} and it's fpr is: {best_fpr}")

