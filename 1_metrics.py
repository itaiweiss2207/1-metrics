import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sb

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


def add_basic_model_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    The func gets a df and adds a bool column, T for positive prediction of the model (diameter greater than 7 cm), F otherwise
    :param df:
    :return: df with the predictin column
    """
    df[BASIC_PRED] = df[DIA] > 7
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
    neg: int = df.shape[1] - pos
    confusion_mat = np.array([[tp_count, fp_count], [fn_count, tn_count]])
    print(confusion_mat)
    tpr = tp_count / pos
    fpr = fp_count / neg
    accuracy = (tp_count + tn_count) / df.shape[0]
    precision = tp_count / (tp_count + fp_count)
    f1_score = 2*(precision * tpr) / (precision + tpr)
    print(f"recall is: {tpr}, precision is {precision}, and f1_score is: {f1_score}")

    return confusion_mat


if __name__ == '__main__':
    data = read_data_from_path()
    data = add_basic_model_prediction(data)
    data = add_confusion_cols(data)
    confus_mat = gen_confusion_mat(data)

