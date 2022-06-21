from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import random
import math
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import xlwt

def loadFile(filePath):
    fileData = pd.read_csv(filePath)
    return fileData

def randomForest():

    DATAS = loadFile("data/new_start_data/new_totall_data1.csv")
    all_features = DATAS.iloc[:, 1:-1]
    all_features_old = all_features
    all_labels = DATAS.loc[:, '腐蚀厚度']

    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    object_feats = all_features.dtypes[all_features.dtypes == 'object'].index

    all_features[numeric_features] = all_features[numeric_features]
    # all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    all_features = pd.get_dummies(all_features, prefix=object_feats, dummy_na=True)
    all_features = all_features.fillna(0)
    all_features = pd.get_dummies(all_features, dummy_na=True)

    # 将训练集划分成8:2（训练集与测试集比例）的比例
    train_features, test_features, train_labels, test_labels = train_test_split(all_features, all_labels, test_size=0.2)

    savePath1 = '/home/wit513/桌面/实验数据/超参数分析1.xlsx'
    ### 测试决策树颗数
    r2_1 = []
    rmse_1 = []
    mae_1 = []
    for n_estimators_value in range(100, 2000, 100):
        rf = RandomForestRegressor(n_estimators=n_estimators_value, max_depth=None)
        rf.fit(train_features, train_labels)  # 训练分类器
        predict_test = rf.predict(test_features)
        roc_auc = metrics.r2_score(test_labels, predict_test)
        rmse_score = np.sqrt(((predict_test - test_labels) ** 2).mean())
        mae_scor = np.sum(np.absolute(predict_test - test_labels)) / len(test_labels)
        r2_1.append(roc_auc)
        rmse_1.append(rmse_score)
        mae_1.append(mae_scor)
    work_book1 = xlwt.Workbook(encoding='utf-8')
    sheet = work_book1.add_sheet("表1")
    sheet.write(0, 0,'r2')
    sheet.write(0, 1,'rmse')
    sheet.write(0, 2, 'mae')
    for i in range(len(r2_1)):
        sheet.write(i + 1, 0, r2_1[i])
        sheet.write(i + 1, 1, rmse_1[i])
        sheet.write(i + 1, 2, mae_1[i])
    work_book1.save(savePath1)

    savePath2 = '/home/wit513/桌面/实验数据/超参数分析2.xlsx'
    ### 测试决策树深度
    r2_2 = []
    rmse_2 = []
    mae_2 = []
    for max_depth_value in range(5, 100, 5):
        rf = RandomForestRegressor(max_depth=max_depth_value)
        rf.fit(train_features, train_labels)  # 训练分类器
        predict_test = rf.predict(test_features)
        roc_auc = metrics.r2_score(test_labels, predict_test)
        rmse_score = np.sqrt(((predict_test - test_labels) ** 2).mean())
        mae_scor = np.sum(np.absolute(predict_test - test_labels)) / len(test_labels)
        r2_2.append(roc_auc)
        rmse_2.append(rmse_score)
        mae_2.append(mae_scor)

    work_book2 = xlwt.Workbook(encoding='utf-8')
    sheet = work_book2.add_sheet("表2")
    sheet.write(0, 0, 'r2')
    sheet.write(0, 1, 'rmse')
    sheet.write(0, 2, 'mae')
    for i in range(len(r2_2)):
        sheet.write(i + 1, 0, r2_2[i])
        sheet.write(i + 1, 1, rmse_2[i])
        sheet.write(i + 1, 2, mae_2[i])
    work_book2.save(savePath2)

    savePath3 = '/home/wit513/桌面/实验数据/超参数分析3.xlsx'
    ### 测试个数和深度的组合
    r2_3 = []
    rmse_3 = []
    mae_3 = []
    n_estimators_value = 100
    for max_depth_value in range(5, 100, 5):
        rf = RandomForestRegressor(n_estimators=n_estimators_value, max_depth=max_depth_value)
        rf.fit(train_features, train_labels)  # 训练分类器
        predict_test = rf.predict(test_features)
        roc_auc = metrics.r2_score(test_labels, predict_test)
        rmse_score = np.sqrt(((predict_test - test_labels) ** 2).mean())
        mae_scor = np.sum(np.absolute(predict_test - test_labels)) / len(test_labels)
        r2_3.append(roc_auc)
        rmse_3.append(rmse_score)
        mae_3.append(mae_scor)
        n_estimators_value = n_estimators_value + 100

    work_book3 = xlwt.Workbook(encoding='utf-8')
    sheet = work_book3.add_sheet("表3")
    sheet.write(0, 0,'r2')
    sheet.write(0, 1,'rmse')
    sheet.write(0, 2, 'mae')
    for i in range(len(r2_3)):
        sheet.write(i + 1, 0, r2_3[i])
        sheet.write(i + 1, 1, rmse_3[i])
        sheet.write(i + 1, 2, mae_3[i])
    work_book3.save(savePath3)


if __name__ == '__main__':
    result = randomForest()
    # print(result)