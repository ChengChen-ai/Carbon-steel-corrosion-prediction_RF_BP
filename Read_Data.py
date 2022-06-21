import torch
from torch.utils.data import Dataset
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
import pickle as pickle
import xlwt
from GA import GA_net

def RF(RF_path, ori_all_features, ori_all_labels):
    global feature_list, rf, target, predictions
    feature_list = list(ori_all_features.columns)
    R2 = 0
    all_features = ori_all_features.iloc[21:,:]
    all_labels = ori_all_labels.iloc[21:]

    index_num = 1
    while(True):
        if R2>=0.9 or index_num >= 10:
            break
        index = random.sample(range(0, len(all_features) // 8), 1)[0]
        train_features, test_features, train_labels, test_labels = train_test_split(all_features, all_labels,
                                                                                    test_size=0.2)
        max_numtree, max_deeptree = GA_net(train_features, test_features, train_labels, test_labels)
        index1 = random.sample(range(30, 50), 1)[0]
        rf = RandomForestRegressor(n_estimators=max_numtree, max_depth=max_deeptree, random_state=index1)
        rf.fit(train_features,train_labels)
        predictions = rf.predict(test_features)
        target = test_labels
        R2 = r2_score(target, predictions)
        index_num += 1
        print("决定系数R^2=", R2)

    ##模型保存
    with open(RF_path, 'wb') as f:
        pickle.dump(rf, f)

    importances = list(rf.feature_importances_)
    all_predictions = rf.predict(ori_all_features)
    # 格式转换
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list,importances)]
    x = range(len(target))
    plt.plot(x, target, marker='o', mec='r', mfc='w', label=u'y=lable')
    plt.plot(x, predictions, marker='*', ms=10, label=u'y=output')
    plt.legend()
    plt.xticks(rotation='60')
    # plt.show()

    return all_predictions, feature_importances


def chooceMainFeature(data_path, train):
    DATAS=pd.read_csv(data_path)
    all_features = DATAS.iloc[:, 1:-1]
    all_labels = DATAS.loc[:, '腐蚀厚度']

    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    object_feats = all_features.dtypes[all_features.dtypes == 'object'].index

    all_features[numeric_features] = all_features[numeric_features]
    all_features = pd.get_dummies(all_features, prefix=object_feats, dummy_na=True)
    all_features = all_features.fillna(0)
    all_features = pd.get_dummies(all_features, dummy_na=True)

    feature_list = list(all_features.columns)

    RF_path = 'RF_pick/model_{}.cpickle'.format('ultimate')
    if os.path.exists(RF_path):
        with open(RF_path, 'rb') as f:
            rf = pickle.load(f)
        train_features = all_features.iloc[21:, :]
        train_labels = all_labels.iloc[21:]
        train_predictions = rf.predict(train_features)
        R2 = r2_score(train_labels, train_predictions)
        all_predictions = rf.predict(all_features)
        importances = list(rf.feature_importances_)
        if R2 < 0.9 and train:
            all_predictions, feature_importances = RF(RF_path, all_features, all_labels)
            return all_features, all_predictions, all_labels, feature_importances
        else:
            if train:
                feature_importances = [(feature, round(importance, 2)) for feature, importance in
                                       zip(feature_list, importances)]
                return all_features, all_predictions, all_labels, feature_importances
            else:
                test_features = all_features[:21]
                test_labels = all_labels[:21]
                predictions = rf.predict(test_features)

                savePath_RF = 'RF_GA.xlsx'
                work_book_RF = xlwt.Workbook(encoding='utf-8')
                sheet = work_book_RF.add_sheet("表1")
                sheet.write(0, 0, 'target')
                sheet.write(0, 1, 'output')
                for i in range(len(predictions)):
                    sheet.write(i + 1, 0, test_labels[i])
                    sheet.write(i + 1, 1, predictions[i])
                work_book_RF.save(savePath_RF)

                x = range(len(test_labels))
                plt.plot(x, test_labels, marker='o', mec='r', mfc='w', label=u'目标值')
                plt.plot(x, predictions, marker='*', ms=10, label=u'预测值')
                plt.legend()  # 让图例生效
                plt.xticks(rotation=45)
                plt.margins(0)
                plt.subplots_adjust(bottom=0.15)
                plt.xlabel(u"样本序号")  # X轴标签
                plt.ylabel("腐蚀率")  # Y轴标签
                plt.title("腐蚀预测(RF模型)")  # 标题
                plt.savefig('RF.png', dpi=500)  # 指定分辨率 r2\ rmse \ mae
                plt.show()

                rmse_score = np.sqrt(((predictions - test_labels) ** 2).mean())
                mae_scor = np.sum(np.absolute(predictions - test_labels)) / len(test_labels)

                print("随机森林的决定系数R^2=", r2_score(test_labels, predictions))
                print('随机森林的均方根误差RMSE=', rmse_score)
                print("随机森林的平均绝对误差MAE=",mae_scor)

                # 格式转换
                feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list,importances)]
                feature_importances1 = sorted(feature_importances,key=lambda x:x[1],reverse=True)
                # 打印
                # [print('Variable:{:20} importance: {}'.format(*pair)) for pair in feature_importances1]

                return all_features, all_predictions, all_labels, feature_importances
    else:
        all_predictions, feature_importances = RF(RF_path, all_features, all_labels)
        return all_features, all_predictions, all_labels, feature_importances

class ReadData(Dataset):
    def __init__(self,opt, root,train=True):
        self.train = train
        super(ReadData, self).__init__()
        self.root = os.path.expanduser(root)
        self.train = train
        self.opt =opt

        self.X, self.y = self.load_data(root, train)

    def __getitem__(self, index):
        return (self.X[index], self.y[index])

    def __len__(self):
        return len(self.X)

    def load_data(self, path, train):
        all_features, all_predictions, all_labels, importances = chooceMainFeature(path, train)

        all_predictions = torch.from_numpy(all_predictions).unsqueeze(dim=1)
        all_labels = torch.from_numpy(np.array(all_labels)).unsqueeze(dim=1)

        new_importances = []
        for data in importances:
            new_importances.append(data[1])
        importances = new_importances
        lable_name = all_features.columns.values
        drop_list = []
        weigh_list = []
        for key, value in enumerate(importances):
            if value == 0:
                drop_list.append(lable_name[key])
                weigh_list.append(value)

        ## RF_BP
        all_features = all_features.drop(drop_list, axis=1)
        # all_features = all_features * np.array(weigh_list)
        all_features = torch.tensor(data=all_features.values)
        all_features = torch.cat([all_features, all_predictions], dim=1)
        all_features = np.array(all_features)
        X_min = all_features.min(axis=0, keepdims=True)
        X_max = all_features.max(axis=0, keepdims=True)
        all_features = (all_features - X_min) / np.clip((X_max - X_min), a_min=1e-6, a_max=None)
        all_features = torch.from_numpy(all_features)

        ## BP
        # all_features = torch.tensor(data=all_features.values)
        # all_features = np.array(all_features)
        # X_min = all_features.min(axis=0, keepdims=True)
        # X_max = all_features.max(axis=0, keepdims=True)
        # all_features = (all_features - X_min) / np.clip((X_max - X_min), a_min=1e-6, a_max=None)
        # all_features = torch.from_numpy(all_features)

        ## no RF_pre
        # all_features = all_features.drop(drop_list, axis=1)
        # # all_features = all_features * np.array(importances)
        # all_features = torch.tensor(data=all_features.values)
        # all_features = np.array(all_features)
        # X_min = all_features.min(axis=0, keepdims=True)
        # X_max = all_features.max(axis=0, keepdims=True)
        # all_features = (all_features - X_min) / np.clip((X_max - X_min), a_min=1e-6, a_max=None)
        # all_features = torch.from_numpy(all_features)


        train_features = all_features[21:,:]
        train_labels = all_labels[21:]
        test_features = all_features[:21,:]
        test_labels = all_labels[:21]
        self.opt.input_size = all_features.shape[-1]
        self.opt.transample_size = train_features.shape[0]
        if train:
            X = train_features
            y = train_labels
            X = X.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor)

        else:
            X = test_features
            y = test_labels
            X = X.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor)

        return X, y