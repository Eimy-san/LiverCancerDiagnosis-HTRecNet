import itertools

import pandas as pd

from structure.datasets import *
from params import *
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

PARAMS = Params()
DEVICE = torch.device("cuda:0")

testdata = r"datasets/val"

def eval():
    y_true_list = []  # 存储真实的标签
    y_scores_list = []  # 存储模型预测的概率或得分
    y_pred_list = []  # 存储模型预测的标签
    # weight_files = ['weight_fold1.pth', 'weight_fold2.pth', 'weight_fold3.pth', 'weight_fold4.pth', 'weight_fold5.pth']
    # for fold_weights in weight_files:
    net = torch.load(f'{PARAMS.result_output_dir}/{PARAMS.model_name}/weight/weight_fold5.pth').to(DEVICE)
    data_test = PaddyDataSet(testdata, val_transform)
    test_loader = DataLoader(data_test, batch_size=PARAMS.batch_size, shuffle=False, num_workers=PARAMS.num_workers)
    net.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = net(X)
            y_probs = torch.softmax(outputs, dim=1)
            y_scores_list.extend(y_probs.cpu().numpy())
            y_pred_list.extend(y_probs.argmax(dim=1).cpu().numpy())
            y_true_list.extend(y.cpu().numpy())

    y_scores_list = np.array(y_scores_list)
    results_df = pd.DataFrame({
        'Score_Class_0': y_scores_list[:, 0],
        'Score_Class_1': y_scores_list[:, 1],
        'Score_Class_2': y_scores_list[:, 2],
        'True': y_true_list,
        'Pred': y_pred_list
    })
    results_df.to_excel('test.xlsx', index=False)

    print(f'Saved')


if __name__ == '__main__':
    eval()