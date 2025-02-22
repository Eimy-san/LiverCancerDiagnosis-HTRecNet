# coding:utf-8
import json
import torch
from model.HTRecNet import *
from structure.datasets import *
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import sklearn.metrics as skm
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from torch import nn
from copy import deepcopy
import pandas as pd
# 参数
PARAMS = Params()

DEVICE = torch.device("cuda:1")  # 使用gpu


def run():
    net = resnet50(num_classes=3).to(DEVICE)
    initial_weights = deepcopy(net.state_dict())
    def predict_lable(test_loader,fold_num):
        y_true_list = []
        y_pred_list = []
        y_scores_list = []
        val_net = torch.load(os.path.join(weight_path, f'weight_fold{fold_num}.pth'))
        val_net.eval()
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                outputs = val_net(X)
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

        output_folder = os.path.join(result_file_path, f'val_predict')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        pre_folder = os.path.join(output_folder, f'fold_{fold_num}_predictions.xlsx')
        # 将汇总DataFrame保存到Excel文件
        results_df.to_excel(pre_folder, index=False)

        print(f'Saved predictions for fold {fold_num} ')

    def evaluate(net, data_iter):
        y_true_list = []
        y_pred_list = []
        running_loss = 0.0
        net.eval()
        for X, y in data_iter:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = net(X)
            y_probs = torch.softmax(outputs, dim=1)
            test_l = loss(outputs, y)
            running_loss += test_l.item()
            y_pred_list.extend(y_probs.argmax(dim=1).cpu().numpy())
            y_true_list.extend(y.cpu().numpy())

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        test_loss = running_loss / len(data_iter)
        test_acc = skm.accuracy_score(y_true, y_pred)
        # 多分类评估指标调整
        test_p, test_r, test_f1, _ = skm.precision_recall_fscore_support(y_true, y_pred, average='macro')
        test_mcc = skm.matthews_corrcoef(y_true, y_pred)
        return test_loss, test_acc, test_p, test_r, test_f1, test_mcc

    def plot_metrics(train_losses, test_losses, train_accs, test_accs, fold_num):
        epochs = range(1, PARAMS.epoch + 1)
        # 绘制训练和测试损失曲线
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, test_losses, label='Testing Loss')
        plt.title('Training and Testing Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        # 绘制训练和测试准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accs, label='Training Accuracy')
        plt.plot(epochs, test_accs, label='Testing Accuracy')
        plt.title('Training and Testing Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(result_file_path, f'fold_{fold_num}_metrics.png'))

    all_train_losses, all_test_losses, all_train_accs, all_test_accs, all_test_p_s, all_test_r_s, all_test_f1_s, all_test_mcc_s = [], [], [], [], [], [], [], []

    # 运行目录
    result_file_path = os.path.join(PARAMS.result_output_dir, f'{PARAMS.model_name}')
    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)
    weight_path = os.path.join(result_file_path, f'weight')
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    index_save_path = os.path.join(result_file_path, f'fold_indices')
    if not os.path.exists(index_save_path):
        os.makedirs(index_save_path)

    for fold_num, (train_index, val_index) in enumerate(kf.split(datasets), start=1):
        # 保存每个fold的索引到JSON文件
        train_indices_file = os.path.join(index_save_path, f'fold_{fold_num}_train_indices.json')
        val_indices_file = os.path.join(index_save_path, f'fold_{fold_num}_val_indices.json')
        with open(train_indices_file, 'w') as f:
            json.dump(train_index.tolist(), f)
        print(f"训练集索引已保存至: {train_indices_file}")
        with open(val_indices_file, 'w') as f:
            json.dump(val_index.tolist(), f)
        print(f"验证集索引已保存至: {val_indices_file}")
        best_acc = 0.0  # 初始化最佳准确率
        best_fold = 0
        best_epoch = 0
        print(f" Fold {fold_num} 开始训练与评估 -------------------------")
        train_fold = torch.utils.data.dataset.Subset(datasets, train_index)  # 使用训练集变换
        val_fold = torch.utils.data.dataset.Subset(datasets, val_index)
        train_data_size = len(train_fold)
        test_data_size = len(val_fold)
        print("训练数据集的长度为：%d" % (train_data_size))
        print("测试数据集的长度为：%d" % (test_data_size))
        # 打包成DataLoader类型 用于 训练
        train_dataloader = DataLoader(dataset=train_fold, batch_size=PARAMS.batch_size, num_workers=PARAMS.num_workers,
                                      shuffle=True, pin_memory=True)
        test_dataloader = DataLoader(dataset=val_fold, batch_size=PARAMS.batch_size, num_workers=PARAMS.num_workers,
                                     shuffle=False, pin_memory=True)
        # net.apply(init_weights)
        # net._initialize_weights()
        net.load_state_dict(initial_weights)
        loss = nn.CrossEntropyLoss().to(DEVICE)
        optim = torch.optim.SGD(params=net.parameters(), lr=PARAMS.lr)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=22, T_mult=2)
        # 初始化表格样式
        table = PrettyTable()
        table.field_names = ["Epoch", "Train Loss", "Train Acc", "Test Loss", "Test Acc", "P", "R", "F1", "MCC"]
        # 创建列表用于保存各epoch的指标
        train_losses, test_losses = [], []
        train_accs, test_accs = [], []
        test_p_s, test_r_s, test_f1_s, test_mcc_s = [], [], [], []

        for epoch in range(PARAMS.epoch):
            net.train()
            running_loss = 0.0
            y_true_list = []
            y_pred_prob_list = []  # 改为存储预测概率
            for X, y in train_dataloader:
                optim.zero_grad()
                X, y = X.to(DEVICE), y.to(DEVICE)
                y_hat = net(X)

                # 添加softmax以得到概率分布
                y_pred_probs = torch.softmax(y_hat, dim=1)

                l = loss(y_hat, y)
                running_loss += l.item()
                l.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 20)  # 梯度裁剪
                optim.step()
                scheduler.step()

                # 存储预测概率和真实标签用于后续评估
                y_pred_prob_list.extend(y_pred_probs.detach().cpu().numpy())
                y_true_list.extend(y.cpu().numpy())

            # 训练结束时，计算训练准确率等指标
            with torch.no_grad():
                y_true = np.array(y_true_list)
                y_pred = np.argmax(np.array(y_pred_prob_list), axis=1)  # 根据概率最大值确定预测类别
                train_loss = running_loss / len(train_dataloader)
                train_acc = skm.accuracy_score(y_true, y_pred)  # 计算训练准确率
            test_loss, test_acc, test_p, test_r, test_f1, test_mcc = evaluate(net, test_dataloader)

            if test_acc > best_acc:
                best_acc = test_acc
                best_model = net  # 保存当前模型为最佳模型
                best_epoch = epoch + 1
                best_fold = fold_num
                # 保存最优模型到指定路径
                best_model_path = os.path.join(weight_path, f'weight_fold{fold_num}.pth')
                torch.save(best_model, best_model_path)
            predict_lable(test_dataloader,fold_num)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            test_p_s.append(test_p)
            test_r_s.append(test_r)
            test_f1_s.append(test_f1)
            test_mcc_s.append(test_mcc)
            # 添加一行到表格数据
            table.add_row(
                [epoch + 1, f"{train_loss:.8f}", f"{train_acc * 100:.2f}%", f"{test_loss:.8f}",
                 f"{test_acc * 100:.2f}%",
                 f"{test_p:.2f}", f"{test_r:.2f}", f"{test_f1:.2f}", f"{test_mcc:.2f}"])

            # 输出每轮结束时的评估结果
            print("\nEpoch Summary:")
            print(table.get_string(start=epoch, end=epoch + 1))
            print("Best Epoch so far:", best_epoch)
            print("\n" + "#" * 80 + "\n")  # 添加分割符

        all_train_losses.append(train_losses)
        all_test_losses.append(test_losses)
        all_train_accs.append(train_accs)
        all_test_accs.append(test_accs)
        all_test_p_s.append(test_p_s)
        all_test_r_s.append(test_r_s)
        all_test_f1_s.append(test_f1_s)
        all_test_mcc_s.append(test_mcc_s)
        # 最后显示整个训练过程的汇总表
        print("\nOverall Training Summary for Fold {}: ".format(fold_num), end="")
        print(table)
        # 将PrettyTable数据转换为pandas DataFrame
        df = pd.DataFrame(table._rows, columns=table.field_names)
        excel_file_path = os.path.join(result_file_path, 'result{}.xlsx'.format(fold_num))
        df.to_excel(excel_file_path, index=False)
        plot_metrics(train_losses, test_losses, train_accs, test_accs, fold_num)
        print(f"\n Fold {fold_num} 结束 -------------------------\n" + "#" * 80 + "\n")
    print(f"最优模型来自 Fold {best_fold}, Epoch {best_epoch}，准确率为：{best_acc * 100:.2f}%")
    epochs = list(range(1, PARAMS.epoch + 1))
    mean_train_losses = [np.mean(epoch_losses) for epoch_losses in list(zip(*all_train_losses))]
    mean_test_losses = [np.mean(epoch_losses) for epoch_losses in list(zip(*all_test_losses))]
    mean_train_accs = [np.mean(epoch_accs) for epoch_accs in list(zip(*all_train_accs))]
    mean_test_accs = [np.mean(epoch_accs) for epoch_accs in list(zip(*all_test_accs))]
    mean_test_precisions = [np.mean(epoch_precisions) for epoch_precisions in list(zip(*all_test_p_s))]
    mean_test_recalls = [np.mean(epoch_recalls) for epoch_recalls in list(zip(*all_test_r_s))]
    mean_test_f1s = [np.mean(epoch_f1s) for epoch_f1s in list(zip(*all_test_f1_s))]
    mean_test_mccs = [np.mean(epoch_mccs) for epoch_mccs in list(zip(*all_test_mcc_s))]
    # 创建汇总的DataFrame
    summary_df = pd.DataFrame({
        'Epoch': epochs,
        'Mean Train Loss': mean_train_losses,
        'Mean Train Accuracy': mean_train_accs,
        'Mean Test Loss': mean_test_losses,
        'Mean Test Accuracy': mean_test_accs,
        'Mean Test Precision': mean_test_precisions,
        'Mean Test Recall': mean_test_recalls,
        'Mean Test F1 Score': mean_test_f1s,
        'Mean Test MCC Score': mean_test_mccs
    })
    print(summary_df)
    # 定义汇总Excel文件的保存路径
    summary_excel_file_path = os.path.join(result_file_path, 'summary_all_epochs.xlsx')
    # 将汇总DataFrame保存到Excel文件
    summary_df.to_excel(summary_excel_file_path, index=False)
    print(f"Summary of all epochs saved to: {summary_excel_file_path}")
    plot_metrics(mean_train_losses, mean_test_losses, mean_train_accs, mean_test_accs, 'avg')
