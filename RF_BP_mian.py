import argparse
import torch
from torch.autograd import Variable
from pylab import *
from Read_Data import ReadData
import os
import torch.nn.functional as F
from sklearn.metrics import r2_score
from Model import Net
import xlwt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置

def parse_arg():
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(asctime)s]: %(levelname)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-dataset', default='carbonSteel')
    parser.add_argument('-batch_size', type=int, default=128)

    parser.add_argument('-feat_dropout', type=float, default=0.3)

    parser.add_argument('-lr', type=float, default=0.001, help="sgd: 10, adam: 0.001")
    parser.add_argument('-gpuid', type=int, default=-1)
    parser.add_argument('-epochs', type=int, default=400)
    parser.add_argument('-report_every', type=int, default=10)
    parser.add_argument('-model_path', default='new_Pretrained', type=str)
    parser.add_argument('-load_epoch', default=18, type=int)  # 18、10
    parser.add_argument('-train', default=False, type=str)

    parser.add_argument('-input_size', type=int, default=184)
    parser.add_argument('-hidden_size', type=int, default=68*2)
    parser.add_argument('-output_size', type=int, default=1)
    parser.add_argument('-transample_size', type=int, default=1000)
    parser.add_argument('-num_layer', type=int, default=68 // 2)

    opt = parser.parse_args()
    return opt


def prepare_db(opt):
    if opt.dataset == 'carbonSteel':
        train_dataset = ReadData(opt, 'data/new_start_data/new_totall_data1.csv', train=True)
        eval_dataset = ReadData(opt, 'data/new_start_data/new_totall_data1.csv', train=False)
        opt.hidden_size = opt.transample_size // (2*(opt.input_size + opt.output_size))
        print(opt.hidden_size)
        return {'train': train_dataset, 'eval': eval_dataset}
    else:
        raise NotImplementedError


def prepare_model(opt):
    model = Net(opt.input_size, opt.hidden_size, opt.output_size)
    if opt.cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    return model


def prepare_optim(model, opt):
    return torch.optim.Adam(model.parameters(), lr=opt.lr)

def train(model, optim, db, opt):
    for epoch in range(1, opt.epochs + 1):
        # Update \Weight
        model.train()
        # 均方损失函数
        criterion = torch.nn.MSELoss()

        train_loader = torch.utils.data.DataLoader(db['train'], batch_size=opt.batch_size, shuffle=True)

        for batch_idx, (data, target) in enumerate(train_loader):
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output1 = model(data)
            output = output1
            loss = criterion(target, output) / torch.var(target) + criterion(output, target) * 100
            optim.zero_grad()
            loss.backward()
            optim.step()

            if batch_idx % opt.report_every == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

        ##模型保存
        model_file = os.path.join(opt.model_path, 'model_{}.pth'.format(epoch))
        torch.save(model.state_dict(), model_file)

        ##模型验证
        model.eval()
        test_loss = 0
        correct = 0
        test_loader = torch.utils.data.DataLoader(db['eval'], batch_size=opt.batch_size, shuffle=True)
        with torch.no_grad():
            for data, target in test_loader:
                if opt.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                output2 = model(data)
                output = output2

                test_loss += criterion(target, output).item()  # sum up batch loss
                pred = output  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()



                x = range(len(target))
                plt.plot(x, target.numpy(), marker='o', mec='r', mfc='w', label=u'目标值')
                plt.plot(x, output.numpy(), marker='*', ms=10, label=u'预测值')
                plt.legend()  # 让图例生效
                plt.xticks(rotation=45)
                plt.margins(0)
                plt.subplots_adjust(bottom=0.15)
                plt.xlabel(u"样本序号")  # X轴标签
                plt.ylabel("腐蚀率")  # Y轴标签
                # plt.title("碳钢腐蚀预测(随机森林+BP神经网络)")  # 标题
                plt.title("碳钢腐蚀预测(BP神经网络)")  # 标题
                # plt.show()

            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f})\n'.format(
                test_loss, correct, len(test_loader.dataset),
                correct / len(test_loader.dataset)))

def test_CNN_RF(model, db, opt):
    i_num = 257
    model.eval()
    indexs=0
    start = 0
    while i_num <= 257:
        # 加载模型
        opt.load_epoch = i_num
        print('loading model parameters from epoch {}...'.format(opt.load_epoch))
        map2device = lambda storage, loc: storage
        test_loader = torch.utils.data.DataLoader(db['eval'], batch_size=opt.batch_size, shuffle=False)
        model_file = os.path.join(opt.model_path, 'model_{}.pth'.format(opt.load_epoch))
        model.load_state_dict(torch.load(model_file, map_location=map2device))

        test_loss = 0
        correct = 0
        targets = []
        datas = []
        with torch.no_grad():
            for data, target in test_loader:
                if opt.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                datas.append(data)
                targets.append(target)

            data = torch.cat(datas, dim=0)
            target = torch.cat(targets, dim=0)

            output1= model(data)
            output = output1

            test_loss += F.mse_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            R2 = r2_score(target, output)
            rmse_score = np.sqrt(((output - target) ** 2).mean())
            mae_scor = torch.sum(np.absolute(output - target),0) / len(output)
            print("神经网络的决定系数R^2=", R2)
            print('神经网络的决定系数的均方根误差RMSE=', rmse_score)
            print("神经网络的决定系数的平均绝对误差MAE=", mae_scor)


            if R2 >=start:
                indexs = i_num
                start = R2

            x = range(len(output))
            savePath_RF_BP = 'RF_BP.xlsx'
            work_book_RF_BP = xlwt.Workbook(encoding='utf-8')
            sheet = work_book_RF_BP.add_sheet("表1")
            sheet.write(0, 0, 'target')
            sheet.write(0, 1, 'output')
            for i in range(len(output)):
                sheet.write(i + 1, 0, str(target[i][0].numpy()))
                sheet.write(i + 1, 1, str(output[i][0].numpy()))
            work_book_RF_BP.save(savePath_RF_BP)
            plt.plot(x, target.numpy(), marker='o', mfc='w', label=u'目标值')
            plt.plot(x, output.numpy(), marker='*', ms=10, mfc='w', label=u'预测值')
            plt.legend()  # 让图例生效
            plt.xticks(range(0,22),rotation=45)
            plt.margins(0)
            plt.subplots_adjust(bottom=0.15)
            plt.xlabel(u"样本序号")  # X轴标签
            plt.ylabel("腐蚀率")  # Y轴标签
            plt.title("腐蚀预测(BP模型)")  # 标题
            plt.savefig('RF_BP.png', dpi=500)  # 指定分辨率 r2\ rmse \ mae
            plt.show()
        i_num += 1

def main():
    opt = parse_arg()
    # GPU
    opt.cuda = opt.gpuid >= 0
    if opt.gpuid >= 0:
        torch.cuda.set_device(opt.gpuid)
    else:
        print("WARNING: RUN WITHOUT GPU")

    db = prepare_db(opt)
    model = prepare_model(opt)
    optim = prepare_optim(model, opt)
    if opt.train:
        train(model, optim, db, opt)
    else:
        test_CNN_RF(model, db, opt)

if __name__ == '__main__':
    main()
