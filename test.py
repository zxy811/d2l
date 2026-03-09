# import os
# import pandas as pd
# import torch
# base_dir = os.path.dirname(os.path.abspath(__file__))
# data_dir = os.path.join(base_dir, 'data')
# os.makedirs(data_dir, exist_ok=True)
# data_file = os.path.join(data_dir, 'house_tiny.csv')
# with open(data_file, 'w') as f:
#     f.write('NumRooms,Alley,Price\n')  # 列名
#     f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')
# data = pd.read_csv(data_file)
# print(data)
# # 删除缺失值最多的列（并列最多则都删）
# missing = data.isna().sum()
# max_missing = missing.max()
# cols_to_drop = missing[missing == max_missing].index.tolist()
# print(cols_to_drop)
# data = data.drop(columns=cols_to_drop)
# print("删除缺失最多的列:", cols_to_drop)
# # 删除后：前几列为输入，最后一列为目标
# inputs, outputs = data.iloc[:, :-1], data.iloc[:, -1]
# # 只对数值列用均值填充，避免对字符串列（如 Alley）求 mean 报错
# inputs = inputs.fillna(inputs.mean(numeric_only=True))
# inputs = pd.get_dummies(inputs, dummy_na=True)
# print(inputs)
# print(outputs)
# x=torch.tensor(inputs.to_numpy(dtype=float))
# y=torch.tensor(outputs.to_numpy(dtype=float))
# print(x)
# print(y)

# import torch
# a = torch.arange(9).reshape(3,3)
# b = torch.arange(9).reshape(3,3)
# print(a)
# print(b)
# print(torch.dot(a,b))
# # print(torch.mv(a,b))
# print(torch.mm(a,b))

# import torch
# import d2l
# import matplotlib.pyplot as plt
# from matplotlib import backend_inline
# import numpy as np

# def use_svg_display():
#     """使用svg格式在Jupyter中显示绘图"""
#     backend_inline.set_matplotlib_formats('svg')

# def set_figsize(figsize=(3.5, 2.5)):  #@save
#     """设置matplotlib的图表大小"""
#     use_svg_display()
#     d2l.plt.rcParams['figure.figsize'] = figsize

# def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
#     """设置matplotlib的轴"""
#     axes.set_xlabel(xlabel)
#     axes.set_ylabel(ylabel)
#     axes.set_xscale(xscale)
#     axes.set_yscale(yscale)
#     axes.set_xlim(xlim)
#     axes.set_ylim(ylim)
#     if legend:
#         axes.legend(legend)
#     axes.grid()

# def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
#     """绘制数据点"""
#     if legend is None:
#         legend = []
#     set_figsize(figsize)
#     axes = axes if axes else d2l.plt.gca()
#     def has_one_axis(X):
#         return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__"))
#     if has_one_axis(X):
#         X = [X]
#     if Y is None:
#         X, Y = [[]] * len(X), X
#     elif has_one_axis(Y):
#         Y = [Y]
#     if len(X) != len(Y):
#         X = X * len(Y)
#     axes.cla()
#     for x, y, fmt in zip(X, Y, fmts):
#         if len(x):
#             axes.plot(x, y, fmt)
#         else:
#             axes.plot(y, fmt)
#     set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
#     #@save
# def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
#          ylim=None, xscale='linear', yscale='linear',
#          fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
#     """绘制数据点"""
#     if legend is None:
#         legend = []

#     set_figsize(figsize)
#     axes = axes if axes else d2l.plt.gca()

#     # 如果X有一个轴，输出True
#     def has_one_axis(X):
#         return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
#                 and not hasattr(X[0], "__len__"))

#     if has_one_axis(X):
#         X = [X]
#     if Y is None:
#         X, Y = [[]] * len(X), X
#     elif has_one_axis(Y):
#         Y = [Y]
#     if len(X) != len(Y):
#         X = X * len(Y)
#     axes.cla()
#     for x, y, fmt in zip(X, Y, fmts):
#         if len(x):
#             axes.plot(x, y, fmt)
#         else:
#             axes.plot(y, fmt)
#     set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
# x = np.arange(0, 3, 0.1)
# plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])

# import torch
# x=torch.arange(4.0)
# print(x)
# x.requires_grad_(True)
# x.grad
# print(x.grad)
# y= torch.dot(x,x)
# print(y)
# y.backward()
# print(y)
# print(x.grad)
# x.grad.zero_()
# y=x.sum()
# print(y)
# y.backward()
# x.grad
# print(x.grad)


# import torch
# from torch.distributions import multinomial
# from d2l import torch as d2l

# fair_probs = torch.ones([6]) / 6
# print(fair_probs)
# counts = multinomial.Multinomial(10, fair_probs).sample((500,))
# cum_counts = counts.cumsum(dim=0)
# estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
# print(estimates)
# d2l.set_figsize((6, 4.5))
# for i in range(6):
#     d2l.plt.plot(estimates[:, i].numpy(),
#                  label=("P(die=" + str(i + 1) + ")"))
# d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
# d2l.plt.gca().set_xlabel('Groups of experiments')
# d2l.plt.gca().set_ylabel('Estimated probability')


# import math
# import numpy as np
# import torch
# import time
# from d2l import torch as d2l
# n=10000
# a=torch.ones([n])
# b=torch.ones([n])
# class Timer:
#     def __init__(self):
#         self.times = []
#         self.start()
#     def start(self):
#         self.start_time = time.time()
#     def stop(self):
#         self.times.append(time.time() - self.start_time)
#         return self.times[-1]
#     def avg(self):
#         return sum(self.times) / len(self.times)
# c=torch.zeros(n)
# timer=Timer()
# for i in range(10):
#     c+=a*b
# print(f'{timer.stop():.5f} sec')
# timer.start()
# for i in range(10):
#     c+=a*b
# print(f'{timer.stop():.5f} sec')




# import torch
# import random
# from d2l import torch as d2l

# def synthetic_data(w, b, num_examples):
#     """生成y=Xw+b+噪声"""
#     X = torch.normal(0, 1, (num_examples, len(w)))
#     y = torch.matmul(X, w) + b
#     y += torch.normal(0, 0.01, y.shape)
#     # print(X)
#     # print(y)
#     # y=y.reshape((-1, 1))
#     # print(y)
#     return X, y.reshape((-1, 1))

# true_w = torch.tensor([2, -3.4])
# true_b = 4.2

# features, labels = synthetic_data(true_w, true_b, 1000)

# # d2l.set_figsize()
# # d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1);
# # d2l.plt.show()

# def data_iter(batch_size, features, labels):
#     num_examples = len(features)
#     indices = list(range(num_examples))
#     random.shuffle(indices)
#     for i in range(0, num_examples, batch_size):
#         batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
#         yield features[batch_indices], labels[batch_indices]
# batch_size = 10

# for X, y in data_iter(batch_size, features, labels):
#     print(f'X shape: {X.shape}, X={X}, y shape: {y.shape}, y={y}')
#     break
# w=torch.normal(0, 0.01, size=(2,1), requires_grad=True)
# b=torch.zeros(1, requires_grad=True)

# def linreg(X, w, b):
#     return torch.matmul(X, w) + b

# def squared_loss(y_hat, y):
#     return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# def sgd(params, lr, batch_size):
#     with torch.no_grad():
#         for param in params:
#             param -= lr * param.grad / batch_size
#             param.grad.zero_()
# lr = 0.03
# num_epochs = 3
# net = linreg
# loss = squared_loss
# for epoch in range(num_epochs):
#     for X, y in data_iter(batch_size, features, labels):
#         l = loss(net(X, w, b), y)
#         l.sum().backward()
#         sgd([w, b], lr, batch_size)
#     with torch.no_grad():
#         train_l = loss(net(features, w, b), labels)
#         print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
# print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
# print(f'b的估计误差: {true_b - b}')

# import torch
# import numpy as np
# from torch.utils import data
# from d2l import torch as d2l
# from torch import nn

# true_w = torch.tensor([2, -3.4])
# true_b = 4.2
# features, labels = d2l.synthetic_data(true_w, true_b, 1000)
# def load_array(data_arrays, batch_size, is_train=True):  #@save
#     """构造一个PyTorch数据迭代器"""
#     dataset = data.TensorDataset(*data_arrays)
#     return data.DataLoader(dataset, batch_size, shuffle=is_train)
# batch_size = 10
# data_iter = load_array((features, labels), batch_size)
# # for X, y in data_iter:
# #     print(f'X shape: {X.shape}, X={X}, y shape: {y.shape}, y={y}')
# #     break
# next(iter(data_iter))
# net = nn.Sequential(nn.Linear(2, 1))
# net[0].weight.data.normal_(0, 0.01)
# net[0].bias.data.fill_(0)
# loss=nn.MSELoss()
# trainer=torch.optim.SGD(net.parameters(), lr=0.03)
# num_epochs = 3
# for epoch in range(num_epochs):
#     for X, y in data_iter:
#         l = loss(net(X) ,y)
#         trainer.zero_grad()
#         l.backward()
#         # print(net[0].weight.grad)
#         # print(net[0].bias.grad)
#         trainer.step()
#     l = loss(net(features), labels)
#     print(f'epoch {epoch + 1}, loss {float(l.mean()):f}')
# w=net[0].weight.data
# b=net[0].bias.data
# print(f'w的估计误差: {true_w - net[0].weight.data.reshape(true_w.shape)}')
# print(f'b的估计误差: {true_b - net[0].bias.data[0]}')


        
            # import torch
            # import torchvision
            # from torch.utils import data
            # from torchvision import transforms
            # from d2l import torch as d2l

            # d2l.use_svg_display()
            # trans = transforms.ToTensor()
            # mnist_train = torchvision.datasets.FashionMNIST(
            #     root="../data", train=True, transform=trans, download=True)
            # mnist_test = torchvision.datasets.FashionMNIST(
            #     root="../data", train=False, transform=trans, download=True)
            # print(len(mnist_train), len(mnist_test))
            # print(mnist_train[0][0].shape)
            # def get_fashion_mnist_labels(labels):  #@save
            #     """返回Fashion-MNIST数据集的文本标签"""
            #     text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
            #                 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
            #     return [text_labels[int(i)] for i in labels]
            # def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
            #     """绘制图像列表"""
            #     figsize = (num_cols * scale, num_rows * scale)
            #     _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
            #     axes = axes.flatten()
            #     for i, (ax, img) in enumerate(zip(axes, imgs)):
            #         if torch.is_tensor(img):
            #             ax.imshow(img.numpy())
            #         else:
            #             ax.imshow(img)
            #         ax.axes.get_xaxis().set_visible(False)
            #         ax.axes.get_yaxis().set_visible(False)
            #         if titles:
            #             ax.set_title(titles[i])
            #     return axes
            # X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
            # show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
            # batch_size = 256
            # def get_dataloader_workers():  #@save
            #     """使用4个进程来读取数据"""
            #     return 4
            # train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())
            # timer = d2l.Timer()
            # for X, y in train_iter:
            #     continue
            # print(f'{timer.stop():.2f} sec')
            # def load_data_fashion_mnist(batch_size, resize=None):  #@save
            #     """下载Fashion-MNIST数据集，然后将其加载到内存中"""
            #     trans = [transforms.ToTensor()]
            #     if resize:
            #         trans.insert(0, transforms.Resize(resize))
            #     trans = transforms.Compose(trans)
            #     mnist_train = torchvision.datasets.FashionMNIST(
            #         root="../data", train=True, transform=trans, download=True)
            #     mnist_test = torchvision.datasets.FashionMNIST(
            #         root="../data", train=False, transform=trans, download=True)
            #     return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()), data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))
            # train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
            # for X, y in train_iter:
            #     print(X.shape, X.dtype, y.shape, y.dtype)
            #     break



# import torch
# from IPython import display
# from d2l import torch as d2l

# batch_size = 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# num_inputs = 784
# num_outputs = 10
# w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
# b = torch.zeros(num_outputs, requires_grad=True)
# def softmax(X):
#     X_exp = torch.exp(X)
#     partition = X_exp.sum(1, keepdim=True)
#     return X_exp / partition
# def net(X):
#     return softmax(torch.matmul(X.reshape(-1, w.shape[0]), w) + b)
# def cross_entropy(y_hat, y):
#     return -torch.log(y_hat[range(len(y_hat)), y])

# # y=torch.tensor([0, 2])
# # y_hat=torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# # a=torch.tensor(y_hat[[0,1], y])
# # print(y_hat)
# # print(a)
# # b=cross_entropy(y_hat, y)
# # print(b)
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
        self.fig, self.axes = d2l.plt.subplots(1, 1, figsize=figsize)
        if legend is None:
            legend = []
        # d2l.use_svg_display()
        self.config_axes = lambda: d2l.set_axes(
            self.axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes.cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes.plot(x, y, fmt)
        self.config_axes()
        # d2l.plt.show()  # 或 self.fig.show()


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
# print(accuracy(y_hat, y)/len(y))

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:

            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# def train_epoch_ch3(net, train_iter, loss, updater):
#     if isinstance(net, torch.nn.Module):
#         net.train()
#     metric = Accumulator(3)
#     for X, y in train_iter:
#         y_hat = net(X)
#         # print(y_hat)
#         l = loss(y_hat, y)
#         # print(l)
#         if isinstance(updater, torch.optim.Optimizer):
#             updater.zero_grad()
#             l.mean().backward()
#             updater.step()
#         else:
#             l.sum().backward()
#             updater(X.shape[0])
#         metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
#         # print(metric)
#     return metric[0] / metric[2], metric[1] / metric[2]

# def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
#     # for X, y in train_iter:
#     #     print(X.shape, X.dtype, y.shape, y.dtype)
#     #     print(X)
#     #     print(y)
#     #     break
#     animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
#                         legend=['train loss', 'train acc', 'test acc'])
#     for epoch in range(num_epochs):
#         train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
#         test_acc = evaluate_accuracy(net, test_iter)
#         print(train_metrics)
#         print(test_acc)
#         animator.add(epoch + 1, train_metrics + (test_acc,))
#     train_loss, train_acc = train_metrics
#     assert train_loss < 0.5, train_loss
#     assert train_acc <= 1 and train_acc > 0.7, train_acc
#     assert test_acc <= 1 and test_acc > 0.7, test_acc

# # # # def predict_ch3(net, test_iter, n=6):
# # # #     for X, y in test_iter:
# # # #         break
# # # #     trues = d2l.get_fashion_mnist_labels(y)
# # # #     preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
# # # #     titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
# # # #     d2l.show_images(
# # # #         X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
# # # # # predict_ch3(net, test_iter)
# # # # def updater(batch_size):
# # # #     return d2l.sgd([w, b], lr, batch_size)
# # # # # print(evaluate_accuracy(net, test_iter))
# # # # def main():
# # # #     # print(accuracy(y_hat

# # # #     train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
# # # # if __name__ == "__main__":
# # # #     lr=0.1
# # # #     num_epochs=10
# # # #     main()
# # # #     d2l.plt.show()

# # # import torch
# # # from torch import nn
# # # from d2l import torch as d2l
# # # batch_size = 256
# # # train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# # # net=nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
# # # def init_weights(m):
# # #     if type(m) == nn.Linear:
# # #         nn.init.normal_(m.weight, std=0.01)
# # # net.apply(init_weights)
# # # loss=nn.CrossEntropyLoss(reduction='none')
# # # trainer=torch.optim.SGD(net.parameters(), lr=0.1)
# # # num_epochs=10
# def train_epoch_ch3(net, train_iter, loss, updater):
#     if isinstance(net, torch.nn.Module):
#         net.train()
#     metric = Accumulator(3)
#     for X, y in train_iter:
#         y_hat = net(X)
#         # print(y_hat)
#         l = loss(y_hat, y)
#         # print(l)
#         if isinstance(updater, torch.optim.Optimizer):
#             updater.zero_grad()
#             l.mean().backward()
#             updater.step()
#         else:
#             l.sum().backward()
#             updater(X.shape[0])
#         metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
#         # print(metric)
#     return metric[0] / metric[2], metric[1] / metric[2]

# def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
#     # for X, y in train_iter:
#     #     print(X.shape, X.dtype, y.shape, y.dtype)
#     #     print(X)
#     #     print(y)
#     #     break
#     animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
#                         legend=['train loss', 'train acc', 'test acc'])
#     for epoch in range(num_epochs):
#         train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
#         test_acc = evaluate_accuracy(net, test_iter)
#         print(train_metrics)
#         print(test_acc)
#         animator.add(epoch + 1, train_metrics + (test_acc,))
#     train_loss, train_acc = train_metrics
#     assert train_loss < 0.5, train_loss
#     assert train_acc <= 1 and train_acc > 0.7, train_acc
#     assert test_acc <= 1 and test_acc > 0.7, test_acc
# # # def main():
# # #     train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

# # # if __name__ == "__main__":
# # #     main()
# # #     d2l.plt.show()

# # # import torch
# # # from d2l import torch as d2l
# # # x=torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
# # # y=torch.relu(x)
# # # d2l.plot(x.detach().numpy(), y.detach().numpy(), 'x', 'relu(x)', figsize=(5, 2.5))
# # # d2l.plt.show()
# # # y.backward(torch.ones_like(x), retain_graph=True)
# # # d2l.plot(x.detach().numpy(), x.grad.detach().numpy(), 'x', 'grad of relu(x)', figsize=(5, 2.5))
# # # d2l.plt.show()

# # # import torch
# # # from d2l import torch as d2l
# # # x=torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
# # # y=torch.tanh(x)
# # # d2l.plot(x.detach().numpy(), y.detach().numpy(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
# # # d2l.plt.show()
# # # y.backward(torch.ones_like(x), retain_graph=True)
# # # d2l.plot(x.detach().numpy(), x.grad.detach().numpy(), 'x', 'grad of sigmoid(x)', figsize=(5, 2.5))
# # # d2l.plt.show()

# # import torch
# # from torch import nn
# # from d2l import torch as d2l
# # batch_size = 256
# # train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# # num_inputs, num_outputs, num_hiddens = 784, 10, 256
# # W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
# # b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
# # W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
# # b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
# # params = [W1, b1, W2, b2]
# # def relu(X):
# #     a = torch.zeros_like(X)
# #     return torch.max(X, a)
# # def net(X):
# #     X = X.reshape((-1, W1.shape[0]))
# #     H = relu(X @ W1 + b1)
# #     return H @ W2 + b2
# # loss=nn.CrossEntropyLoss(reduction='none')
# # trainer=torch.optim.SGD(params, lr=0.1)
# # num_epochs=10
# # def main():
# #     train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# #     d2l.plt.show()
# # if __name__ == "__main__":
# #     main()


# import torch
# from torch import nn
# from d2l import torch as d2l
# net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256),nn.ReLU(),nn.Linear(256, 10))
# def init_weights(m):
#     if type(m) == nn.Linear:
#         nn.init.normal_(m.weight, std=0.01)
# net.apply(init_weights)
# batch_size,lr,num_epochs=256,0.1,10
# loss=nn.CrossEntropyLoss(reduction='none')
# trainer=torch.optim.SGD(net.parameters(), lr=lr)
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# def main():
#     train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
#     d2l.plt.show()
# if __name__ == "__main__":
#     main()

import numpy as np
import math
from torch import nn
import torch
from d2l import torch as d2l
max_degree=20
n_train, n_test = 100, 100
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])
features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)
# print(features[:2])
# print(poly_features[:2])
# print(labels[:2])
true_w,features,poly_features,labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]
# print(features[:2])
# print(poly_features[:2])
# print(labels[:2])
def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    # print(train_features.shape, test_features.shape, train_labels.shape, test_labels.shape)
    # print(train_features[:2])
    # print(test_features[:2])
    # print(train_labels[:2])
    # print(test_labels[:2])
    # input_shape = train_features.shape[-1]
    # print("--------------------------------")
    # print(input_shape)
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    # print(train_labels)
    # print(train_labels.reshape(-1,1))
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())

def main():
    train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])
    # d2l.plt.show()
if __name__ == "__main__":
    main()