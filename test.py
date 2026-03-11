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
# class Accumulator:
#     def __init__(self, n):
#         self.data = [0.0] * n
#     def add(self, *args):
#         self.data = [a + float(b) for a, b in zip(self.data, args)]
#     def reset(self):
#         self.data = [0.0] * len(self.data)
#     def __getitem__(self, idx):
#         return self.data[idx]

# class Animator:
#     def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
#         self.fig, self.axes = d2l.plt.subplots(1, 1, figsize=figsize)
#         if legend is None:
#             legend = []
#         # d2l.use_svg_display()
#         self.config_axes = lambda: d2l.set_axes(
#             self.axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
#         self.X, self.Y, self.fmts = None, None, fmts
#     def add(self, x, y):
#         if not hasattr(y, "__len__"):
#             y = [y]
#         n = len(y)
#         if not hasattr(x, "__len__"):
#             x = [x] * n
#         if not self.X:
#             self.X = [[] for _ in range(n)]
#         if not self.Y:
#             self.Y = [[] for _ in range(n)]
#         for i, (a, b) in enumerate(zip(x, y)):
#             if a is not None and b is not None:
#                 self.X[i].append(a)
#                 self.Y[i].append(b)
#         self.axes.cla()
#         for x, y, fmt in zip(self.X, self.Y, self.fmts):
#             self.axes.plot(x, y, fmt)
#         self.config_axes()
#         # d2l.plt.show()  # 或 self.fig.show()


# def accuracy(y_hat, y):
#     """计算预测正确的数量"""
#     if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
#         y_hat = y_hat.argmax(dim=1)
#     cmp = y_hat.type(y.dtype) == y
#     return float(cmp.type(y.dtype).sum())
# # print(accuracy(y_hat, y)/len(y))

# def evaluate_accuracy(net, data_iter):
#     if isinstance(net, torch.nn.Module):
#         net.eval()
#     metric = Accumulator(2)
#     with torch.no_grad():
#         for X, y in data_iter:

#             metric.add(accuracy(net(X), y), y.numel())
#     return metric[0] / metric[1]


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
# # # # num_epochs=10
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
# # # # def main():
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

# import numpy as np
# import math
# from torch import nn
# import torch
# from d2l import torch as d2l
# max_degree=20
# n_train, n_test = 100, 100
# true_w = np.zeros(max_degree)
# true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])
# features = np.random.normal(size=(n_train + n_test, 1))
# np.random.shuffle(features)
# poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
# for i in range(max_degree):
#     poly_features[:, i] /= math.gamma(i + 1)
# labels = np.dot(poly_features, true_w)
# labels += np.random.normal(scale=0.1, size=labels.shape)
# # print(features[:2])
# # print(poly_features[:2])
# # print(labels[:2])
# true_w,features,poly_features,labels = [torch.tensor(x, dtype=
#     torch.float32) for x in [true_w, features, poly_features, labels]]
# # print(features[:2])
# # print(poly_features[:2])
# # print(labels[:2])
# def evaluate_loss(net, data_iter, loss):
#     """评估给定数据集上模型的损失"""
#     metric = d2l.Accumulator(2)
#     for X, y in data_iter:
#         out = net(X)
#         y = y.reshape(out.shape)
#         l = loss(out, y)
#         metric.add(l.sum(), l.numel())
#     return metric[0] / metric[1]
# def train(train_features, test_features, train_labels, test_labels,
#           num_epochs=400):
#     loss = nn.MSELoss(reduction='none')
#     # print(train_features.shape, test_features.shape, train_labels.shape, test_labels.shape)
#     # print(train_features[:2])
#     # print(test_features[:2])
#     # print(train_labels[:2])
#     # print(test_labels[:2])
#     # input_shape = train_features.shape[-1]
#     # print("--------------------------------")
#     # print(input_shape)
#     # 不设置偏置，因为我们已经在多项式中实现了它
#     net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
#     batch_size = min(10, train_labels.shape[0])
#     train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
#                                 batch_size)
#     # print(train_labels)
#     # print(train_labels.reshape(-1,1))
#     test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
#                                batch_size, is_train=False)
#     trainer = torch.optim.SGD(net.parameters(), lr=0.01)
#     animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
#                             xlim=[1, num_epochs], ylim=[1e-3, 1e2],
#                             legend=['train', 'test'])
#     for epoch in range(num_epochs):
#         train_epoch_ch3(net, train_iter, loss, trainer)
#         if epoch == 0 or (epoch + 1) % 20 == 0:
#             animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
#                                      evaluate_loss(net, test_iter, loss)))
#     print('weight:', net[0].weight.data.numpy())

# def main():
#     train(poly_features[:n_train, :4], poly_features[n_train:, :4],
#       labels[:n_train], labels[n_train:])
#     # d2l.plt.show()
# if __name__ == "__main__":
#     main()

# import torch
# from torch import nn
# from d2l import torch as d2l
# n_train, n_test, num_inputs, batch_size,num_epochs = 20, 100, 200, 5, 100 
# true_w = torch.ones((num_inputs, 1)) * 0.01
# true_b = 0.05
# train_data = d2l.synthetic_data(true_w, true_b, n_train)
# train_iter = d2l.load_array(train_data, batch_size)
# test_data = d2l.synthetic_data(true_w, true_b, n_test)
# test_iter = d2l.load_array(test_data, batch_size, is_train=False)
# # def init_params():
# #     w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
# #     b = torch.zeros(1, requires_grad=True)
# #     return [w, b]
# # def l2_penalty(w):
# #     return torch.sum(w**2) / 2
# # def train(lambd):
# #     w, b = init_params()
# #     net = lambda X: d2l.linreg(X, w, b)
# #     loss = lambda y_hat, y: (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
# #     trainer = torch.optim.SGD([{"params": w, "weight_decay": lambd}, {"params": b}], lr=0.01)
# #     animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs], legend=['train', 'test'])
# #     for epoch in range(num_epochs):
# #         for X, y in train_iter:
# #             trainer.zero_grad()
# #             l = loss(net(X), y) + lambd * l2_penalty(w)
# #             l.sum().backward()
# #             trainer.step()
# #         if (epoch + 1) % 5 == 0:
# #             animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss), d2l.evaluate_loss(net, test_iter, loss)))
# #     print('w的L2范数是：', torch.norm(w).item())

# def train_concise(wd):
#     net = nn.Sequential(nn.Linear(num_inputs, 1))
#     for param in net.parameters():
#         param.data.normal_()
#     loss = nn.MSELoss(reduction='none')
#     num_epochs, lr = 100, 0.003
#     # 偏置参数没有衰减
#     trainer = torch.optim.SGD([
#         {"params":net[0].weight,'weight_decay': wd},
#         {"params":net[0].bias}], lr=lr)
#     animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
#                             xlim=[5, num_epochs], legend=['train', 'test'])
#     for epoch in range(num_epochs):
#         for X, y in train_iter:
#             trainer.zero_grad()
#             l = loss(net(X), y)
#             l.mean().backward()
#             trainer.step()
#         if (epoch + 1) % 5 == 0:
#             animator.add(epoch + 1,
#                          (d2l.evaluate_loss(net, train_iter, loss),
#                           d2l.evaluate_loss(net, test_iter, loss)))
#     print(f'w的L2范数：{torch.norm(net[0].weight).item():f}')

# def main():
#     train_concise(wd=3)
# #     d2l.plt.show()
# # if __name__ == "__main__":
# #     main()

# import torch
# from torch import nn
# from d2l import torch as d2l
# # def dropout_layer(X, dropout):
# #     assert 0 <= dropout <= 1
# #     if dropout == 1:
# #         return torch.zeros_like(X)
# #     if dropout == 0:
# #         return X
# #     mask = (torch.rand(X.shape) > dropout).float()
# #     return mask * X / (1.0 - dropout)
# num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
# dropout1, dropout2 = 0.2, 0.5
# # class Net(nn.Module):
# #     def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, dropout1, dropout2):
# #         super(Net, self).__init__()
# #         self.num_inputs = num_inputs
# #         self.num_outputs = num_outputs
# #         self.num_hiddens1 = num_hiddens1
# #         self.num_hiddens2 = num_hiddens2
# #         self.dropout1 = dropout1
# #         self.dropout2 = dropout2
# #         self.training = True
# #         self.relu = nn.ReLU()
# #         self.linear1 = nn.Linear(num_inputs, num_hiddens1)
# #         self.linear2 = nn.Linear(num_hiddens1, num_hiddens2)
# #         self.linear3 = nn.Linear(num_hiddens2, num_outputs)
# #     def forward(self, X):
# #         H1 = self.relu(self.linear1(X.reshape((-1, self.num_inputs))))
# #         if self.training == True:
# #             H1 = dropout_layer(H1, self.dropout1)
# #         H2 = self.relu(self.linear2(H1))
# #         if self.training == True:
# #             H2 = dropout_layer(H2, self.dropout2)
# #         out = self.linear3(H2)
# #         return out
# # net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2, dropout1, dropout2)
# net=nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Dropout(dropout1), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout2), nn.Linear(256, 10))
# def init_weights(m):
#     if type(m) == nn.Linear:
#         nn.init.normal_(m.weight, std=0.01)
# net.apply(init_weights)


# num_epochs,lr,batch_size = 10,0.5,256
# loss = nn.CrossEntropyLoss(reduction='none')
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# trainer = torch.optim.SGD(net.parameters(), lr=lr)

# def main():
#     train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
#     d2l.plt.show()
# if __name__ == "__main__":
#     main()

# import torch
# from torch import nn
# from d2l import torch as d2l
# from torch.nn import functional as F

# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden = nn.Linear(20, 256)
#         self.out = nn.Linear(256, 10)
#     def forward(self, X):
#         print(X)
#         X = torch.relu(self.hidden(X))
#         return self.out(X)
# net = MLP()
# print(net(torch.rand(2, 20)))
# class MySequential(nn.Module):
#     def __init__(self, *args):
#         super().__init__()
#         for idx, module in enumerate(args):
#             self._modules[str(idx)] = module
#     def forward(self, X):
#         for block in self._modules.values():
#             X = block(X)
#         return X
# net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
# print(net(torch.rand(2, 20)))

# import torch
# from torch import nn
# net = nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,1))
# X=torch.rand(size=(2, 4))
# print(net(X))
# print(net[0].weight)
# print(net[0].bias)
# print(net[2].weight)
# print(net[2].bias)
# print(*[(name, param.shape) for name, param in net.named_parameters()])
# print(*[(name, param.shape) for name, param in net[0].named_parameters()])
# print(net.state_dict()['0.weight'])
# print(net.state_dict()['0.bias'])
# print(net.state_dict()['2.weight'])
# print(net.state_dict()['2.bias'])
# def block1():
#     return nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,4), nn.ReLU())
# def block2():
#     net = nn.Sequential()
#     for i in range(4):
#         net.add_module(f'block {i}', block1())
#     return net
# rgnet = nn.Sequential(block2(), nn.Linear(4,1))
# X=torch.rand(size=(2, 4))
# print(rgnet)
# print(rgnet[0][1][0].weight.data)
# print(*[(name, param.shape) for name, param in rgnet.named_parameters()])
# print(*[(name, param.shape) for name, param in rgnet[0][1][0].named_parameters()])
# print(rgnet[0][1][0].weight)
# print(rgnet[0][1][0].bias)
# print(rgnet[1].weight)
# print(rgnet[1].bias)

# import torch
# from torch import nn

# """延后初始化"""
# net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
# # print(net[0].weight)  # 尚未初始化
# print(net)
# print("--------------------------------")
# X = torch.rand(2, 20)
# net(X)
# print(net)
# import torch
# import torch.nn.functional as F
# from torch import nn
# class CenteredLayer(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, X):
#         return X - X.mean()
# layer = CenteredLayer()
# print(layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)))
# net = nn.Sequential(nn.LazyLinear(8), CenteredLayer())
# print(net(torch.rand(2, 64)))
# print(net)
# class MyLinear(nn.Module):
#     def __init__(self, in_units, units):
#         super().__init__()
#         self.weight = nn.Parameter(torch.randn(in_units, units))
#         self.bias = nn.Parameter(torch.randn(units,))
#     def forward(self, X):
#         linear = torch.matmul(X, self.weight.data) + self.bias.data
#         return F.relu(linear)
# linear = MyLinear(5, 3)
# print(linear.weight)
# print(linear.bias)
# print(linear(torch.rand(2, 5)))
# import torch
# from torch import nn
# from torch.nn import functional as F
# x=torch.arange(4.0)
# y=torch.zeros(4)
# torch.save(x,'x-file.py')
# x2=torch.load('x-file.py')
# print(x2)
# mydict={'x':x,'y':y}
# torch.save(mydict,'mydict.pth')
# mydict2=torch.load('mydict.pth')
# print(mydict2)
# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden = nn.Linear(20, 256)
#         self.out = nn.Linear(256, 10)
#     def forward(self, X):
#         return self.out(F.relu(self.hidden(X)))
# net = MLP()
# X = torch.randn(size=(2, 20))
# Y = net(X)
# print(Y)
# torch.save(net.state_dict(), 'mlp.params')
# mlp = MLP()
# mlp.load_state_dict(torch.load('mlp.params'))
# print(mlp(X))
# # torch.cuda.device_count()
# def try_gpu(i=0):
#     if torch.cuda.device_count() >= i + 1:
#         return torch.device(f'cuda:{i}')
#     return torch.device('cpu')
def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
# # print(try_gpu())
# # print(try_gpu(10))
# # print(try_all_gpus())
# x=torch.arange(4)
# print(x.device)


# import torch
# from torch import nn
# from d2l import torch as d2l

# def corr2d(X, K):
#     h, w = K.shape
#     Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
#     for i in range(Y.shape[0]):
#         for j in range(Y.shape[1]):
#             Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
#     return Y
# # X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
# # K = torch.tensor([[0, 1], [2, 3]])
# # print(corr2d(X, K))

# class Conv2D(nn.Module):
#     def __init__(self, kernel_size):
#         super().__init__()
#         self.weight = nn.Parameter(torch.rand(kernel_size))
#         self.bias = nn.Parameter(torch.zeros(1))
#     def forward(self, X):
#         return corr2d(X, self.weight) + self.bias
# conv2d = Conv2D(kernel_size=(1, 2))
# print(conv2d(X))

# X=torch.ones((6, 8))
# X[:, 2:6] = 0   # 将X的第2到第6列设置为0
# # print(X)
# K = torch.tensor([[1.0, -1.0]])
# # print(K)
# Y = corr2d(X, K)
# # print(Y)
# conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)
# X = X.reshape((1, 1, 6, 8))
# Y = Y.reshape((1, 1, 6, 7))
# lr = 3e-2  # 学习率
# for i in range(10):
#     Y_hat = conv2d(X)
#     l = (Y_hat - Y) ** 2
#     conv2d.zero_grad()
#     l.sum().backward()
#     conv2d.weight.data[:] -= lr * conv2d.weight.grad
#     if (i + 1) % 2 == 0:
#         print(f'epoch {i + 1}, loss {float(l.sum()):f}')
# print(conv2d.weight.data)

# import torch
# from torch import nn
# from d2l import torch as d2l
# from torch.nn import functional as F

# def comp_conv2d(conv2d, X):
#     X = X.reshape((1, 1) + X.shape)
#     Y = conv2d(X)
#     return Y.reshape(Y.shape[2:])
# conv2d = nn.Conv2d(1,1, kernel_size=3, padding=1,stride=2)
# X = torch.rand(size=(8, 8))
# print(comp_conv2d(conv2d, X).shape)

# import torch
# from d2l import torch as d2l
# def corr2d_multi_in(X, K):
#     return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
# # X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
# #                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
# # K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
# # print(corr2d_multi_in(X, K))    
# # print(X.shape)
# # print(K.shape)
# # print(corr2d_multi_in(X, K).shape)  

# def corr2d_multi_in_out(X, K):
#     return torch.stack([corr2d_multi_in(X, k) for k in K], dim=0)
# X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
#                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
# K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
# K=torch.stack((K, K + 1, K + 2))
# print(corr2d_multi_in_out(X, K))    
# print(X.shape)
# print(X)
# print(K.shape)
# print(K)
# print(corr2d_multi_in_out(X, K).shape)  

# def corr2d_multi_in_out_1x1(X, K):
#     c_i, h, w = X.shape
#     c_o = K.shape[0]
#     X = X.reshape((c_i, h * w))
#     K = K.reshape((c_o, c_i))
#     Y = torch.matmul(K, X)
#     return Y.reshape((c_o, h, w))
# X = torch.normal(0, 1, (3, 3, 3))
# K = torch.normal(0, 1, (2, 3, 1, 1))
# print(corr2d_multi_in_out_1x1(X, K))
# print(X.shape)
# print(K.shape)

# import torch
# from torch import nn
# from d2l import torch as d2l
# # def pool2d(X, pool_size, mode='max'):
# #     p_h, p_w = pool_size
# #     Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
# #     for i in range(Y.shape[0]):
# #         for j in range(Y.shape[1]):
# #             if mode == 'max':
# #                 Y[i, j] = X[i:i+p_h, j:j+p_w].max()
# #             elif mode == 'avg':
# #                 Y[i, j] = X[i:i+p_h, j:j+p_w].mean()
# #     return Y
# X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
# print(X)
# # pool2d=nn.MaxPool2d(3)
# pool2d=nn.MaxPool2d(3,padding=1,stride=2)
# print(pool2d(X))
# import torch

# import torch
# from torch import nn
# from d2l import torch as d2l


def evaluate_accuracy(net,data_iter):
            if isinstance(net,torch.nn.Module):
                net.eval()
            metric=Accumulator(2)
            with torch.no_grad():
                for X,y in data_iter:
                    metric.add(accuracy(net(X),y),y.numel())
            return metric[0] / metric[1]
class Accumulator:
            def __init__(self,n):
                self.data=[0.0]*n
            def add(self,*args):
                self.data=[a+float(b) for a,b in zip(self.data,args)]
            def reset(self):
                self.data=[0.0]*len(self.data)
            def __getitem__(self,idx):
                return self.data[idx]
def accuracy(y_hat,y):
            if len(y_hat.shape)>1 and y_hat.shape[1]>1:
                y_hat=y_hat.argmax(dim=1)
            cmp=y_hat.type(y.dtype)==y
            return float(cmp.type(y.dtype).sum())
def train_ch6(net,train_iter,test_iter,num_epochs,lr,device):
            def init_weights(m):
                if type(m)==nn.Linear or type(m)==nn.Conv2d:
                    nn.init.xavier_uniform_(m.weight)
            net.apply(init_weights)
            print('training on',device)
            net.to(device)
            optimizer=torch.optim.SGD(net.parameters(),lr=lr)
            loss=nn.CrossEntropyLoss()
            animator=d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],legend=['train loss','train acc','test acc'])
            timer,num_batches=d2l.Timer(),len(train_iter)
            for epoch in range(num_epochs):
                metric=d2l.Accumulator(3)
                net.train()
                for i,(X,y) in enumerate(train_iter):
                    timer.start()
                    optimizer.zero_grad()
                    X,y=X.to(device),y.to(device)
                    y_hat=net(X)
                    l=loss(y_hat,y)
                    l.backward()
                    optimizer.step()
                    with torch.no_grad():
                        metric.add(l*X.shape[0],accuracy(y_hat,y),y.numel())
                    timer.stop()
                    train_l=metric[0]/metric[2]
                    train_acc=metric[1]/metric[2]
                    for layer in net:
                        X=layer(X)
                        print(layer.__class__.__name__,'output shape:\t',X.shape)
                    if (i+1)%(num_batches//5)==0 or i==num_batches-1:
                        animator.add(epoch+(i+1)/num_batches,(train_l,train_acc,None))
                test_acc=evaluate_accuracy(net,test_iter)
                animator.add(epoch+1,(None,None,test_acc))
            print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
            print(f'{metric[2]*num_epochs/timer.sum():.1f} examples/sec on {str(device)}')


# def vgg_block(num_convs, in_channels, out_channels):
#     layers = []
#     for _ in range(num_convs):
#         layers.append(nn.Conv2d(in_channels, out_channels,
#                                 kernel_size=3, padding=1))
#         layers.append(nn.ReLU())
#         in_channels = out_channels
#     layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
#     return nn.Sequential(*layers)
# conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
# def vgg(conv_arch):
#     conv_blks = []
#     in_channels = 1
#     for (num_convs, out_channels) in conv_arch:
#         conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
#         in_channels = out_channels
#     return nn.Sequential(*conv_blks, nn.Flatten(), nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 10))
# # net = vgg(conv_arch)
# # print(net)
# # X=torch.rand(size=(1, 1, 224, 224))
# # for blk in net:
# #     X=blk(X)
# #     print(blk.__class__.__name__,'output shape:\t',X.shape)
# ratio=4
# small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
# net = vgg(small_conv_arch)
# print(net)
# X=torch.rand(size=(1, 1, 224, 224))
# for blk in net:
#     X=blk(X)
#     print(blk.__class__.__name__,'output shape:\t',X.shape)
# lr,num_epochs,batch_size = 0.05,10,128
# total_params = sum(p.numel() for p in net.parameters())
# trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

# print("总参数量：", total_params)
# print("可训练参数量：", trainable_params)
# train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size,resize=224)
# d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu())




# import torch
# from torch import nn
# from d2l import torch as d2l

# def nin_block(in_channels, out_channels, kernel_size, strides, padding):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
#         nn.ReLU(),
#         nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
#         nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

# print(nin_block(1, 96, kernel_size=11, strides=4, padding=0))

# net = nn.Sequential(
#     nin_block(1, 96, kernel_size=11, strides=4, padding=0),
#     nn.MaxPool2d(3, stride=2),
#     nin_block(96, 256, kernel_size=5, strides=1, padding=2),
#     nn.MaxPool2d(3, stride=2),
#     nin_block(256, 384, kernel_size=3, strides=1, padding=1),
#     nn.MaxPool2d(3, stride=2), nn.Dropout(0.5),
#     nin_block(384, 10, kernel_size=3, strides=1, padding=1),
#     nn.AdaptiveAvgPool2d((1, 1)),
#     nn.Flatten())
# print(net)
# X = torch.randn(size=(1, 1, 224, 224))
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__,'output shape:\t',X.shape)
# lr,num_epochs,batch_size = 0.1,10,128
# total_params = sum(p.numel() for p in net.parameters())
# trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

# print("总参数量：", total_params)
# print("可训练参数量：", trainable_params)
# train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size,resize=224)
# d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu())

# import torch
# from torch import nn
# from d2l import torch as d2l
# from torch.nn import functional as F

# class Inception(nn.Module):
#     def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
#         super(Inception, self).__init__(**kwargs)
#         self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
#         self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
#         self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
#         self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
#         self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
#         self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#         self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
#     def forward(self, x):
#         p1 = F.relu(self.p1_1(x))
#         p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
#         p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
#         p4 = F.relu(self.p4_2(self.p4_1(x)))
#         return torch.cat([p1, p2, p3, p4], dim=1)
# b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
#                    nn.ReLU(),
#                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
#                    nn.ReLU(),
#                    nn.Conv2d(64, 192, kernel_size=3, padding=1),
#                    nn.ReLU(),
#                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
#                    Inception(256, 128, (128, 192), (32, 96), 64),
#                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
#                    Inception(512, 160, (112, 224), (24, 64), 64),
#                    Inception(512, 128, (128, 256), (24, 64), 64),
#                    Inception(512, 112, (144, 288), (32, 64), 64),
#                    Inception(528, 256, (160, 320), (32, 128), 128),
#                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
#                    Inception(832, 384, (192, 384), (48, 128), 128),
#                    nn.AdaptiveAvgPool2d((1,1)),
#                    nn.Flatten())
# net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
# print(net)
# print("--------------------------------")
# print(b5)
# X = torch.rand(size=(1, 1, 96, 96))
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__,'output shape:\t',X.shape)
# lr, num_epochs, batch_size = 0.1, 10, 128
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
# d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

# import torch
# from torch import nn
# from d2l import torch as d2l
# from torch.nn import functional as F

# def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
#     if not torch.is_grad_enabled():
#         X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
#     else:
#         assert len(X.shape) in (2, 4)
#         if len(X.shape) == 2:
#             mean = X.mean(dim=0)
#             var = ((X - mean) ** 2).mean(dim=0)
#         else:
#             mean = X.mean(dim=(0, 2, 3), keepdim=True)
#             var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
#         X_hat = (X - mean) / torch.sqrt(var + eps)
#         moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
#         moving_var = momentum * moving_var + (1.0 - momentum) * var
#     Y = gamma * X_hat + beta
#     return Y, moving_mean, moving_var

# class BatchNorm(nn.Module):
#     def __init__(self, num_features, num_dims):
#         super(BatchNorm, self).__init__()
#         if num_dims == 2:
#             shape = (1, num_features)
#         else:
#             shape = (1, num_features, 1, 1)
#         self.gamma = nn.Parameter(torch.ones(shape))
#         self.beta = nn.Parameter(torch.zeros(shape))
#         self.moving_mean = torch.zeros(shape)
#         self.moving_var = torch.ones(shape)
#     def forward(self, X):
#         if self.moving_mean.device != X.device:
#             self.moving_mean = self.moving_mean.to(X.device)
#             self.moving_var = self.moving_var.to(X.device)
#         Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
#         return Y
    
# # net = nn.Sequential(
# #     nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
# #     nn.AvgPool2d(kernel_size=2, stride=2),
# #     nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
# #     nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
# #     nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
# #     nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
# #     nn.Linear(84, 10))
# net=nn.Sequential(nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
#     nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
#     nn.Linear(16*4*4, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
#     nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
#     nn.Linear(84, 10))
# # print(net)
# # X = torch.rand(size=(1, 1, 28, 28))
# # for layer in net:
# #     X = layer(X)
# #     print(layer.__class__.__name__,'output shape:\t',X.shape)

# lr,num_epochs,batch_size = 0.1,10,32

# # train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
# # for X,y in train_iter:
# #     print(X.shape)
# #     print(y.shape)
# #     break
# # d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu())
# # print(net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,)))
# def main():
#     train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
#     for X,y in train_iter:
#         print(X.shape)
#         print(y.shape)
#         break
#     d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu())
# if __name__ == "__main__":
#     main()

# import torch
# from torch import nn
# from d2l import torch as d2l
# from torch.nn import functional as F
# class Residual(nn.Module):
#     def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
#         self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
#         if use_1x1conv:
#             self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
#         else:
#             self.conv3 = None
#         self.bn1 = nn.BatchNorm2d(num_channels)
#         self.bn2 = nn.BatchNorm2d(num_channels)
#     def forward(self, X):
#         Y = F.relu(self.bn1(self.conv1(X)))
#         Y = self.bn2(self.conv2(Y))
#         if self.conv3:
#             X = self.conv3(X)
#         Y += X
#         return F.relu(Y)
# blk = Residual(3,6,use_1x1conv=True)
# X = torch.rand(4, 3, 6, 6)
# Y = blk(X)
# print(Y.shape)
# b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
#                    nn.BatchNorm2d(64), nn.ReLU(),
#                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# print(b1)
# X = torch.rand(size=(1, 1, 224, 224))
# for layer in b1:
#     X = layer(X)
#     print(layer.__class__.__name__,'output shape:\t',X.shape)
# def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
#     blk = []
#     for i in range(num_residuals):
#         if i == 0 and not first_block:
#             blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
#         else:
#             blk.append(Residual(num_channels, num_channels))
#     return blk
# b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
# b3 = nn.Sequential(*resnet_block(64, 128, 2))
# b4 = nn.Sequential(*resnet_block(128, 256, 2))
# b5 = nn.Sequential(*resnet_block(256, 512, 2))
# net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(512, 10))
# # print(net)
# # X = torch.rand(size=(1, 1, 224, 224))
# # for layer in net:
# #     X = layer(X)
# #     print(layer.__class__.__name__,'output shape:\t',X.shape)
# lr,num_epochs,batch_size = 0.05,10,256
# train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size,resize=96)
# d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu())

import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X
# blk=DenseBlock(2,3,10)
# X=torch.rand(size=(4,3,8,8))
# Y=blk(X)
# print(Y.shape)
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
# blk=transition_block(23,10)
# X=torch.rand(size=(4,23,8,8))
# Y=blk(X)
# print(Y.shape)
# net=nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), DenseBlock(4, 64, 32), transition_block(32, 16), DenseBlock(4, 16, 16), transition_block(16, 8), DenseBlock(4, 8, 8), nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(8, 10))
# print(net)
# X=torch.rand(size=(1, 1, 224, 224))
b1=nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    num_channels += num_convs * growth_rate
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2
net = nn.Sequential(b1, *blks, nn.BatchNorm2d(num_channels), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(num_channels, 10))
# print(net)
# X=torch.rand(size=(1, 1, 224, 224))
# for layer in net:
#     X=layer(X)
#     print(layer.__class__.__name__,'output shape:\t',X.shape)
lr,num_epochs,batch_size = 0.1,10,256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size,resize=96)
d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu())
   