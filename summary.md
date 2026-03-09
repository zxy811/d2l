## 1、内存优化管理方法
    Z[:] = X + Y 这种原地操作通过直接复用变量 Z 原有的内存空间来存储计算结果，避免了像 Z = X + Y 那样因创建新对象而产生的额外内存申请和瞬时内存峰值，从而显著提升了在大规模数据处理时的内存使用效率。
    ```bash
        可以通过下面的代码进行查看变量前后的内存位置是否一致
        import torch

        X = torch.ones(3)
        Y = torch.ones(3)
        Z = torch.zeros(3)

        before_id = id(Z)

        # 情况 A: 普通赋值 (Z 会指向新内存)
        # Z = X + Y 

        # 情况 B: 切片赋值 (Z 在原位更新数据)
        Z[:] = X + Y

        print(id(Z) == before_id)  # 在 PyTorch 中依然输出 True
    ```
## 创建文件夹、写入文件操作、处理缺失值、转为张量格式
    '''bash
        import os
        import pandas as pd
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        data_file = os.path.join(data_dir, 'house_tiny.csv')
        with open(data_file, 'w') as f:
            f.write('NumRooms,Alley,Price\n')  # 列名
            f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
            f.write('2,NA,106000\n')
            f.write('4,NA,178100\n')
            f.write('NA,NA,140000\n')
        data = pd.read_csv(data_file)
        print(data)
        inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
        # 只对数值列用均值填充，避免对字符串列（如 Alley）求 mean 报错
        inputs = inputs.fillna(inputs.mean(numeric_only=True))
        inputs = pd.get_dummies(inputs, dummy_na=True)
        print(inputs)
        print(outputs)
        x=torch.tensor(inputs.to_numpy(dtype=float))
        y=torch.tensor(outputs.to_numpy(dtype=float))
        print(x)
        print(y)
    '''
## 删除缺失值最多的列
    '''bash
        missing = data.isna().sum()
        max_missing = missing.max()
        cols_to_drop = missing[missing == max_missing].index.tolist()
        print(cols_to_drop)
        data = data.drop(columns=cols_to_drop)
        print("删除缺失最多的列:", cols_to_drop)
    '''
## 张量与向量的区别与联系
1. 张量是包含向量的
    '''bash
        张量（Tensor）是一个总称，而向量（Vector）是张量的一种特例。
        标量（Scalar）：0 维张量。例如：3.5（一个孤立的点）。
        向量（Vector）：1 维张量。例如：[1, 2, 3]（一条线）。
        矩阵（Matrix）：2 维张量。例如：一个表格（一个面）。
        张量（Tensor）：N 维数据容器。当维度大于 2 时，我们通常直接叫它张量。
    '''
2. 辨析torch.tensor与torch.arrange
    前者用于手动添加数据，后者则是用于自动生成满足格式要求的序列
## 张量相关的计算公式
    '''bash 
        A.mean()//求平均值
        A.sum()//求数据综合
        A.nume1()//求总数字数
        矩阵降维可以通过对行、列求和或者是求均值都可以达到该目标
        axis=0是按列计算，=1是按行计算
        A.cumsum(axis=0)//这个是累计求和的作用，不会改变张量的维度只是将累计求和结果更新到对应的每一个位置
        torch.dot(x,y)//表示将两个一维向量进行点积进行计算
        torch.mv(x,y)//表示将矩阵*向量进行求向量积进行计算
        torch.mm(x,y)//表示矩阵*矩阵
        torch.norm(u)//表示向量的范数也就是将向量转为一个标量（这里是L2范数也就是平方和开根号）
        torch.abs(u).sum()//表示L1范数也就是绝对值之和
        torch.norm(torch.ones((4, 9)))//Frobenius范数指的是矩阵的平方和的平方根
    '''
## 使用matlab进行绘图
    '''bash
        
        import torch
        import d2l
        import matplotlib.pyplot as plt
        from matplotlib import backend_inline
        import numpy as np

        def use_svg_display():
            """使用svg格式在Jupyter中显示绘图"""
            backend_inline.set_matplotlib_formats('svg')

        def set_figsize(figsize=(3.5, 2.5)):  #@save
            """设置matplotlib的图表大小"""
            use_svg_display()
            d2l.plt.rcParams['figure.figsize'] = figsize

        def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
            """设置matplotlib的轴"""
            axes.set_xlabel(xlabel)
            axes.set_ylabel(ylabel)
            axes.set_xscale(xscale)
            axes.set_yscale(yscale)
            axes.set_xlim(xlim)
            axes.set_ylim(ylim)
            if legend:
                axes.legend(legend)
            axes.grid()

        def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
            """绘制数据点"""
            if legend is None:
                legend = []
            set_figsize(figsize)
            axes = axes if axes else d2l.plt.gca()
            def has_one_axis(X):
                return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__"))
            if has_one_axis(X):
                X = [X]
            if Y is None:
                X, Y = [[]] * len(X), X
            elif has_one_axis(Y):
                Y = [Y]
            if len(X) != len(Y):
                X = X * len(Y)
            axes.cla()
            for x, y, fmt in zip(X, Y, fmts):
                if len(x):
                    axes.plot(x, y, fmt)
                else:
                    axes.plot(y, fmt)
            set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
            #@save
        def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
                ylim=None, xscale='linear', yscale='linear',
                fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
            """绘制数据点"""
            if legend is None:
                legend = []

            set_figsize(figsize)
            axes = axes if axes else d2l.plt.gca()

            # 如果X有一个轴，输出True
            def has_one_axis(X):
                return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                        and not hasattr(X[0], "__len__"))

            if has_one_axis(X):
                X = [X]
            if Y is None:
                X, Y = [[]] * len(X), X
            elif has_one_axis(Y):
                Y = [Y]
            if len(X) != len(Y):
                X = X * len(Y)
            axes.cla()
            for x, y, fmt in zip(X, Y, fmts):
                if len(x):
                    axes.plot(x, y, fmt)
                else:
                    axes.plot(y, fmt)
            set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        x = np.arange(0, 3, 0.1)
        plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])

    '''
## sum函数的反向传播
    '''bash
        这里你需要进行注意就是sum函数本质上是sum=x1+x2+x3+x4因此对于每一个函数而言都是一个一次函数，故而夏敏粒子的输出是【1，1，1，1】
                import torch
                x=torch.arange(4.0)
                print(x)
                x.requires_grad_(True)
                x.grad
                print(x.grad)
                y= torch.dot(x,x)
                print(y)
                y.backward()
                print(y)
                print(x.grad)
                x.grad.zero_()
                y=x.sum()
                print(y)
                y.backward()
                x.grad
                print(x.grad)）
        如果y不是一个标量那么无法直接使用.backward()进行梯度回传，可以进行求导或者是范数将其变为标量后进行梯度回传
    '''
## 模拟掷骰子-概率分布的实验
    '''bash
        import torch
        from torch.distributions import multinomial
        from d2l import torch as d2l

        fair_probs = torch.ones([6]) / 6
        print(fair_probs)
        counts = multinomial.Multinomial(10, fair_probs).sample((500,))
        cum_counts = counts.cumsum(dim=0)
        estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
        print(estimates)
        d2l.set_figsize((6, 4.5))
        for i in range(6):
            d2l.plt.plot(estimates[:, i].numpy(),
                        label=("P(die=" + str(i + 1) + ")"))
        d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
        d2l.plt.gca().set_xlabel('Groups of experiments')
        d2l.plt.gca().set_ylabel('Estimated probability')
        输出结果都是接近1/6
    ...
## 小批量随机梯度下降
    指的是在进行每次梯度下降的时候进行梯度回传如果是使用全量数据集的话那么会很慢所以就可以先抽样一个小批量的样本用于梯度回传，这样的样本叫做是小批量梯度下降
## 线性回归中平方损失函数的统计学正当性：
    线性回归之所以普遍采用均方误差（MSE）作为损失函数，其深层数学依据在于极大似然估计（MLE）与高斯分布的内在等价性。如果我们假设观测数据中的随机噪声服从均值为零的正态分布，那么在数学推导上，最大化观测到当前数据集的总概率（即极大似然），就完全等价于最小化预测值与真实值之间的平方误差之和。这一结论有力地证明了：在噪声符合高斯分布的假设下，最小二乘法不仅是一种直观的几何选择，更是统计学意义上的最优解。\
    下面图片中的结论都是通过极大似然法求出的
![L1与L2对应噪声模型对比图](D:/d2l/image.png)

## 线性回归实现
![线性结构网络图](D:/d2l/image1.png)
    '''bash
        import torch
        import random
        from d2l import torch as d2l
        def synthetic_data(w, b, num_examples):
            """生成y=Xw+b+噪声"""
            X = torch.normal(0, 1, (num_examples, len(w)))
            y = torch.matmul(X, w) + b
            y += torch.normal(0, 0.01, y.shape)
            print(X)
            print(y)
            y=y.reshape((-1, 1))
            print(y)
            return X, y.reshape((-1, 1))
        true_w = torch.tensor([2, -3.4])
        true_b = 4.2
        features, labels = synthetic_data(true_w, true_b, 1000)
        # d2l.set_figsize()
        # d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1);
        # d2l.plt.show()
        def data_iter(batch_size, features, labels):
            num_examples = len(features)
            indices = list(range(num_examples))
            random.shuffle(indices)
            for i in range(0, num_examples, batch_size):
                batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
                yield features[batch_indices], labels[batch_indices]
        batch_size = 10
        for X, y in data_iter(batch_size, features, labels):
            print(f'X shape: {X.shape}, X={X}, y shape: {y.shape}, y={y}')
            break
        w=torch.normal(0, 0.01, size=(2,1), requires_grad=True)
        b=torch.zeros(1, requires_grad=True)
        def linreg(X, w, b):
            return torch.matmul(X, w) + b
        def squared_loss(y_hat, y):
            return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
        def sgd(params, lr, batch_size):
            with torch.no_grad():
                for param in params:
                    param -= lr * param.grad / batch_size
                    param.grad.zero_()
        lr = 0.03
        num_epochs = 3
        net = linreg
        loss = squared_loss
        for epoch in range(num_epochs):
            for X, y in data_iter(batch_size, features, labels):
                l = loss(net(X, w, b), y)
                l.sum().backward()
                sgd([w, b], lr, batch_size)
            with torch.no_grad():
                train_l = loss(net(features, w, b), labels)
                print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
        print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
        print(f'b的估计误差: {true_b - b}')
    '''
## 线性回归简洁实现
    '''bash 
        import torch
        import numpy as np
        from torch.utils import data
        from d2l import torch as d2l
        from torch import nn

        true_w = torch.tensor([2, -3.4])
        true_b = 4.2
        features, labels = d2l.synthetic_data(true_w, true_b, 1000)
        def load_array(data_arrays, batch_size, is_train=True):  #@save
            """构造一个PyTorch数据迭代器"""
            dataset = data.TensorDataset(*data_arrays)
            return data.DataLoader(dataset, batch_size, shuffle=is_train)
        batch_size = 10
        data_iter = load_array((features, labels), batch_size)
        # for X, y in data_iter:
        #     print(f'X shape: {X.shape}, X={X}, y shape: {y.shape}, y={y}')
        #     break
        next(iter(data_iter))
        net = nn.Sequential(nn.Linear(2, 1))
        net[0].weight.data.normal_(0, 0.01)
        net[0].bias.data.fill_(0)
        loss=nn.MSELoss()
        trainer=torch.optim.SGD(net.parameters(), lr=0.03)
        num_epochs = 3
        for epoch in range(num_epochs):
            for X, y in data_iter:
                l = loss(net(X) ,y)
                trainer.zero_grad()
                l.backward()
                # print(net[0].weight.grad)
                # print(net[0].bias.grad)
                trainer.step()
            l = loss(net(features), labels)
            print(f'epoch {epoch + 1}, loss {float(l.mean()):f}')
        w=net[0].weight.data
        b=net[0].bias.data
        print(f'w的估计误差: {true_w - net[0].weight.data.reshape(true_w.shape)}')
        print(f'b的估计误差: {true_b - net[0].bias.data[0]}')
    '''
## 独热编码
    独热编码是一个向量，他的分量和类别一样多，类别对应的分量是1，其余则是0，例如有三种类别，我们可以使用(1,0,0)对应于猫、（0，1，0）对应于鸡、（0，0，1）对应于狗
## 交叉熵损失
![交叉熵损失](D:/d2l/image3.png)
1. 损失函数的设计逻辑：为了衡量预测的好坏，引入了交叉熵损失。它利用独热编码作为“开关”只提取正确类别的概率，并通过负对数）将概率转化为损失值。这样设计的妙处在于：当预测越离谱，损失就呈爆炸式增长，从而形成强大的惩罚机制。
2. 参数的闭环更新：在训练循环中，通过 l.backward() 计算梯度（即每个参数对误差的“贡献度”），再利用 trainer.step() 根据梯度微调权重。由于交叉熵与 Softmax 在数学上能完美抵消指数项，这使得模型更新的步伐（梯度）刚好等于“预测值与真值的差值”，保证了训练的既快又稳。
## 下载加载数据集
    '''bash
        
            import torch
            import torchvision
            from torch.utils import data
            from torchvision import transforms
            from d2l import torch as d2l

            d2l.use_svg_display()
            trans = transforms.ToTensor()
            mnist_train = torchvision.datasets.FashionMNIST(
                root="../data", train=True, transform=trans, download=True)
            mnist_test = torchvision.datasets.FashionMNIST(
                root="../data", train=False, transform=trans, download=True)
            print(len(mnist_train), len(mnist_test))
            print(mnist_train[0][0].shape)
            def get_fashion_mnist_labels(labels):  #@save
                """返回Fashion-MNIST数据集的文本标签"""
                text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                            'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
                return [text_labels[int(i)] for i in labels]
            def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
                """绘制图像列表"""
                figsize = (num_cols * scale, num_rows * scale)
                _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
                axes = axes.flatten()
                for i, (ax, img) in enumerate(zip(axes, imgs)):
                    if torch.is_tensor(img):
                        ax.imshow(img.numpy())
                    else:
                        ax.imshow(img)
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    if titles:
                        ax.set_title(titles[i])
                return axes
            X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
            show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
            batch_size = 256
            def get_dataloader_workers():  #@save
                """使用4个进程来读取数据"""
                return 4
            train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())
            timer = d2l.Timer()
            for X, y in train_iter:
                continue
            print(f'{timer.stop():.2f} sec')
            def load_data_fashion_mnist(batch_size, resize=None):  #@save
                """下载Fashion-MNIST数据集，然后将其加载到内存中"""
                trans = [transforms.ToTensor()]
                if resize:
                    trans.insert(0, transforms.Resize(resize))
                trans = transforms.Compose(trans)
                mnist_train = torchvision.datasets.FashionMNIST(
                    root="../data", train=True, transform=trans, download=True)
                mnist_test = torchvision.datasets.FashionMNIST(
                    root="../data", train=False, transform=trans, download=True)
                return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()), data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))
            train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
            for X, y in train_iter:
                print(X.shape, X.dtype, y.shape, y.dtype)
                break
    '''
## softmax多分类问题实现
![softmax结构网络图](D:/d2l/image2.png)
1. softmax是一个非线性函数但是softmax回归的输出仍然是由输入特征的仿射变换决定的，因此softmax回归是一个线性模型
    '''bash
        
            import torch
            from IPython import display
            from d2l import torch as d2l

            batch_size = 256
            train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
            num_inputs = 784
            num_outputs = 10
            w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
            b = torch.zeros(num_outputs, requires_grad=True)
            def softmax(X):
                X_exp = torch.exp(X)
                partition = X_exp.sum(1, keepdim=True)
                return X_exp / partition
            def net(X):
                return softmax(torch.matmul(X.reshape(-1, w.shape[0]), w) + b)
            def cross_entropy(y_hat, y):
                return -torch.log(y_hat[range(len(y_hat)), y])

            y=torch.tensor([0, 2])
            y_hat=torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
            # a=torch.tensor(y_hat[[0,1], y])
            # print(y_hat)
            # print(a)
            # b=cross_entropy(y_hat, y)
            # print(b)
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

            # def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
            #     animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
            #                         legend=['train loss', 'train acc', 'test acc'])
            #     for epoch in range(num_epochs):
            #         train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
            #         test_acc = evaluate_accuracy(net, test_iter)
            #         animator.add(epoch + 1, train_metrics + (test_acc,))
            #     train_loss, train_acc = train_metrics
            #     assert train_loss < 0.5, train_loss
            #     assert train_acc <= 1 and train_acc > 0.7, train_acc
            #     assert test_acc <= 1 and test_acc > 0.7, test_acc
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


            def train_epoch_ch3(net, train_iter, loss, updater):
                if isinstance(net, torch.nn.Module):
                    net.train()
                metric = Accumulator(3)
                for X, y in train_iter:
                    y_hat = net(X)
                    # print(y_hat)
                    l = loss(y_hat, y)
                    # print(l)
                    if isinstance(updater, torch.optim.Optimizer):
                        updater.zero_grad()
                        l.mean().backward()
                        updater.step()
                    else:
                        l.sum().backward()
                        updater(X.shape[0])
                    metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
                    # print(metric)
                return metric[0] / metric[2], metric[1] / metric[2]

            def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
                # for X, y in train_iter:
                #     print(X.shape, X.dtype, y.shape, y.dtype)
                #     print(X)
                #     print(y)
                #     break
                animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                                    legend=['train loss', 'train acc', 'test acc'])
                for epoch in range(num_epochs):
                    train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
                    test_acc = evaluate_accuracy(net, test_iter)
                    print(train_metrics)
                    print(test_acc)
                    animator.add(epoch + 1, train_metrics + (test_acc,))
                train_loss, train_acc = train_metrics
                assert train_loss < 0.5, train_loss
                assert train_acc <= 1 and train_acc > 0.7, train_acc
                assert test_acc <= 1 and test_acc > 0.7, test_acc

            def predict_ch3(net, test_iter, n=6):
                for X, y in test_iter:
                    break
                trues = d2l.get_fashion_mnist_labels(y)
                preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
                titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
                d2l.show_images(
                    X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
            # predict_ch3(net, test_iter)
            def updater(batch_size):
                return d2l.sgd([w, b], lr, batch_size)
            # print(evaluate_accuracy(net, test_iter))
            def main():
                # print(accuracy(y_hat

                train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
            if __name__ == "__main__":
                lr=0.1
                num_epochs=10
                main()
                d2l.plt.show()

    '''
## softmax的简洁实现
    '''bash
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

        # # def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
        # #     animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
        # #                         legend=['train loss', 'train acc', 'test acc'])
        # #     for epoch in range(num_epochs):
        # #         train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        # #         test_acc = evaluate_accuracy(net, test_iter)
        # #         animator.add(epoch + 1, train_metrics + (test_acc,))
        # #     train_loss, train_acc = train_metrics
        # #     assert train_loss < 0.5, train_loss
        # #     assert train_acc <= 1 and train_acc > 0.7, train_acc
        # #     assert test_acc <= 1 and test_acc > 0.7, test_acc
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


        def train_epoch_ch3(net, train_iter, loss, updater):
            if isinstance(net, torch.nn.Module):
                net.train()
            metric = Accumulator(3)
            for X, y in train_iter:
                y_hat = net(X)
                # print(y_hat)
                l = loss(y_hat, y)
                # print(l)
                if isinstance(updater, torch.optim.Optimizer):
                    updater.zero_grad()
                    l.mean().backward()
                    updater.step()
                else:
                    l.sum().backward()
                    updater(X.shape[0])
                metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
                # print(metric)
            return metric[0] / metric[2], metric[1] / metric[2]

        def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
            # for X, y in train_iter:
            #     print(X.shape, X.dtype, y.shape, y.dtype)
            #     print(X)
            #     print(y)
            #     break
            animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                                legend=['train loss', 'train acc', 'test acc'])
            for epoch in range(num_epochs):
                train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
                test_acc = evaluate_accuracy(net, test_iter)
                print(train_metrics)
                print(test_acc)
                animator.add(epoch + 1, train_metrics + (test_acc,))
            train_loss, train_acc = train_metrics
            assert train_loss < 0.5, train_loss
            assert train_acc <= 1 and train_acc > 0.7, train_acc
            assert test_acc <= 1 and test_acc > 0.7, test_acc

        # def predict_ch3(net, test_iter, n=6):
        #     for X, y in test_iter:
        #         break
        #     trues = d2l.get_fashion_mnist_labels(y)
        #     preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
        #     titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
        #     d2l.show_images(
        #         X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
        # # predict_ch3(net, test_iter)
        # def updater(batch_size):
        #     return d2l.sgd([w, b], lr, batch_size)
        # # print(evaluate_accuracy(net, test_iter))
        # def main():
        #     # print(accuracy(y_hat

        #     train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
        # if __name__ == "__main__":
        #     lr=0.1
        #     num_epochs=10
        #     main()
        #     d2l.plt.show()

        import torch
        from torch import nn
        from d2l import torch as d2l
        batch_size = 256
        train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
        net=nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)
        net.apply(init_weights)
        loss=nn.CrossEntropyLoss(reduction='none')
        trainer=torch.optim.SGD(net.parameters(), lr=0.1)
        num_epochs=10
        def train_epoch_ch3(net, train_iter, loss, updater):
            if isinstance(net, torch.nn.Module):
                net.train()
            metric = Accumulator(3)
            for X, y in train_iter:
                y_hat = net(X)
                # print(y_hat)
                l = loss(y_hat, y)
                # print(l)
                if isinstance(updater, torch.optim.Optimizer):
                    updater.zero_grad()
                    l.mean().backward()
                    updater.step()
                else:
                    l.sum().backward()
                    updater(X.shape[0])
                metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
                # print(metric)
            return metric[0] / metric[2], metric[1] / metric[2]

        def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
            # for X, y in train_iter:
            #     print(X.shape, X.dtype, y.shape, y.dtype)
            #     print(X)
            #     print(y)
            #     break
            animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                                legend=['train loss', 'train acc', 'test acc'])
            for epoch in range(num_epochs):
                train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
                test_acc = evaluate_accuracy(net, test_iter)
                print(train_metrics)
                print(test_acc)
                animator.add(epoch + 1, train_metrics + (test_acc,))
            train_loss, train_acc = train_metrics
            assert train_loss < 0.5, train_loss
            assert train_acc <= 1 and train_acc > 0.7, train_acc
            assert test_acc <= 1 and test_acc > 0.7, test_acc
        def main():
            train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

        if __name__ == "__main__":
            main()
            d2l.plt.show()
    ...
## 激活函数的实现
1. ReLU激活函数
![ReLU激活函数](D:/d2l/image4.png)
2. sigmoid函数
![sigmoid激活函数](D:/d2l/image5.png)
3. tanh函数
![tanh激活函数](D:/d2l/image6.png)
## 感知机的实现
    实际上这个多层感知机的实现与softmax是基本上一样的只不过是在网络结构上面有区别而已
    '''bash
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



        def train_epoch_ch3(net, train_iter, loss, updater):
            if isinstance(net, torch.nn.Module):
                net.train()
            metric = Accumulator(3)
            for X, y in train_iter:
                y_hat = net(X)
                # print(y_hat)
                l = loss(y_hat, y)
                # print(l)
                if isinstance(updater, torch.optim.Optimizer):
                    updater.zero_grad()
                    l.mean().backward()
                    updater.step()
                else:
                    l.sum().backward()
                    updater(X.shape[0])
                metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
                # print(metric)
            return metric[0] / metric[2], metric[1] / metric[2]

        def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
            # for X, y in train_iter:
            #     print(X.shape, X.dtype, y.shape, y.dtype)
            #     print(X)
            #     print(y)
            #     break
            animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                                legend=['train loss', 'train acc', 'test acc'])
            for epoch in range(num_epochs):
                train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
                test_acc = evaluate_accuracy(net, test_iter)
                print(train_metrics)
                print(test_acc)
                animator.add(epoch + 1, train_metrics + (test_acc,))
            train_loss, train_acc = train_metrics
            assert train_loss < 0.5, train_loss
            assert train_acc <= 1 and train_acc > 0.7, train_acc
            assert test_acc <= 1 and test_acc > 0.7, test_acc
       

        import torch
        from torch import nn
        from d2l import torch as d2l
        batch_size = 256
        train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
        num_inputs, num_outputs, num_hiddens = 784, 10, 256
        W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
        b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
        W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
        b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
        params = [W1, b1, W2, b2]
        def relu(X):
            a = torch.zeros_like(X)
            return torch.max(X, a)
        def net(X):
            X = X.reshape((-1, W1.shape[0]))
            H = relu(X @ W1 + b1)
            return H @ W2 + b2
        loss=nn.CrossEntropyLoss(reduction='none')
        trainer=torch.optim.SGD(params, lr=0.1)
        num_epochs=10
        def main():
            train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
            d2l.plt.show()
        if __name__ == "__main__":
            main()
    '''
## 感知机的简洁实现
'''bash
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


        def train_epoch_ch3(net, train_iter, loss, updater):
            if isinstance(net, torch.nn.Module):
                net.train()
            metric = Accumulator(3)
            for X, y in train_iter:
                y_hat = net(X)
                # print(y_hat)
                l = loss(y_hat, y)
                # print(l)
                if isinstance(updater, torch.optim.Optimizer):
                    updater.zero_grad()
                    l.mean().backward()
                    updater.step()
                else:
                    l.sum().backward()
                    updater(X.shape[0])
                metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
                # print(metric)
            return metric[0] / metric[2], metric[1] / metric[2]

        def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
            # for X, y in train_iter:
            #     print(X.shape, X.dtype, y.shape, y.dtype)
            #     print(X)
            #     print(y)
            #     break
            animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                                legend=['train loss', 'train acc', 'test acc'])
            for epoch in range(num_epochs):
                train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
                test_acc = evaluate_accuracy(net, test_iter)
                print(train_metrics)
                print(test_acc)
                animator.add(epoch + 1, train_metrics + (test_acc,))
            train_loss, train_acc = train_metrics
            assert train_loss < 0.5, train_loss
            assert train_acc <= 1 and train_acc > 0.7, train_acc
            assert test_acc <= 1 and test_acc > 0.7, test_acc

        import torch
        from torch import nn
        from d2l import torch as d2l
        net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256),nn.ReLU(),nn.Linear(256, 10))
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)
        net.apply(init_weights)
        batch_size,lr,num_epochs=256,0.1,10
        loss=nn.CrossEntropyLoss(reduction='none')
        trainer=torch.optim.SGD(net.parameters(), lr=lr)
        train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
        def main():
            train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
            d2l.plt.show()
        if __name__ == "__main__":
            main()
'''
## 正则化