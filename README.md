## 内存优化管理方法
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
![L1与L2对应噪声模型对比图](D:/d2l/image.png)image

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
## 正则化-权重衰减-L2正则化
用于对抗过拟合的手段成为正则化
    '''bash
        import torch
        from torch import nn
        from d2l import torch as d2l
        n_train, n_test, num_inputs, batch_size,num_epochs = 20, 100, 200, 5, 100 
        true_w = torch.ones((num_inputs, 1)) * 0.01
        true_b = 0.05
        train_data = d2l.synthetic_data(true_w, true_b, n_train)
        train_iter = d2l.load_array(train_data, batch_size)
        test_data = d2l.synthetic_data(true_w, true_b, n_test)
        test_iter = d2l.load_array(test_data, batch_size, is_train=False)
        # def init_params():
        #     w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
        #     b = torch.zeros(1, requires_grad=True)
        #     return [w, b]
        # def l2_penalty(w):
        #     return torch.sum(w**2) / 2
        # def train(lambd):
        #     w, b = init_params()
        #     net = lambda X: d2l.linreg(X, w, b)
        #     loss = lambda y_hat, y: (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
        #     trainer = torch.optim.SGD([{"params": w, "weight_decay": lambd}, {"params": b}], lr=0.01)
        #     animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs], legend=['train', 'test'])
        #     for epoch in range(num_epochs):
        #         for X, y in train_iter:
        #             trainer.zero_grad()
        #             l = loss(net(X), y) + lambd * l2_penalty(w)
        #             l.sum().backward()
        #             trainer.step()
        #         if (epoch + 1) % 5 == 0:
        #             animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss), d2l.evaluate_loss(net, test_iter, loss)))
        #     print('w的L2范数是：', torch.norm(w).item())

        def train_concise(wd):
            net = nn.Sequential(nn.Linear(num_inputs, 1))
            for param in net.parameters():
                param.data.normal_()
            loss = nn.MSELoss(reduction='none')
            num_epochs, lr = 100, 0.003
            # 偏置参数没有衰减
            trainer = torch.optim.SGD([
                {"params":net[0].weight,'weight_decay': wd},
                {"params":net[0].bias}], lr=lr)
            animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                                    xlim=[5, num_epochs], legend=['train', 'test'])
            for epoch in range(num_epochs):
                for X, y in train_iter:
                    trainer.zero_grad()
                    l = loss(net(X), y)
                    l.mean().backward()
                    trainer.step()
                if (epoch + 1) % 5 == 0:
                    animator.add(epoch + 1,
                                (d2l.evaluate_loss(net, train_iter, loss),
                                d2l.evaluate_loss(net, test_iter, loss)))
            print(f'w的L2范数：{torch.norm(net[0].weight).item():f}')

        def main():
            train_concise(wd=3)
            d2l.plt.show()
        if __name__ == "__main__":
            main()
    '''
## 正则化-dropout
自定义实现
'''bash

import torch
from torch import nn
from d2l import torch as d2l
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, dropout1, dropout2):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens1 = num_hiddens1
        self.num_hiddens2 = num_hiddens2
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.training = True
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(num_inputs, num_hiddens1)
        self.linear2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.linear3 = nn.Linear(num_hiddens2, num_outputs)
    def forward(self, X):
        H1 = self.relu(self.linear1(X.reshape((-1, self.num_inputs))))
        if self.training == True:
            H1 = dropout_layer(H1, self.dropout1)
        H2 = self.relu(self.linear2(H1))
        if self.training == True:
            H2 = dropout_layer(H2, self.dropout2)
        out = self.linear3(H2)
        return out
net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2, dropout1, dropout2)
num_epochs,lr,batch_size = 10,0.5,256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)

def main():
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    d2l.plt.show()
if __name__ == "__main__":
    main()
'''
简洁实现
'''bash 
import torch
from torch import nn
from d2l import torch as d2l

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5

net=nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Dropout(dropout1), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout2), nn.Linear(256, 10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)


num_epochs,lr,batch_size = 10,0.5,256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)

def main():
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    d2l.plt.show()
if __name__ == "__main__":
    main()
'''
## 多项式回归问题
    '''bash           
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
    '''
## Kaggle 房价预测
'''bash
        import hashlib
        import os
        import tarfile
        import zipfile
        import requests
        import pandas as pd
        import torch
        from torch import nn
        from d2l import torch as d2l
        import numpy as np


        #@save
        DATA_HUB = dict()
        DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

        def download(name, cache_dir=os.path.join('..', 'data')):  #@save
            """下载一个DATA_HUB中的文件，返回本地文件名"""
            assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
            url, sha1_hash = DATA_HUB[name]
            os.makedirs(cache_dir, exist_ok=True)
            fname = os.path.join(cache_dir, url.split('/')[-1])
            if os.path.exists(fname):
                sha1 = hashlib.sha1()
                with open(fname, 'rb') as f:
                    while True:
                        data = f.read(1048576)
                        if not data:
                            break
                        sha1.update(data)
                if sha1.hexdigest() == sha1_hash:
                    return fname  # 命中缓存
            print(f'正在从{url}下载{fname}...')
            r = requests.get(url, stream=True, verify=True)
            with open(fname, 'wb') as f:
                f.write(r.content)
            return fname
        # download('kaggle_house_train')
        def download_extract(name, folder=None):  #@save
            """下载并解压zip/tar文件"""
            fname = download(name)
            base_dir = os.path.dirname(fname)
            data_dir, ext = os.path.splitext(fname)
            if ext == '.zip':
                fp = zipfile.ZipFile(fname, 'r')
            elif ext == '.tar':
                fp = tarfile.open(fname, 'r')
            else:
                assert False, '只有zip/tar文件可以被解压缩'
            fp.extractall(base_dir)
            return os.path.join(base_dir, folder) if folder else data_dir
        def download_all():  #@save
            """下载DATA_HUB中的所有文件"""
            for name in DATA_HUB:
                download(name)
        DATA_HUB['kaggle_house_train'] = (  #@save
        DATA_URL + 'kaggle_house_pred_train.csv','585e9cc93e70b39160e7921475f9bcd7d31219ce')
        DATA_HUB['kaggle_house_test'] = (  #@save
        DATA_URL + 'kaggle_house_pred_test.csv','fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
        train_data = pd.read_csv(download('kaggle_house_train'))
        test_data = pd.read_csv(download('kaggle_house_test'))
        print(train_data.shape)
        print(test_data.shape)
        # print(train_data.head())
        # print(test_data.head())
        # print(train_data.info())
        # print(test_data.info())
        all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
        print(all_features.shape)
        numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
        all_features[numeric_features] = all_features[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std()))
        # 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
        all_features[numeric_features] = all_features[numeric_features].fillna(0)
        all_features = pd.get_dummies(all_features, dummy_na=True)
        print(all_features.shape)
        n_train = train_data.shape[0]
        train_features = torch.tensor(all_features[:n_train].to_numpy(dtype=np.float32))
        test_features = torch.tensor(all_features[n_train:].to_numpy(dtype=np.float32))
        train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

        print(train_features.shape)
        print(test_features.shape)
        print(train_labels.shape)
        loss=nn.MSELoss()
        in_features = train_features.shape[1]
        def get_net():
            net = nn.Sequential(nn.Linear(in_features,1))
            return net
        net = get_net()
        def log_rmse(net, features, labels):
            clipp_preds = torch.clamp(net(features), 1, float('inf'))
            rmse = torch.sqrt(loss(torch.log(clipp_preds), torch.log(labels)))
            return rmse.item()
        def train(net, train_features, train_labels, test_features, test_labels, num_epochs, lr, weight_decay, batch_size):
            train_ls, test_ls = [], []
            train_iter = d2l.load_array((train_features, train_labels), batch_size)
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
            net.train()
            for epoch in range(num_epochs):
                for X, y in train_iter:
                    optimizer.zero_grad()
                    l = loss(net(X), y)
                    l.backward()
                    optimizer.step()
                train_ls.append(log_rmse(net, train_features, train_labels))
                if test_labels is not None:
                    test_ls.append(log_rmse(net, test_features, test_labels))
            return train_ls, test_ls
        def get_k_fold_data(k, i, X, y):
            assert k > 1
            fold_size = X.shape[0] // k
            X_train, y_train = None, None
            for j in range(k):
                idx = slice(j * fold_size, (j + 1) * fold_size)
                X_part, y_part = X[idx, :], y[idx]
                if j == i:
                    X_valid, y_valid = X_part, y_part
                elif X_train is None:
                    X_train, y_train = X_part, y_part
                else:
                    X_train = torch.cat([X_train, X_part], 0)
                    y_train = torch.cat([y_train, y_part], 0)
            return X_train, y_train, X_valid, y_valid
        #下面的这个是K折交叉验证，一般用于我们去选超参数的设置的，然后你试出来哪一个最好之后并于下面的计算
        # def k_fold(k, X_train, y_train, num_epochs, lr, weight_decay, batch_size):
        #     train_l_sum, valid_l_sum = 0, 0
        #     for i in range(k):
        #         data = get_k_fold_data(k, i, X_train, y_train)
        #         net = get_net()
        #         train_ls, valid_ls = train(net, *data, num_epochs, lr, weight_decay, batch_size)
        #         train_l_sum += train_ls[-1]
        #         valid_l_sum += valid_ls[-1]
        #         if i == 0:
        #             d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls], xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs], legend=['train', 'valid'], yscale='log')
        #         print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, 验证log rmse{float(valid_ls[-1]):f}')
        #     return train_l_sum / k, valid_l_sum / k
        k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
        # train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
        # print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, 平均验证log rmse: {float(valid_l):f}')
        # d2l.plt.show()

        def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
            net = get_net()
            train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
            d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs], yscale='log')
            print(f'训练log rmse：{float(train_ls[-1]):f}')
            preds = net(test_features).detach().numpy()
            test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
            submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
            submission.to_csv('submission.csv', index=False)
        train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
'''
## 多输入输出通道的实现
就是这里需要注意你如果是多个输入通道但是只有一个输出通道那么这个情况就是，K的三个通道在返回的时候直接叠加了，如果你想要实现多个输出通道那么就相当于进行了多次上面的重复的计算
'''bash
        import torch
        from d2l import torch as d2l
        def corr2d_multi_in(X, K):
            return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
        # X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
        #                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
        # K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
        # print(corr2d_multi_in(X, K))    
        # print(X.shape)
        # print(K.shape)
        # print(corr2d_multi_in(X, K).shape)  

        def corr2d_multi_in_out(X, K):
            return torch.stack([corr2d_multi_in(X, k) for k in K], dim=0)
        X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
        K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
        K=torch.stack((K, K + 1, K + 2))
        print(corr2d_multi_in_out(X, K))    
        print(X.shape)
        print(X)
        print(K.shape)
        print(K)
        print(corr2d_multi_in_out(X, K).shape)  
'''
## 为什么有时候需要降低分辨率
降低分辨率并不是简单地抛弃信息，而是通过聚合和抽象，将像素级的细节转化为语义级的表示。它带来的好处包括：计算高效、感受野扩大、平移不变性增强、过拟合减少、特征层次化。因此，尽管直觉上“更清楚更好”，但在深度学习中，合适的分辨率变化才是设计的关键。
## 池化层/汇聚层
1. 实际上就是一个窗口与卷积一样只不过不是进行交叉运算而是将窗口内的数字取平均或者是最大，从而形成新的数字替代该位置\
2. 与卷积不同的是汇聚层的输入输出通道数目是一样的不会在最后输出的时候进行求和运算
    '''bash
        import torch
        from torch import nn
        from d2l import torch as d2l
        # def pool2d(X, pool_size, mode='max'):
        #     p_h, p_w = pool_size
        #     Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
        #     for i in range(Y.shape[0]):
        #         for j in range(Y.shape[1]):
        #             if mode == 'max':
        #                 Y[i, j] = X[i:i+p_h, j:j+p_w].max()
        #             elif mode == 'avg':
        #                 Y[i, j] = X[i:i+p_h, j:j+p_w].mean()
        #     return Y
        X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
        print(X)
        # pool2d=nn.MaxPool2d(3)
        pool2d=nn.MaxPool2d(3,padding=1,stride=2)
        print(pool2d(X))

    '''
## LeNet的实现
    '''bash
        import torch
        from torch import nn
        from d2l import torch as d2l


        net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10))
        print(net)
        # x=torch.rand(size=(1, 1, 28, 28))
        # for layer in net:
        #     x=layer(x)
        #     print(layer.__class__.__name__,'output shape:\t',x.shape)
        batch_size=256
        train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
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
        device=d2l.try_gpu()
        print(f'training on {device}')
        net.to(device)
        train_ch6(net,train_iter,test_iter,num_epochs=10,lr=0.1,device=device)
    '''
## VGG模块的实现
    '''bash
        def vgg_block(num_convs, in_channels, out_channels):
            layers = []
            for _ in range(num_convs):
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
            return nn.Sequential(*layers)
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        def vgg(conv_arch):
            conv_blks = []
            in_channels = 1
            for (num_convs, out_channels) in conv_arch:
                conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
                in_channels = out_channels
            return nn.Sequential(*conv_blks, nn.Flatten(), nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 10))
        # net = vgg(conv_arch)
        # print(net)
        # X=torch.rand(size=(1, 1, 224, 224))
        # for blk in net:
        #     X=blk(X)
        #     print(blk.__class__.__name__,'output shape:\t',X.shape)
        ratio=4
        small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
        net = vgg(small_conv_arch)
        print(net)
        X=torch.rand(size=(1, 1, 224, 224))
    '''
## nin模型的实现
这个的突破性实现就是不使用全连接层，而是使用1*1的卷积代替全连接层的逻辑
    '''bash
        import torch
        from torch import nn
        from d2l import torch as d2l

        def nin_block(in_channels, out_channels, kernel_size, strides, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

        print(nin_block(1, 96, kernel_size=11, strides=4, padding=0))

        net = nn.Sequential(
            nin_block(1, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            nin_block(96, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            nin_block(256, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2), nn.Dropout(0.5),
            nin_block(384, 10, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())
        print(net)
        X = torch.randn(size=(1, 1, 224, 224))
        for layer in net:
            X = layer(X)
            print(layer.__class__.__name__,'output shape:\t',X.shape)
        lr,num_epochs,batch_size = 0.1,10,128
        train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size,resize=224)
        d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu())
    '''
## 计算模型的参数量
        '''bash
            total_params = sum(p.numel() for p in net.parameters())
            trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

            print("总参数量：", total_params)
            print("可训练参数量：", trainable_params)
        '''
## goolenet
这个里面也还是存在全连接层的，只不过与VGG相比实现了降维因为他在最后加了一个全局平均池化进行了降维也就是将图片的尺度降为1*1，大大减少了后面全连接层的参数量
    '''bash

        import torch
        from torch import nn
        from d2l import torch as d2l
        from torch.nn import functional as F

        class Inception(nn.Module):
            def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
                super(Inception, self).__init__(**kwargs)
                self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
                self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
                self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
                self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
                self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
                self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
            def forward(self, x):
                p1 = F.relu(self.p1_1(x))
                p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
                p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
                p4 = F.relu(self.p4_2(self.p4_1(x)))
                return torch.cat([p1, p2, p3, p4], dim=1)
        b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 192, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                        Inception(256, 128, (128, 192), (32, 96), 64),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                        Inception(512, 160, (112, 224), (24, 64), 64),
                        Inception(512, 128, (128, 256), (24, 64), 64),
                        Inception(512, 112, (144, 288), (32, 64), 64),
                        Inception(528, 256, (160, 320), (32, 128), 128),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                        Inception(832, 384, (192, 384), (48, 128), 128),
                        nn.AdaptiveAvgPool2d((1,1)),
                        nn.Flatten())
        net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
        print(net)
        X = torch.rand(size=(1, 1, 96, 96))
        for layer in net:
            X = layer(X)
            print(layer.__class__.__name__,'output shape:\t',X.shape)
        lr, num_epochs, batch_size = 0.1, 10, 128
        train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
        d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    '''
## 数据的批量规范化
进行数据批量化的原因主要是：一是为了解决数据之间大小不均衡的问题、二是进行这样的均衡化之后相当于引入了噪声模型的效果更好，类似于进行了正则化，使得模型不容易过拟合

    '''bash

        import torch
        from torch import nn
        from d2l import torch as d2l
        from torch.nn import functional as F

        def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
            if not torch.is_grad_enabled():
                X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
            else:
                assert len(X.shape) in (2, 4)
                if len(X.shape) == 2:
                    mean = X.mean(dim=0)
                    var = ((X - mean) ** 2).mean(dim=0)
                else:
                    mean = X.mean(dim=(0, 2, 3), keepdim=True)
                    var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
                X_hat = (X - mean) / torch.sqrt(var + eps)
                moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
                moving_var = momentum * moving_var + (1.0 - momentum) * var
            Y = gamma * X_hat + beta
            return Y, moving_mean, moving_var

        class BatchNorm(nn.Module):
            def __init__(self, num_features, num_dims):
                super(BatchNorm, self).__init__()
                if num_dims == 2:
                    shape = (1, num_features)
                else:
                    shape = (1, num_features, 1, 1)
                self.gamma = nn.Parameter(torch.ones(shape))
                self.beta = nn.Parameter(torch.zeros(shape))
                self.moving_mean = torch.zeros(shape)
                self.moving_var = torch.ones(shape)
            def forward(self, X):
                if self.moving_mean.device != X.device:
                    self.moving_mean = self.moving_mean.to(X.device)
                    self.moving_var = self.moving_var.to(X.device)
                Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
                return Y
            
        # net = nn.Sequential(
        #     nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
        #     nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
        #     nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
        #     nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
        #     nn.Linear(84, 10))
        net=nn.Sequential(nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
            nn.Linear(16*4*4, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
            nn.Linear(84, 10))
        # print(net)
        # X = torch.rand(size=(1, 1, 28, 28))
        # for layer in net:
        #     X = layer(X)
        #     print(layer.__class__.__name__,'output shape:\t',X.shape)

        lr,num_epochs,batch_size = 0.1,10,256

        train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
        d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu())
        print(net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,)))
    '''
## resnet模型结构
    '''bash

        import torch
        from torch import nn
        from d2l import torch as d2l
        from torch.nn import functional as F
        class Residual(nn.Module):
            def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
                super().__init__()
                self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
                self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
                if use_1x1conv:
                    self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
                else:
                    self.conv3 = None
                self.bn1 = nn.BatchNorm2d(num_channels)
                self.bn2 = nn.BatchNorm2d(num_channels)
            def forward(self, X):
                Y = F.relu(self.bn1(self.conv1(X)))
                Y = self.bn2(self.conv2(Y))
                if self.conv3:
                    X = self.conv3(X)
                Y += X
                return F.relu(Y)
        blk = Residual(3,6,use_1x1conv=True)
        X = torch.rand(4, 3, 6, 6)
        Y = blk(X)
        print(Y.shape)
        b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                        nn.BatchNorm2d(64), nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        print(b1)
        X = torch.rand(size=(1, 1, 224, 224))
        for layer in b1:
            X = layer(X)
            print(layer.__class__.__name__,'output shape:\t',X.shape)
        def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
            blk = []
            for i in range(num_residuals):
                if i == 0 and not first_block:
                    blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
                else:
                    blk.append(Residual(num_channels, num_channels))
            return blk
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(512, 10))
        print(net)
        X = torch.rand(size=(1, 1, 224, 224))
        for layer in net:
            X = layer(X)
            print(layer.__class__.__name__,'output shape:\t',X.shape)
    '''
## DenseNet -稠密连接网络
相当于是RESNET的变体，两者的思想其实都是类似于泰勒展开的思想
    '''bash
        
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
        
    '''
## 文本序列的预处理
    '''bash
        import collections
        import re
        from d2l import torch as d2l
        d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                        '090b5e7e70c295757f55df93cb0a180b9691891a')
        def read_time_machine():
            with open(d2l.download('time_machine'), 'r') as f:
                lines = f.readlines()
            return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
        lines = read_time_machine()
        print(lines[0])
        print(lines[10])
        print(len(lines))

        def tokenize(lines, token='word'):
            if token == 'word':
                return [line.split() for line in lines]
            elif token == 'char':
                return [list(line) for line in lines]
            else:
                print('错误：未知词元类型：' + token)
        tokens = tokenize(lines)
        print(tokens[0])
        # for i in range(11):
            # print(tokens[i])   
        class Vocab:  #@save
            """文本词表"""
            def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
                if tokens is None:
                    tokens = []
                if reserved_tokens is None:
                    reserved_tokens = []
                # 按出现频率排序
                counter = count_corpus(tokens)
                # print(counter)
                self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                        reverse=True)
                # 未知词元的索引为0
                self.idx_to_token = ['<unk>'] + reserved_tokens
                self.token_to_idx = {token: idx
                                    for idx, token in enumerate(self.idx_to_token)}
                for token, freq in self._token_freqs:
                    if freq < min_freq:
                        break
                    if token not in self.token_to_idx:
                        self.idx_to_token.append(token)
                        self.token_to_idx[token] = len(self.idx_to_token) - 1

            def __len__(self):
                return len(self.idx_to_token)

            def __getitem__(self, tokens):
                if not isinstance(tokens, (list, tuple)):
                    return self.token_to_idx.get(tokens, self.unk)
                return [self.__getitem__(token) for token in tokens]

            def to_tokens(self, indices):
                if not isinstance(indices, (list, tuple)):
                    return self.idx_to_token[indices]
                return [self.idx_to_token[index] for index in indices]

            @property
            def unk(self):  # 未知词元的索引为0
                return 0

            @property
            def token_freqs(self):
                return self._token_freqs

        def count_corpus(tokens):  #@save
            """统计词元的频率"""
            # 这里的tokens是1D列表或2D列表
            if len(tokens) == 0 or isinstance(tokens[0], list):
                # 将词元列表展平成一个列表
                tokens = [token for line in tokens for token in line]
            return collections.Counter(tokens)
        vocab = Vocab(tokens)
        print(len(vocab))
        # print(list(vocab.token_to_idx.items())[:10])
        # print(len(vocab))
        # print(vocab['hello'])
        for i in [0,10]:
            print('token: ', tokens[i])
            print('index: ', vocab[tokens[i]])

        def load_corpus_time_machine(max_tokens=-1):
            print("--------------------------------")
            lines = read_time_machine()
            print("read_time_machine 读取的行数:", len(lines))
            print("示例原始行[0]:", lines[0])
            print("示例原始行[10]:", lines[10])

            tokens = tokenize(lines, 'char')
            print("tokenize 后行数(应该与行数相同):", len(tokens))
            print("第 0 行被切成的字符列表:", tokens[0])
            print("第 0 行字符个数:", len(tokens[0]))

            line_lengths = [len(line) for line in tokens]
            total_chars = sum(line_lengths)
            print("前 5 行的字符数:", line_lengths[:5])
            print("所有行字符数之和 (total_chars):", total_chars)

            vocab = Vocab(tokens)
            print("字符级词表大小 len(vocab):", len(vocab))
            print("词表中前 20 个 token:", vocab.idx_to_token[:20])

            corpus = [vocab[token] for line in tokens for token in line]
            print("展开后的 corpus 长度 len(corpus):", len(corpus))
            print("len(corpus) 是否等于 total_chars:", len(corpus) == total_chars)
            if max_tokens > 0:
                corpus = corpus[:max_tokens]
            print(len(corpus))
            return corpus, vocab
        corpus, vocab = load_corpus_time_machine()
        print(len(corpus), len(vocab))
    '''
## 循环神经网络RNN-数据预处理
    '''bash

        import collections
        import re
        from d2l import torch as d2l
        d2l.DATA_HUB['time_machine'] = (
            d2l.DATA_URL + 'timemachine.txt',
            '090b5e7e70c295757f55df93cb0a180b9691891a')

        def read_time_machine():
            with open(d2l.download('time_machine'), 'r') as f:
                lines = f.readlines()
            return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
        lines = read_time_machine()
        print(lines[0])
        print(lines[10])
        print(len(lines))

        def tokenize(lines, token='word'):
            if token == 'word':
                return [line.split() for line in lines]
            elif token == 'char':
                return [list(line) for line in lines]
            else:
                print('错误：未知词元类型：' + token)
        tokens = tokenize(lines)
        print(tokens[0])
        # for i in range(11):
            # print(tokens[i])   
        class Vocab:  #@save
            """文本词表"""
            def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
                if tokens is None:
                    tokens = []
                if reserved_tokens is None:
                    reserved_tokens = []
                # 按出现频率排序
                counter = count_corpus(tokens)
                # print(counter)
                self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                        reverse=True)
                # 未知词元的索引为0
                self.idx_to_token = ['<unk>'] + reserved_tokens
                self.token_to_idx = {token: idx
                                    for idx, token in enumerate(self.idx_to_token)}
                for token, freq in self._token_freqs:
                    if freq < min_freq:
                        break
                    if token not in self.token_to_idx:
                        self.idx_to_token.append(token)
                        self.token_to_idx[token] = len(self.idx_to_token) - 1

            def __len__(self):
                return len(self.idx_to_token)

            def __getitem__(self, tokens):
                if not isinstance(tokens, (list, tuple)):
                    return self.token_to_idx.get(tokens, self.unk)
                return [self.__getitem__(token) for token in tokens]

            def to_tokens(self, indices):
                if not isinstance(indices, (list, tuple)):
                    return self.idx_to_token[indices]
                return [self.idx_to_token[index] for index in indices]

            @property
            def unk(self):  # 未知词元的索引为0
                return 0

            @property
            def token_freqs(self):
                return self._token_freqs

        def count_corpus(tokens):  #@save
            """统计词元的频率"""
            # 这里的tokens是1D列表或2D列表
            if len(tokens) == 0 or isinstance(tokens[0], list):
                # 将词元列表展平成一个列表
                tokens = [token for line in tokens for token in line]
            return collections.Counter(tokens)
        vocab = Vocab(tokens)
        print(len(vocab))
        # print(list(vocab.token_to_idx.items())[:10])
        # print(len(vocab))
        # print(vocab['hello'])
        for i in [0,10]:
            print('token: ', tokens[i])
            print('index: ', vocab[tokens[i]])

        def load_corpus_time_machine(max_tokens=-1):
            print("--------------------------------")
            lines = read_time_machine()
            # print("read_time_machine 读取的行数:", len(lines))
            # print("示例原始行[0]:", lines[0])
            # print("示例原始行[10]:", lines[10])

            tokens = tokenize(lines, 'char')
            # print("tokenize 后行数(应该与行数相同):", len(tokens))
            # print("第 0 行被切成的字符列表:", tokens[0])
            # print("第 0 行字符个数:", len(tokens[0]))

            line_lengths = [len(line) for line in tokens]
            total_chars = sum(line_lengths)
            # print("前 5 行的字符数:", line_lengths[:5])
            # print("所有行字符数之和 (total_chars):", total_chars)

            vocab = Vocab(tokens)
            # print("这里是vocab--------------------------------")
            # print(vocab)
            # for token, freq in vocab.token_freqs[:20]:
            #     print(token, freq)
            # print("字符级词表大小 len(vocab):", len(vocab))
            # print("词表中前 20 个 token:", vocab.idx_to_token[:20])

            corpus = [vocab[token] for line in tokens for token in line]
            print("展开后的 corpus 长度 len(corpus):", len(corpus))
            print("len(corpus) 是否等于 total_chars:", len(corpus) == total_chars)
            if max_tokens > 0:
                corpus = corpus[:max_tokens]
            print(len(corpus))
            return corpus, vocab
        # corpus, vocab = load_corpus_time_machine()
        # print(len(corpus), len(vocab))

        import torch
        from torch import nn
        from d2l import torch as d2l
        from torch.nn import functional as F
        import random
        import collections
        # T=1000

        def seq_data_iter_random(corpus, batch_size, num_steps):
            corpus = corpus[random.randint(0, num_steps - 1):]
            print(corpus)
            num_subseqs = (len(corpus) - 1) // num_steps
            print(num_subseqs)
            initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
            print(initial_indices)
            random.shuffle(initial_indices)
            print(initial_indices)
            def data(pos):
                return corpus[pos: pos + num_steps]
            num_batches = num_subseqs // batch_size
            print(num_batches)
            for i in range(0, batch_size * num_batches, batch_size):
                print(i)
                initial_indices_per_batch = initial_indices[i: i + batch_size]
                print(initial_indices_per_batch)
                X = [data(j) for j in initial_indices_per_batch]
                print(X)
                Y = [data(j + 1) for j in initial_indices_per_batch]
                print(Y)
                yield torch.tensor(X), torch.tensor(Y)
        my_seq = list(range(35))
        print(my_seq)
        # for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
        #     print('X: ', X, '\nY:', Y)

        def seq_data_iter_sequential(corpus, batch_size, num_steps):
            offset = random.randint(0, num_steps)
            print(offset)
            num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
            print(num_tokens)
            Xs = torch.tensor(corpus[offset: offset + num_tokens])
            print(Xs)
            Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
            print(Ys)
            Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
            print(Xs)
            print(Ys)
            num_batches = Xs.shape[1] // num_steps
            print(num_batches)
            for i in range(0, num_steps * num_batches, num_steps):
                X = Xs[:, i: i + num_steps]
                print(X)
                Y = Ys[:, i: i + num_steps]
                print(Y)
                yield X, Y
        for X,Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5): 
            print('X: ', X, '\nY:', Y)

        class SeqDataLoader:
            def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
                if use_random_iter:
                    self.data_iter_fn = seq_data_iter_random
                else:
                    self.data_iter_fn = seq_data_iter_sequential
                self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
                self.batch_size, self.num_steps = batch_size, num_steps
            def __iter__(self):
                return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
            def __len__(self):
                return len(self.corpus) // self.batch_size * self.num_steps

        def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000): 

            data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
            return data_iter, data_iter.vocab
        data_iter, vocab = load_data_time_machine(batch_size=2, num_steps=5, use_random_iter=True, max_tokens=10000)
        # for X, Y in data_iter:
        #     print("开始训练--------------------------------")
        #     print('X: ', X, '\nY:', Y)
        #     break
    '''
## 循环神经网络训练的实现
    '''bash
        import math
        from d2l import torch as d2l
        import torch
        from torch import nn
        from torch.nn import functional as F
        batch_size, num_steps = 32, 35
        train_iter, vocab = load_data_time_machine(batch_size, num_steps)

        X=torch.arange(10).reshape((2,5))
        F.one_hot(X.T,28).shape
        print(F.one_hot(X.T,28).shape)

        def get_params(vocab_size, num_hiddens, device):
            num_inputs = num_outputs = vocab_size
            def normal(shape):
                return torch.randn(size=shape, device=device) * 0.01
            W_xh = normal((num_inputs, num_hiddens))
            W_hh = normal((num_hiddens, num_hiddens))
            b_h = torch.zeros(num_hiddens, device=device)
            W_hq = normal((num_hiddens, num_outputs))
            b_q = torch.zeros(num_outputs, device=device)
            params = (W_xh, W_hh, b_h, W_hq, b_q)
            for param in params:
                param.requires_grad_(True)
            return params
        def init_rnn_state(batch_size, num_hiddens, device):
            return (torch.zeros((batch_size, num_hiddens), device=device),)
        def rnn(inputs, state, params):
            W_xh, W_hh, b_h, W_hq, b_q = params
            H, = state
            outputs = []
            for X in inputs:
                H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
                Y = torch.mm(H, W_hq) + b_q
                outputs.append(Y)
            return torch.cat(outputs, dim=0), (H,)
        class RNNModelScratch: #@save
            """从零开始实现的循环神经网络模型"""
            def __init__(self, vocab_size, num_hiddens, device,
                        get_params, init_state, forward_fn):
                self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
                self.params = get_params(vocab_size, num_hiddens, device)
                self.init_state, self.forward_fn = init_state, forward_fn
            def __call__(self, X, state):
                X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
                return self.forward_fn(X, state, self.params)
            def begin_state(self, batch_size, device):
                # 需要同时传入 num_hiddens 和 device
                return self.init_state(batch_size, self.num_hiddens, device)
        num_hiddens = 512
        net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                            init_rnn_state, rnn)
        state = net.begin_state(X.shape[0], d2l.try_gpu())
        Y, new_state = net(X.to(d2l.try_gpu()), state)
        print(Y.shape, len(new_state), new_state[0].shape)

        def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
            """在prefix后面生成新字符"""
            state = net.begin_state(batch_size=1, device=device)
            outputs = [vocab[prefix[0]]]
            get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
            for y in prefix[1:]:  # 预热期
                _, state = net(get_input(), state)
                outputs.append(vocab[y])
            for _ in range(num_preds):  # 预测num_preds步
                y, state = net(get_input(), state)
                outputs.append(int(y.argmax(dim=1).reshape(1)))
            return ''.join(vocab.to_tokens(outputs))
        print(predict_ch8('time traveller', 10, net, vocab, d2l.try_gpu()))

        def grad_clipping(net, theta):  #@save
            """裁剪梯度"""
            if isinstance(net, nn.Module):
                params = [p for p in net.parameters() if p.requires_grad]
            else:
                params = net.params
            norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
            if norm > theta:
                for param in params:
                    param.grad[:] *= theta / norm
        def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
            """训练网络一个迭代周期（定义见第8章）"""
            state, timer = None, d2l.Timer()
            metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
            for X, Y in train_iter:
                if state is None or use_random_iter:
                    # 在第一次迭代或使用随机抽样时初始化state
                    state = net.begin_state(batch_size=X.shape[0], device=device)
                else:
                    if isinstance(net, nn.Module) and not isinstance(state, tuple):
                        # state对于nn.GRU是个张量
                        state.detach_()
                    else:
                        # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                        for s in state:
                            s.detach_()
                y = Y.T.reshape(-1)
                X, y = X.to(device), y.to(device)
                y_hat, state = net(X, state)
                l = loss(y_hat, y.long()).mean()
                if isinstance(updater, torch.optim.Optimizer):
                    updater.zero_grad()
                    l.backward()
                    grad_clipping(net, 1)
                    updater.step()
                else:
                    l.backward()
                    grad_clipping(net, 1)
                    # 因为已经调用了mean函数
                    updater(batch_size=1)
                metric.add(l * y.numel(), y.numel())
            return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
        def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
                    use_random_iter=False):
            """训练模型（定义见第8章）"""
            loss = nn.CrossEntropyLoss()
            animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                                    legend=['train'], xlim=[10, num_epochs])
            # 初始化
            if isinstance(net, nn.Module):
                updater = torch.optim.SGD(net.parameters(), lr)
            else:
                updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
            predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
            # 训练和预测
            for epoch in range(num_epochs):
                ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
                if (epoch + 1) % 10 == 0:
                    print(predict('time traveller'))
                    animator.add(epoch + 1, [ppl])
            print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
            print(predict('time traveller'))
            print(predict('traveller'))
        num_epochs, lr = 500, 1
        train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(), use_random_iter=True)
    '''
## 循环网络的简洁实现
    '''bash

        import torch
        from torch import nn
        from d2l import torch as d2l
        from torch.nn import functional as F
        import random
        import collections
        batch_size, num_steps = 32, 35
        train_iter, vocab = load_data_time_machine(batch_size, num_steps)
        num_hiddens = 256
        rnn_layer = nn.RNN(len(vocab), num_hiddens)
        state = torch.zeros((1, batch_size, num_hiddens))
        X = torch.rand(size=(num_steps, batch_size, len(vocab)))
        Y, state_new = rnn_layer(X, state)
        print(Y.shape, state_new.shape)
        class RNNModel(nn.Module):
            def __init__(self, rnn_layer, vocab_size, **kwargs):
                super(RNNModel, self).__init__(**kwargs)
                self.rnn = rnn_layer
                self.vocab_size = vocab_size
                self.num_hiddens = self.rnn.hidden_size
                self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
            def forward(self, inputs, state):
                X = F.one_hot(inputs.T.long(), self.vocab_size)
                X = X.to(torch.float32)
                Y, state = self.rnn(X, state)
                output = self.linear(Y.reshape((-1, Y.shape[-1])))
                return output, state
            def begin_state(self, device, batch_size=1):
                if not isinstance(self.rnn, nn.LSTM):
                    return torch.zeros((self.num_directions * self.rnn.num_layers,
                                        batch_size, self.num_hiddens), device=device)
                else:
                    return (torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                                self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens), device=device))
        net = RNNModel(rnn_layer, vocab_size=len(vocab))
        print(net)
    '''
## GRU 的实现
    '''bash
        import torch
        from torch import nn
        from d2l import torch as d2l
        batch_size, num_steps = 32, 35
        train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

        def get_params(vocab_size, num_hiddens, device):
            num_inputs = num_outputs = vocab_size
            def normal(shape):
                return torch.randn(size=shape, device=device) * 0.01
            def three():
                return (normal((num_inputs, num_hiddens)),
                        normal((num_hiddens, num_hiddens)),
                        torch.zeros(num_hiddens, device=device))
            W_xz, W_hz, b_z = three()  # 更新门参数
            W_xr, W_hr, b_r = three()  # 重置门参数
            W_xh, W_hh, b_h = three()  # 候选隐状态参数
            # 输出层参数
            W_hq = normal((num_hiddens, num_outputs))
            b_q = torch.zeros(num_outputs, device=device)
            # 附加梯度
            params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
            for param in params:
                param.requires_grad_(True)
            return params
        def init_gru_state(batch_size, num_hiddens, device):
            return (torch.zeros((batch_size, num_hiddens), device=device),)
        # def gru(inputs, state, params):
        #     W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
        #     H, = state
        #     outputs = []
        #     for X in inputs:
        #         Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        #         R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        #         H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        #         H = Z * H + (1 - Z) * H_tilda
        #         Y = H @ W_hq + b_q
        #         outputs.append(Y)
        #     return torch.cat(outputs, dim=0), (H,)
        gru_layer = nn.GRU(len(vocab), 256)

        vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
        num_epochs, lr = 500, 1
        # model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
        #                             init_gru_state, gru)
        model = d2l.RNNModel(gru_layer, len(vocab))
        d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    '''
## LSTM的实现
这个与上面GRU的区别是变得更复杂了一些，GRU是在这个基础上发展出来的
    '''bash
        import torch
        from torch import nn
        from d2l import torch as d2l

        batch_size, num_steps = 32, 35
        train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
        def get_lstm_params(vocab_size, num_hiddens, device):
            num_inputs = num_outputs = vocab_size
            def normal(shape):
                return torch.randn(size=shape, device=device) * 0.01
            def three():
                return (normal((num_inputs, num_hiddens)),
                        normal((num_hiddens, num_hiddens)),
                        torch.zeros(num_hiddens, device=device))
            W_xi, W_hi, b_i = three()  # 输入门参数
            W_xf, W_hf, b_f = three()  # 遗忘门参数
            W_xo, W_ho, b_o = three()  # 输出门参数
            W_xc, W_hc, b_c = three()  # 候选记忆元参数 
            # 输出层参数
            W_hq = normal((num_hiddens, num_outputs))
            b_q = torch.zeros(num_outputs, device=device)
            # 附加梯度
            params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
                    b_c, W_hq, b_q]
            for param in params:
                param.requires_grad_(True)
            return params
        def init_lstm_state(batch_size, num_hiddens, device):
            return (torch.zeros((batch_size, num_hiddens), device=device),
                    torch.zeros((batch_size, num_hiddens), device=device))
        def lstm(inputs, state, params):

            [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
                    b_c, W_hq, b_q] = params
            (H, C) = state
            outputs = []
            for X in inputs:
                I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
                F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
                O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
                C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
                C = F * C + I * C_tilda
                H = O * torch.tanh(C)
                Y = (H @ W_hq) + b_q
                outputs.append(Y)
            return torch.cat(outputs, dim=0), (H, C)
        vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
        num_epochs, lr = 500, 1
        model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                                    init_lstm_state, lstm)
        d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    '''
## 深层循环神经网络
这个与上面的区别也就是我们增加了隐藏层的层数，这个我们直接使用高级API模式情况下直接指定层的数目就可以得到了
    '''bash
        import torch
        from torch import nn
        from d2l import torch as d2l
        batch_size, num_steps = 32, 35
        train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
        vocab_size,num_hiddens,num_layers = len(vocab),256,2
        num_inputs = vocab_size
        device = d2l.try_gpu()
        lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
        model = d2l.RNNModel(lstm_layer, len(vocab))
        model = model.to(device)
        num_epochs, lr = 500, 2
        d2l.train_ch6(model, train_iter, vocab, lr, num_epochs, device)
    '''
## 双向循环神经网络
这个实际上就是把我们的输入按照正常的顺序训练一遍之后再将输入进行颠倒之后训练得到的结果也要颠倒之后就是正常的结果，这个不能用于预测，这个很大情况下最适合做完形填空或者是考察对句子的理解,实现起来与前面的区别也就是添加了一个开关而已
    '''bash
        import torch
        from torch import nn
        from d2l import torch as d2l
        batch_size, num_steps = 32, 35
        train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
        vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
        num_inputs = vocab_size
        device = d2l.try_gpu()
        lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
        model = d2l.RNNModel(lstm_layer, len(vocab))
        model = model.to(device)
        num_epochs, lr = 500, 1
        d2l.train_ch6(model, train_iter, vocab, lr, num_epochs, device)
    '''
## 机器翻译数据集的构建
    '''bash

        import os
        import torch
        from d2l import torch as d2l
        d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                                '94646ad1522d91a60194c3d40b4fd879da9b4e21')
        def read_data_nmt():
            """载入“英语-法语”数据集"""
            data_dir = d2l.download_extract('fra-eng')
            with open(os.path.join(data_dir, 'fra.txt'), 'r',encoding='utf-8') as f:
                return f.read()
        raw_text = read_data_nmt()
        print(raw_text[:75])
        def preprocess_nmt(text):
            """预处理“英语-法语”数据集"""
            def no_space(char, prev_char):
                return char in set(',.!?') and prev_char != ' '
            text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
            out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
                for i, char in enumerate(text)]
            return ''.join(out)
        text = preprocess_nmt(raw_text)
        print(text[:80])
        def tokenize_nmt(text, num_examples=600):
            """词元化“英语-法语”数据数据集"""
            source, target = [], []
            for i, line in enumerate(text.split('\n')):
                if i >= num_examples:
                    break
                parts = line.split('\t')
                if len(parts) == 2:
                    source.append(parts[0].split(' '))
                    target.append(parts[1].split(' '))
            return source, target
        source, target = tokenize_nmt(text)
        print(source[:6])
        print(target[:6])

        def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
            """绘制列表长度对的直方图"""
            d2l.set_figsize()
            _, _, patches = d2l.plt.hist(
                [[len(l) for l in xlist], [len(l) for l in ylist]])
            d2l.plt.xlabel(xlabel)
            d2l.plt.ylabel(ylabel)
            for patch in patches[1].patches:
                patch.set_hatch('/')
            d2l.plt.legend(legend)
        show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                                'count', source, target)
        src_vocab = d2l.Vocab(source, min_freq=2,
                            reserved_tokens=['<pad>', '<bos>', '<eos>'])
        def truncate_pad(line, num_steps, padding_token):
            """截断或填充文本序列"""
            if len(line) > num_steps:
                return line[:num_steps]  # 截断
            return line + [padding_token] * (num_steps - len(line))  # 填充
        truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
        print(truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))
        def build_array_nmt(lines, vocab, num_steps):
            """将机器翻译的文本序列转换成小批量"""
            lines = [vocab[line] for line in lines]
            lines = [line + [vocab['<eos>']] for line in lines]
            array = torch.tensor([truncate_pad(line, num_steps, vocab['<pad>']) for line in lines])
            valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
            return array, valid_len
        def load_data_nmt(batch_size, num_steps, num_examples=600):
            """返回翻译数据集的迭代器和词表"""
            text = preprocess_nmt(read_data_nmt())
            source, target = tokenize_nmt(text, num_examples)
            src_vocab = d2l.Vocab(source, min_freq=2,
                                reserved_tokens=['<pad>', '<bos>', '<eos>'])
            tgt_vocab = d2l.Vocab(target, min_freq=2,
                                reserved_tokens=['<pad>', '<bos>', '<eos>'])
            src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
            tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
            data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
            data_iter = d2l.load_array(data_arrays, batch_size)
            return data_iter, src_vocab, tgt_vocab
        train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
        for X, X_valid_len, Y, Y_valid_len in train_iter:
            print('X:', X.type(torch.int32))
            print('X的有效长度:', X_valid_len)
            print('Y:', Y.type(torch.int32))
            print('Y的有效长度:', Y_valid_len)
            break
    '''
## 编码器解码器结构
    '''bash

        import collections
        import math
        from torch import nn
        from d2l import torch as d2l
        import torch
        class EncoderDecoder(nn.Module):
            """编码器-解码器架构的基类"""
            def __init__(self, encoder, decoder, **kwargs):
                super(EncoderDecoder, self).__init__(**kwargs)
                self.encoder = encoder
                self.decoder = decoder

            def forward(self, enc_X, dec_X, *args):
                enc_outputs = self.encoder(enc_X, *args)
                dec_state = self.decoder.init_state(enc_outputs, *args)
                return self.decoder(dec_X, dec_state)
        class Seq2SeqEncoder(d2l.Encoder):
            """用于序列到序列学习的循环神经网络编码器"""
            def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                        dropout=0, **kwargs):
                super(Seq2SeqEncoder, self).__init__(**kwargs)
                # 嵌入层
                self.embedding = nn.Embedding(vocab_size, embed_size)
                self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                                dropout=dropout)
            def forward(self, X, *args):
                # 输出'X'的形状：(batch_size,num_steps,embed_size)
                print('X.shape:',X.shape)
                X = self.embedding(X)
                print('Xemdedding之后的形状.shape:',X.shape)
                # 在循环神经网络模型中，第一个轴对应于时间步
                X = X.permute(1, 0, 2)
                print('X.shape:',X.shape)
                # 如果未提及状态，则默认为0
                output, state = self.rnn(X)
                print('output.shape:',output.shape)
                print('state.shape:',state.shape)
                # output的形状:(num_steps,batch_size,num_hiddens)
                # state的形状:(num_layers,batch_size,num_hiddens)
                return output, state
        # encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
        # encoder.eval()
        # X=torch.zeros((4,7),dtype=torch.long)
        # output, state = encoder(X)
        # print(output.shape, len(state))
        # for X,  Y in zip(output, state):
        #     print(X.shape, Y.shape)
        #     print(X)
        #     print(Y)
        class Seq2SeqDecoder(d2l.Decoder):
            """用于序列到序列学习的循环神经网络解码器"""
            def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                        dropout=0, **kwargs):
                super(Seq2SeqDecoder, self).__init__(**kwargs)
                self.embedding = nn.Embedding(vocab_size, embed_size)
                self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                                dropout=dropout)
                self.dense = nn.Linear(num_hiddens, vocab_size)
            def init_state(self, enc_outputs, *args):
                return enc_outputs[1]

            def forward(self, X, state):
                # X: (batch_size, num_steps) 的词元索引
                # 嵌入后形状: (batch_size, num_steps, embed_size)
                print('X.shape:',X.shape)
                X = self.embedding(X)
                print('Xembedding之后的形状.shape:',X.shape)
                # 变换为以时间步为第一维: (num_steps, batch_size, embed_size)
                X = X.permute(1, 0, 2)
                print('X.shape:',X.shape)
                # 取编码器最后一层的隐状态, 作为上下文
                # state 形状: (num_layers, batch_size, num_hiddens)
                # context 形状: (num_steps, batch_size, num_hiddens)
                context = state[-1].repeat(X.shape[0], 1, 1)
                print('context.shape:',context.shape)
                # 在特征维上拼接上下文: (num_steps, batch_size, embed_size + num_hiddens)
                X_and_context = torch.cat((X, context), 2)
                print('X_and_context.shape:',X_and_context.shape)
                output, state = self.rnn(X_and_context, state)
                print('output.shape:',output.shape)
                print('state.shape:',state.shape)
                # output: (num_steps, batch_size, num_hiddens) -> (batch_size, num_steps, num_hiddens)
                output = self.dense(output).permute(1, 0, 2)
                print('output.shape:',output.shape)
                return output, state
        # decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
        # decoder.eval()
        # # 确保传入编码器和解码器的都是长整型索引
        # state = decoder.init_state(encoder(X.long()))
        # output, state = decoder(X.long(), state)
        # print(output.shape, len(state), state[0].shape)
        def sequence_mask(X, valid_len, value=0):
            """在序列中屏蔽不相关的项"""
            maxlen = X.size(1)
            mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
            X[~mask] = value
            return X
        # X = torch.tensor([[1, 2, 3], [4, 5, 6]])
        # print(sequence_mask(X, torch.tensor([1, 2])))
        def masked_softmax(X, valid_lens):
            """通过在最后⼀个轴上遮蔽元素来执⾏ softmax 操作"""
            if valid_lens is None:
                return nn.functional.softmax(X, dim=-1)
        class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
            """带遮蔽的softmax交叉熵损失函数"""
            # pred的形状：(batch_size,num_steps,vocab_size)
            # label的形状：(batch_size,num_steps)
            # valid_len的形状：(batch_size,)
            def forward(self, pred, label, valid_len):
                weights = torch.ones_like(label)
                weights = sequence_mask(weights, valid_len)
                self.reduction='none'
                unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
                    pred.permute(0, 2, 1), label)
                weighted_loss = (unweighted_loss * weights).mean(dim=1)
                return weighted_loss
        # loss = MaskedSoftmaxCELoss()
        def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
            """训练序列到序列模型"""
            def xavier_init_weights(m):
                if type(m) == nn.Linear:
                    nn.init.xavier_uniform_(m.weight)
                if type(m) == nn.GRU:
                    for param in m._flat_weights_names:
                        if "weight" in param:
                            nn.init.xavier_uniform_(m._parameters[param])
            net.apply(xavier_init_weights)
            net.to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            loss = MaskedSoftmaxCELoss()
            net.train()
            animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
            for epoch in range(num_epochs):
                timer = d2l.Timer()
                metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
                for batch in data_iter:
                    optimizer.zero_grad()
                    X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
                    bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                                device=device).reshape(-1, 1)
                    dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学  
                    # 兼容 EncoderDecoder 的返回值格式，只取预测结果
                    outputs = net(X, dec_input, X_valid_len)
                    Y_hat = outputs[0] if isinstance(outputs, tuple) else outputs
                    l = loss(Y_hat, Y, Y_valid_len)
                    l.sum().backward()	# 损失函数的标量进行“反向传播”
                    d2l.grad_clipping(net, 1)
                    num_tokens = Y_valid_len.sum()
                    optimizer.step()
                    with torch.no_grad():
                        metric.add(l.sum(), num_tokens)
                if (epoch + 1) % 10 == 0:
                    animator.add(epoch + 1, (metric[0] / metric[1],))
            print(f"loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} "
                f"tokens/sec on {str(device)}")
        embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
        batch_size, num_steps = 64, 10
        lr, num_epochs, device = 0.005, 300, d2l.try_gpu()
        train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
        encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                                dropout)
        decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                                dropout)
        net = d2l.EncoderDecoder(encoder, decoder)
        train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

        def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                            device, save_attention_weights=False):
            """序列到序列模型的预测"""
            # 在预测时将net设置为评估模式
            net.eval()
            src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
                src_vocab['<eos>']]
            enc_valid_len = torch.tensor([len(src_tokens)], device=device)
            src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
            # 添加批量轴
            enc_X = torch.unsqueeze(
                torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
            enc_outputs = net.encoder(enc_X, enc_valid_len)
            dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
            # 添加批量轴
            dec_X = torch.unsqueeze(torch.tensor(
                [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
            output_seq, attention_weight_seq = [], []
            for _ in range(num_steps):
                Y, dec_state = net.decoder(dec_X, dec_state)
                # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
                dec_X = Y.argmax(dim=2)
                pred = dec_X.squeeze(dim=0).type(torch.int32).item()
                # 保存注意力权重（稍后讨论）
                if save_attention_weights:
                    attention_weight_seq.append(net.decoder.attention_weights)
                # 一旦序列结束词元被预测，输出序列的生成就完成了
                if pred == tgt_vocab['<eos>']:
                    break
                output_seq.append(pred)
            return ' '.join(tgt_vocab.to_tokens(output_seq)),attention_weight_seq
    '''
## 注意力机制的实现
简单来说实际上注意力机制就是我们使权重有权重的关注我们想要的对应的键值对（就是训练数据组成的实际上）然后这个过程你可以用非参数学习的关注（softmax实现），也可以使用有参数的进行学习来关注这样的话好处就是结果可能更加符合我们的训练时候的键值对但是不一定真的符合我们的曲线可能会有尖端或者是不平滑现象的产生
    '''
        import torch
        from torch import nn
        from d2l import torch as d2l
        n_train = 50  # 训练样本数
        x_train,_=torch.sort(torch.rand(n_train)*5)
        def f(x):
            return 2*torch.sin(x)+x**0.8
        y_train=f(x_train)+torch.normal(0,0.5,(n_train,))
        x_test=torch.arange(0,5,0.1)
        y_truth=f(x_test)
        n_test=len(x_test)
        print(n_test)
        def plot_kernel_reg(y_hat):
            d2l.plot(x_test,[y_truth,y_hat],'x','y',legend=['Truth','Pred'],xlim=[0,5],ylim=[-1,5])
            d2l.plt.plot(x_train,y_train,'o',alpha=0.5)
        y_hat=torch.repeat_interleave(y_train.mean(),n_test)
        plot_kernel_reg(y_hat)
        X_repeat=x_test.repeat_interleave(n_train).reshape((-1,n_train))
        attention_weights=nn.functional.softmax(-(X_repeat-x_train)**2/2,dim=1)
        y_hat=torch.matmul(attention_weights,y_train)
        plot_kernel_reg(y_hat)
        class NWKernelRegression(nn.Module):
            def __init__(self,**kwargs):
                super().__init__(**kwargs)
                self.w=nn.Parameter(torch.rand((1,),requires_grad=True))
            def forward(self,queries,keys,values):
                queries=queries.repeat_interleave(keys.shape[1]).reshape((-1,keys.shape[1]))
                self.attention_weights=nn.functional.softmax(-((queries-keys)*self.w)**2/2,dim=1)
                return torch.bmm(self.attention_weights.unsqueeze(1),values.unsqueeze(-1)).reshape(-1)
        x_tile=x_train.repeat((n_train,1))
        y_tile=y_train.repeat((n_train,1))
        keys=x_tile[(1-torch.eye(n_train)).type(torch.bool)].reshape((n_train,-1))
        values=y_tile[(1-torch.eye(n_train)).type(torch.bool)].reshape((n_train,-1))
        net=NWKernelRegression()
        loss=nn.MSELoss(reduction='none')
        trainer=torch.optim.SGD(net.parameters(),lr=0.5)
        animator=d2l.Animator(xlabel='epoch',ylabel='loss',xlim=[1,5])
        for epoch in range(5):
            trainer.zero_grad()
            l=loss(net(x_test,keys,values),y_truth)
            l.sum().backward()
            trainer.step()
            print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
            animator.add(epoch + 1, float(l.sum()))
        plot_kernel_reg(net(x_test, keys, values).unsqueeze(1).reshape(-1))
        d2l.plt.show()
    '''
## 评分函数
这里介绍了两个评分函数一个是加性的评分，一个是点积性的评分，加性的是指当你的查询和键值的维度不一样的时候，或者是含义不同的时候使用我们这里给他们了可以学习的参数，所以相当于通过了一个MLP进行的评分，但是这里的点积形式的只能处理维度一样的，我们在前期可能会通过学习的方式将每一个键值对换成向量矩阵的形式这里的点击就是没有学习的参数纯进行的数学运算，根据矩阵的（向量的）乘法来进行的

    '''bash
        class AdditiveAttention(nn.Module):
            def __init__(self,key_size,query_size,num_hiddens,dropout,**kwargs):
                super().__init__(**kwargs)
                self.W_k=nn.Linear(key_size,num_hiddens,bias=False)
                self.W_q=nn.Linear(query_size,num_hiddens,bias=False)
                self.w_v=nn.Linear(num_hiddens,1,bias=False)
                self.dropout=nn.Dropout(dropout)
            def forward(self,queries,keys,values,valid_lens):
                queries=self.W_q(queries)
                keys=self.W_k(keys)
                features=queries.unsqueeze(2)+keys.unsqueeze(1)
                features=torch.tanh(features)
        class DotProductAttention(nn.Module):
            def __init__(self,dropout,**kwargs):
                super().__init__(**kwargs)
                self.dropout=nn.Dropout(dropout)
            def forward(self,queries,keys,values,valid_lens=None):
                d=queries.shape[-1]
                scores=torch.bmm(queries,keys.transpose(1,2))/math.sqrt(d)
                self.attention_weights=masked_softmax(scores,valid_lens)
                return torch.bmm(self.dropout(self.attention_weights),values)
        queries=torch.normal(0,1,(2,1,20))
        keys=torch.normal(0,1,(2,10,2))
        values=torch.normal(0,1,(2,10,4))
    '''
## 带有线性加权注意力机制的seqtoseq翻译模型
1. 首先有一点你需要知道就是翻译不是一对一的因为不同语言的语法是不一样的，所以相当于翻译就是知道你原句子的大概意思之后进行预测的工作
2. 现在这个下面的不同之处在decode，原本我们的decode工作是把encode输出的最后一层hidden_state与当前的x（这个是指正确的上一个预测的值，因为我们不可能真的使用我们预测的作为输入，因为刚开始肯定有偏差这样的话偏差会越来越大的，所以一般情况下我们使用的都是正确答案作为训练）一起拼在一起作为输入进行的预测，之后输出预测结果的
3. 现在这个不同的时候引入了一个注意力的机制，这个注意力机制的输入query是上一个时刻encode输出的最后一层的隐藏层也就是记录了当前所有的记忆的那一个层，之后key和value指encode在历史的每一次预测结束之后的所有的隐藏层，之后经过注意力机制之后相当于给之前的所有的记忆进行了一个加权求和的操作之后再与x拼接起来之后才输入进decode的最后一层输出预测层
    '''bash
        class Seq2SeqAttentionDecoder(AttentionDecoder):
            def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                        dropout=0, **kwargs):
                super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
                self.attention = d2l.AdditiveAttention(
                    num_hiddens, num_hiddens, num_hiddens, dropout)
                self.embedding = nn.Embedding(vocab_size, embed_size)
                self.rnn = nn.GRU(
                    embed_size + num_hiddens, num_hiddens, num_layers,
                    dropout=dropout)
                self.dense = nn.Linear(num_hiddens, vocab_size)

            def init_state(self, enc_outputs, enc_valid_lens, *args):
                # outputs的形状为(batch_size，num_steps，num_hiddens).
                # hidden_state的形状为(num_layers，batch_size，num_hiddens)
                outputs, hidden_state = enc_outputs
                return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

            def forward(self, X, state):
                # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
                # hidden_state的形状为(num_layers,batch_size,
                # num_hiddens)
                enc_outputs, hidden_state, enc_valid_lens = state
                # 输出X的形状为(num_steps,batch_size,embed_size)
                X = self.embedding(X).permute(1, 0, 2)
                outputs, self._attention_weights = [], []
                for x in X:
                    # query的形状为(batch_size,1,num_hiddens)
                    query = torch.unsqueeze(hidden_state[-1], dim=1)
                    # context的形状为(batch_size,1,num_hiddens)
                    context = self.attention(
                        query, enc_outputs, enc_outputs, enc_valid_lens)
                    # 在特征维度上连结
                    x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
                    # 将x变形为(1,batch_size,embed_size+num_hiddens)
                    out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
                    outputs.append(out)
                    self._attention_weights.append(self.attention.attention_weights)
                # 全连接层变换后，outputs的形状为
                # (num_steps,batch_size,vocab_size)
                outputs = self.dense(torch.cat(outputs, dim=0))
                return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                                enc_valid_lens]

            @property
            def attention_weights(self):
                return self._attention_weights
    '''
## 自注意力机制
实际上就是上面的注意力机制当quary\key\value都是一样的就代表是自注意力机制了
## 多头注意力机制
就是并行开了几个单注意力机制让他去学不同的特征在得到输出之后再结合起来再进入到下面就是了
## 梯度下降优化算法
'''bash
一、 基础族：走稳每一步
11.3 梯度下降 (Gradient Descent, GD)

核心逻辑：计算全量数据集的梯度，沿着梯度负方向更新参数。

解决的问题：提供了最基本的优化方向。

优点：方向最精准，如果损失函数是凸的，它一定能找到全局最优。

缺点：太慢了！ 如果你有 1 亿条数据，更新一次参数就要跑完 1 亿条，内存和时间都吃不消。

11.4 随机梯度下降 (Stochastic GD, SGD)

核心逻辑：每次只随机挑一个样本计算梯度。

解决的问题：解决了 GD 训练速度慢、内存压力大的问题。

优点：计算极快；由于引入了随机性（噪声），它更容易跳出局部最小值。

缺点：太“抖”了。更新方向极其不稳定，收敛过程像喝醉了酒一样左右摇摆，很难真正到达最低点。

11.5 小批量随机梯度下降 (Mini-batch SGD)

核心逻辑：每次挑一小块数据（如 32、64 个样本）算梯度。

解决的问题：平衡了 GD 的精准和 SGD 的速度。

优点：利用了矩阵运算的加速，既比全量快，又比单个样本稳。这是目前最常用的基础优化方式。

缺点：学习率（Learning Rate）极难调。设大了震荡，设小了跑不动。

二、 加速族：带上惯性
11.6 动量法 (Momentum)

核心逻辑：不仅看当下的梯度，还加上“过去的惯性”（动量）。

解决的问题：解决了 SGD 在“峡谷”地形中左右震荡、前进缓慢的问题。

优点：在平坦的地方能加速（像球越滚越快），在震荡的地方能通过惯性抵消噪音。

缺点：需要额外调一个超参数（动量因子），且可能因为惯性太大冲过头。

三、 自适应族：给每个参数配个“变速箱”

这一族算法的共同特点是：不再全局共用一个学习率，而是为每一个参数自动调整步长。

11.7 AdaGrad

核心逻辑：记录每个参数的历史梯度平方和。梯度越大，学习率衰减越快。

解决的问题：解决了稀疏数据中，低频特征更新不足的问题。

优点：前期收敛快，无需手动调学习率。

缺点：后期跑不动。 随着训练进行，梯度累积越来越大，学习率会无限趋近于 0，导致模型还没训练好就“提前收工”。

11.8 RMSProp

核心逻辑：对梯度平方和做了“移动平均”（只看最近一段时间的梯度）。

解决的问题：解决了 AdaGrad 后期学习率消失的问题。

优点：克服了 AdaGrad 的早停问题，在循环神经网络（RNN）中表现极佳。

缺点：依然需要手动设定全局初始学习率。

11.10 Adam (Adaptive Moment Estimation)

核心逻辑：Momentum + RMSProp 的合体。既考虑了惯性（一阶动量），又考虑了自适应步长（二阶动量）。

解决的问题：几乎解决了上述所有常见的优化痛点。

优点：“懒人神器”。非常鲁棒，通常不需要怎么调参就能跑出不错的结果，是目前深度学习最主流的优化器。

缺点：在某些视觉任务中，最后的泛化能力可能略逊于精调后的 SGD；由于要存动量，显存占用比 SGD 高。
将深度学习优化算法的演进逻辑清晰地划分为三个维度：首先是以 Momentum 为代表的梯度更新，通过引入动量机制优化“方向”，起到消除震荡、加速收敛的作用；其次是以 AdaGrad 和 RMSProp 为代表的学习率更新，利用自适应缩放优化“步长”，有效解决了不同参数间尺度不一的痛点并实现了自动调参；而以 Adam 为代表的全方位更新则将一阶与二阶动量相结合，完美兼顾了运动惯性与精准调速，是目前深层模型训练中性能最均衡、应用最广泛的主流优化方案。
'''
## 基于锚框的目标检测的发展
1. R-CNN：这个实际上就是在一个图片上面生成锚框之后经过一个卷积去抽取特征之后，不加处理直接丢进线性层去预测每一个锚框分类的精度以及与标准框的偏移
2. Fast-RCNN：这个跟上面的区别就是他认为上面那个锚框太多了这样的话计算量会非常的大，所以他进行的操作就是你先不提取锚框，你先对图片进行ROI降维，之后对原始图片上面提取锚框之后，将这个锚框按照线性变换变换到你经过ROI之后的尺寸上，之后再对这个变换之后的锚框的内容进行丢入线性层进行分类以及边框偏移预测
3. Faster-RCNN：这个呢实际上就是他还是觉得你上面那个不够快，他进行的操作就是他提出了一个双阶段，就是首先你还是对原始图像进行卷积抽特征，之后你进行锚框选取的时候实际上训练了一个小网络进行对选择的锚框的分类精度以及偏移进行打分之后，经过NMS再将这个的输出与刚才对原始图片抽取特征之后的结果叠上去，再经过一个大网络之后进行锚框中的分类以及边框偏移的预测
4. SSD则是单阶段他直接是在像素上生成尺寸大小不同的锚框之后进行的预测，只不过他这里锚框数量有一个限制并没有那么多，之后分类的预测以及偏移的预测都是通过卷积得到的，在通道上面反映出来的同一个像素不同锚框的效果打分，而且他是对多个尺度进行的目标检测就是经过不同的卷积层之后都会接一个检测的头进行检测
5. 而YOLO呢就是前期他锚框的生成是直接基于你的图像进行切割的，然后每一个锚框会预测多个边缘框是这个样子的
## 转置卷积
这个一般用在分割模型里面，因为分割模型是基于像素的，但是你每次经过卷积之后图像都会变小，这个不利于我们之后的分割因为我们是基于像素的，所以现在需要一种转置卷积他的目的是将经过卷积之后图像的大小会变大，因为如果用矩阵乘法实现卷积和转置卷积的话两个用矩阵表达出来的形式是互为转置的所以称为是转置卷积