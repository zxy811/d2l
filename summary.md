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
