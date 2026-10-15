import os
import random
import numpy as np
import torch


def setup_seed(seed=42):
    """
    固定所有可能的随机种子，确保实验结果可复现
    """
    # 1. Python 原生随机种子
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. Numpy 随机种子
    np.random.seed(seed)

    # 3. PyTorch CPU 随机种子
    torch.manual_seed(seed)

    # 4. PyTorch GPU 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多张显卡

    # 5. CuDNN 确定性算法 (关键)
    # benchmark = True 会让 CuDNN 自动寻找最适合当前硬件的算法，这会导致结果不确定
    # deterministic = True 强制使用确定性算法，虽然可能稍慢一点，但为了复现是必须的
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(f"[Info] Random seed set to: {seed}")


def create_lr_scheduler(optimizer, num_step: int, epochs: int, warmup=True, warmup_epochs=1, warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
