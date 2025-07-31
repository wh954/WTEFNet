import torch.nn as nn


class Select(nn.Module):
    def __init__(self, idx, *args, **kwargs):
        super().__init__()
        self.idx = idx
    def forward(self, x):
        if isinstance(x, (tuple, list)):
            if self.idx >= len(x):
                # print(f"[WARNING] Select.idx={self.idx} 超出范围（实际len={len(x)}），返回第0个特征！")
                return x[0]
            return x[self.idx]
        else:
            print(f"Select.forward got {type(x)} {x.shape if hasattr(x,'shape') else None} self.idx: {self.idx}")
            return x







