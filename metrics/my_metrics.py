import torch
from torchmetrics import Metric


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, target): # 直接接收 preds 和 target
        preds, target = (
            preds.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )

        if preds.ndim != 1 or target.ndim != 1 or preds.shape != target.shape:
            # 如果形状不匹配，打印警告并尝试继续（或直接返回以避免错误）
            print(f"Warning/Error: Unexpected shapes in Accuracy.update. preds: {preds.shape}, target: {target.shape}. Ensure inputs are filtered 1D tensors.")
            # 尝试去除多余维度（如果适用）
            try:
                preds = preds.squeeze()
                target = target.squeeze()
                if preds.ndim != 1 or target.ndim != 1 or preds.shape != target.shape:
                    print(f"Error: Cannot reconcile shapes after squeeze. Returning.")
                    return
            except Exception as e:
                 print(f"Error: Failed to reconcile shapes ({e}). Returning.")
                 return

        if target.numel() == 0:
            return
        assert preds.shape == target.shape, f"Shape mismatch after potential squeeze: preds {preds.shape}, target {target.shape}"

        self.correct += torch.sum(preds == target)
        self.total += target.numel()


    def compute(self):
        if self.total == 0:
            return torch.tensor(0.0, device=self.correct.device, dtype=torch.float)
        return self.correct.float() / self.total


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        # Add check for zero total
        if self.total == 0:
            # Return tensor(0.0) on the correct device
            return torch.tensor(0.0, device=self.scalar.device, dtype=torch.float)
        return self.scalar.float() / self.total


class VQAScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros_like(target)
        one_hots.scatter_(1, logits.view(-1, 1).long(), 1)
        scores = one_hots * target

        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        if self.total == 0:
            # Return tensor(0.0) on the correct device
            return torch.tensor(0.0, device=self.score.device, dtype=torch.float)
        # score and total should be compatible for division
        return self.score.float() / self.total
