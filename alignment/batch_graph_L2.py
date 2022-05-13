import torch
import torch.nn as nn


# TODO more documentation here
class BatchGraphL2(nn.Module):
    def __init__(self):
        super(BatchGraphL2, self).__init__()

    def forward(self, flow_gt, flow_pred, valid_solve, deformations_validity):
        batch_size = flow_gt.shape[0]

        assert flow_gt.shape[2] == 3
        assert flow_pred.shape[2] == 3

        assert torch.isfinite(flow_gt).all(), flow_gt

        diff = flow_pred - flow_gt
        diff_squared = diff * diff

        deformations_mask = deformations_validity.type(torch.float32)
        deformations_mask = deformations_mask.view(batch_size, -1, 1).repeat(1, 1, 3)

        diff2_masked = deformations_mask * diff_squared

        loss = torch.zeros(batch_size, dtype=diff_squared.dtype, device=diff_squared.device)
        mask = []
        for i in range(batch_size):
            num_valid_nodes = deformations_validity[i].sum()

            if valid_solve[i] and num_valid_nodes > 0:
                loss[i] = torch.sum(diff2_masked[i]) / num_valid_nodes
                mask.append(i)

        assert torch.isfinite(loss).all()

        if len(mask) == 0:
            return torch.zeros((1), dtype=diff_squared.dtype, device=flow_gt.device)
        else:
            loss = loss[mask]
            return torch.sum(loss) / len(mask)
