import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):

    def __init__(self, cost_class: float = 1, cost_reg: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_reg: This is the relative weight of the L1 error of the coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_reg = cost_reg
        # assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, 1] with the classification logits
                 "pred_regs": Tensor of dim [batch_size, num_queries, 3] with the pred logits
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_regs"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = targets['labels']
        tgt_regression = targets['regression']

        # Compute the L1 cost between regression
        cost_reg = torch.cdist(out_bbox, tgt_regression, p=1)

        # Final cost matrix
        C = self.cost_reg * cost_reg + self.cost_class
        C = C.view(bs, num_queries, -1).cpu()

        # print("--c shape", C.shape)
        # print("---t shape", targets['regression'].shape)
        sizes = [5] * targets['regression'].shape[0]

        # print("sizes", sizes, "csplit", C.split(sizes, -1))
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]