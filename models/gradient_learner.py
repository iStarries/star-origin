import torch
import torch.nn as nn


class GradientLearner(nn.Module):
    """A lightweight MLP that predicts per-logit gradients.

    The module keeps the parameter count small so it can be trained alongside
    the main segmentation model without noticeable overhead.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, input_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Ensure the logits match the module parameter dtype (e.g., when main
        # model runs in fp16/amp but the gradient learner stays in fp32).
        target_dtype = next(self.parameters()).dtype
        return self.net(logits.to(dtype=target_dtype))
