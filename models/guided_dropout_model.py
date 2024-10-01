import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class GuidedDropoutModel(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.2):
        super(GuidedDropoutModel, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 8192)
        self.fc2 = nn.Linear(8192, 8192)
        self.fc3 = nn.Linear(8192, num_classes)
        # Initialize strength parameters for each node
        self.strength_fc1 = nn.Parameter(torch.ones(8192))
        self.strength_fc2 = nn.Parameter(torch.ones(8192))
        self.dropout_rate = dropout_rate
        # Flag to enable/disable dropout
        self.dropout_enabled = False

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.apply_layer_with_guided_dropout(x, self.fc1, self.strength_fc1)
        x = self.apply_layer_with_guided_dropout(x, self.fc2, self.strength_fc2)
        x = self.fc3(x)
        return x

    def apply_layer_with_guided_dropout(self, x, layer, strength):
        x = F.relu(layer(x))
        # Apply strength parameters
        x = x * strength

        if self.dropout_enabled:
            # Apply guided dropout (DR)
            x = self.apply_guided_dropout_dr(x, strength)
        return x

    def apply_guided_dropout_dr(self, activations, strength):
        # Binning strength values
        num_bins = 100
        # Convert strength to CPU numpy array for binning
        strength_np = strength.detach().cpu().numpy()
        hist, bin_edges = np.histogram(strength_np, bins=num_bins)
        max_bin_count = hist.max()
        max_bin_index = hist.argmax()
        # Find the bin range corresponding to inactive nodes
        bin_lower = bin_edges[max_bin_index]
        bin_upper = bin_edges[max_bin_index + 1]
        # Create mask for inactive nodes
        inactive_mask = ((strength >= bin_lower) & (strength < bin_upper)).float()
        # Active nodes are not in the inactive bin
        active_mask = 1.0 - inactive_mask
        f_m = inactive_mask.sum().item()  # Number of inactive nodes
        N = strength.numel()              # Total number of nodes
        theta = 1.0 - self.dropout_rate   # Keep probability

        # Adjusted dropout probability
        p_adjusted = 1.0 - (f_m / N) * (1.0 - theta)
        # Avoid division by zero
        p_adjusted = max(p_adjusted, 1e-8)

        # Sample dropout mask for active nodes
        dropout_mask = torch.ones_like(active_mask)
        dropout_prob = self.dropout_rate  # Dropout rate for active nodes
        # Generate dropout indices only for active nodes
        dropout_indices = torch.bernoulli(torch.full_like(active_mask, dropout_prob))
        dropout_mask = dropout_mask - (dropout_indices * active_mask)

        # Apply dropout mask
        masked_activations = activations * dropout_mask

        # Scale activations to maintain expected value
        masked_activations = masked_activations / p_adjusted

        return masked_activations
