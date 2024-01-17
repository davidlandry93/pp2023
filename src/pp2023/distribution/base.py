import torch


def quantile_loss(quantile_values, obs, quantiles=None):
    """Args
    quantile_values: Tensor with the last dimension reserved for quantile values.
    obs: The observation to score against.
    quantiles: Tensor containing the quantile that are target by the values. If
        None, will consider the quantiles are uniformly spred from 0 to 1. Must
        have the same length as quantile_values.shape[-1]"""

    if quantiles is None:
        n_quantiles = quantile_values.shape[-1]
        lower_bound = 1.0 / (n_quantiles + 1)
        upper_bound = 1.0 - lower_bound
        quantiles = torch.linspace(
            lower_bound, upper_bound, n_quantiles, device=quantile_values.device
        )

    obs = obs.unsqueeze(-1)
    left_quantile_loss = -1.0 * (quantiles.unsqueeze(0)) * (quantile_values - obs)
    right_quantile_loss = (1 - quantiles.unsqueeze(0)) * (quantile_values - obs)

    quantile_loss = torch.where(
        quantile_values < obs, left_quantile_loss, right_quantile_loss
    )

    return quantile_loss.mean(dim=-1)
