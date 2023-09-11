from typing import Callable
import numpy as np
import torch
import scipy.special


def binomial(n: int, k: int) -> torch.Tensor:
    """https://github.com/pytorch/pytorch/issues/47841"""
    mask = n.detach() >= k.detach()
    n = mask * n
    k = mask * k
    a = torch.lgamma(n + 1) - torch.lgamma((n - k) + 1) - torch.lgamma(k + 1)
    return torch.exp(a) * mask


def bernstein_polynomial_torch(
    degree: int, device=None
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a function that adds a trailing dimension which samples every term of the
    polynomial at sample points. The idea is that you would use the generated table to
    apply your coefficients and sum them up to get the final value of the polynomial."""
    nus = torch.arange(0, degree + 1, device=device).unsqueeze(-1)

    binom_coefs = binomial(torch.tensor(degree), nus)
    left_exp = nus
    right_exp = degree - nus

    def poly(x):
        """Args
            x: Tensor of shape [..., n]
        Returns
            Tensor of shape [..., degree+1, n] that contains the value of each term
            of the Bernstein Polynomial at the sample point."""
        return binom_coefs * x**left_exp * (1.0 - x) ** right_exp

    return poly


def bernstein_polynomial_np(degree: int) -> Callable[[np.array], np.array]:
    """Return a function that adds a trailing dimension which samples every term of the
    polynomial at sample points. The idea is that you would use the generated table to
    apply your coefficients and sum them up to get the final value of the polynomial."""
    nus = np.expand_dims(np.arange(0, degree + 1), axis=-1)

    binom_coefs = scipy.special.binom(degree, nus)
    left_exp = nus
    right_exp = degree - nus

    def poly(x):
        """Args
            x: Tensor of shape [..., n]
        Returns
            Tensor of shape [..., degree+1, n] that contains the value of each term
            of the Bernstein Polynomial at the sample point."""
        return binom_coefs * x**left_exp * (1.0 - x) ** right_exp

    return poly
