import properscoring
import xarray as xr


def crps_gaussian_xr(obs: xr.DataArray, mean: xr.DataArray, std: xr.DataArray):
    crps = properscoring.crps_gaussian(obs, mean, std)
    crps_xr = xr.zeros_like(obs)
    crps_xr.values = crps

    return crps_xr
