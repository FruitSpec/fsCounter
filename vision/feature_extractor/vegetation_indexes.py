import numpy as np
import cupy as cp

def num_deno_nan_divide(numerator, denominator, return_numpy=True):
    if isinstance(numerator, type(np.array([]))):
        numerator = cp.array(numerator)
        denominator = cp.array(denominator)
    result = cp.empty(numerator.shape)
    denominator_no_0 = cp.not_equal(denominator, 0)
    result[denominator_no_0] = cp.divide(numerator[denominator_no_0], denominator[denominator_no_0])
    result[1-denominator_no_0] = cp.nan
    if return_numpy:
        return cp.asnumpy(result)
    else:
        return result


def num_deno_nan_divide_np(numerator, denominator):
    result = np.full(numerator.shape, np.nan)
    denominator_no_0 = np.abs(denominator) >= 1
    result[denominator_no_0] = numerator[denominator_no_0] / denominator[denominator_no_0]
    result[1-denominator_no_0] = np.nan
    return result


def dvi(nir, red, **kwargs):
    """
    :return: nir - red
    """
    return nir - red


def evi(nir, red, blue, **kwargs):
    nir = nir/255
    red = red/255
    blue = blue/255
    numerator = nir - red
    denominator = nir + 6 * red - 7.5 * blue + 1
    return num_deno_nan_divide(2.5 * numerator, denominator)


def ndvi(nir, red, **kwargs):
    numerator = nir - red
    denominator = nir + red
    return num_deno_nan_divide(numerator, denominator)


def ndvi_cuda(nir, red, **kwargs):
    red = cp.array(red)
    nir = cp.array(nir)
    numerator = nir - red
    denominator = nir + red
    return num_deno_nan_divide(numerator, denominator)


def gemi(nir, red, **kwargs):
    nir = nir/255
    red = red/255
    eta = (2 * (nir ** 2 - red ** 2) + 1.5 * nir + 0.5 * red) / (nir + red + 0.5)
    gemi_val = eta * (1 - 0.25 * eta) - num_deno_nan_divide((red - 0.125), (1 - red), return_numpy=False)
    gemi_val[red == 1] = cp.nan
    return gemi_val


def gari(nir, red, green, blue, gamma=1.7, **kwargs):
    numerator = nir - (green - gamma * (blue - red))
    denominator = nir + (green - gamma * (blue - red))
    return num_deno_nan_divide(numerator, denominator)


def gci(nir, green, **kwargs):
    return num_deno_nan_divide(nir, green) - 1


def gdvi(nir, green, **kwargs):
    return nir - green


def gli(red, green, blue, **kwargs):
    numerator = 2 * green - red - blue
    denominator = 2 * green + red + blue
    return num_deno_nan_divide(numerator, denominator)


def gndvi(nir, green, **wkargs):
    return num_deno_nan_divide(nir - green, nir + green)


def gosavi(nir, green, **wkargs):
    nir = nir / 255
    green = green / 255
    return num_deno_nan_divide(nir - green, nir + green + 0.16)


def grvi(nir, green, **kwargs):
    return num_deno_nan_divide(nir, green)


def ipvi(nir, red, **kwargs):
    return num_deno_nan_divide(nir, nir + red)


def lai(nir, red, blue, **kwargs):
    return evi(nir, red, blue) * 3.618 - 0.118


def mnli(nir, red, l_cons=0.5, **kwargs):
    nir_squred = nir ** 2
    return num_deno_nan_divide((nir_squred - red) * (1 + l_cons), nir_squred + red + l_cons)


def msavi2(nir, red, **kwargs):
    nir = nir / 255
    red = red / 255
    return (2 * nir + 1 - cp.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2


def msr(nir, red, **kwargs):
    nir_dev_red = num_deno_nan_divide(nir, red, return_numpy=False)
    numerator = nir_dev_red - 1
    denominator = cp.sqrt(nir_dev_red) + 1
    return num_deno_nan_divide(numerator, denominator)


def nli(nir, red, **kwargs):
    nir_squred = nir ** 2
    numerator = nir_squred - red
    denominator = nir_squred + red
    return num_deno_nan_divide(numerator, denominator)


def rdvi(nir, red, **kwargs):
    numerator = nir - red
    denominator = cp.sqrt(nir + red)
    return num_deno_nan_divide(numerator, denominator)


def savi(nir, red, **kwargs):
    nir = nir / 255
    red = red / 255
    numerator = 1.5 * (nir - red)
    denominator = nir + red + 0.5
    return num_deno_nan_divide(numerator, denominator)


def sr(nir, red, **kwargs):
    return num_deno_nan_divide(nir, red)


def tdvi(nir, red, **kwargs):
    nir = nir / 255
    red = red / 255
    numerator = nir - red
    denominator = cp.sqrt(nir ** 2 + red + 0.5)
    return 1.5 * num_deno_nan_divide(numerator, denominator)


def tgi(red, green, blue, lambda_red=670, lambda_green=550, lambda_blue=480, **kwargs):
    return ((lambda_red - lambda_blue) * (red - green) - (lambda_red - lambda_green) * (red - blue)) / 2


def vari(red, green, blue, **kwargs):
    numerator = green - red
    denominator = green + red - blue
    return num_deno_nan_divide(numerator, denominator)


def wdrvi(nir, red, alpha=0.2, **kwargs):
    numerator = alpha * nir - red
    denominator = alpha * nir + red
    return num_deno_nan_divide(numerator, denominator)


def ndre(nir, red, **kwargs):
    red_edge = num_deno_nan_divide(nir, red, return_numpy=False) - 1
    numerator = nir - red_edge
    denominator = nir + red_edge
    return num_deno_nan_divide(numerator, denominator)


def arvi(nir, red, blue, **kwargs):
    numerator = nir - 2 * red + blue
    denominator = nir + 2 * red + blue
    return num_deno_nan_divide(numerator, denominator)


def sipi(nir, red, blue, **kwargs):
    numerator = nir - blue
    denominator = nir - red
    return num_deno_nan_divide(numerator, denominator)


def pvi(nir, red, a=0.3, b=0.5, **kwargs):
    nir = nir / 255
    red = red / 255
    return (nir - a * red - b) / cp.sqrt(1 + a**2)


def ndri(nir, swir_975, **wkargs):
    return num_deno_nan_divide(nir - swir_975, nir + swir_975)


def vegetation_functions():
    return {"dvi": dvi,
            "ndvi": ndvi,
            "evi": evi,
            "gemi": gemi,
            "gari": gari,
            "gci": gci,
            "gdvi": gdvi,
            "gli": gli,
            "gndvi": gndvi,
            "gosavi": gosavi,
            "grvi": grvi,
            "ipvi": ipvi,
            "lai": lai,
            "mnli": mnli,
            "msavi2": msavi2,
            "msr": msr,
            "nli": nli,
            "rdvi": rdvi,
            "savi": savi,
            "sr": sr,
            "tdvi": tdvi,
            "tgi": tgi,
            "vari": vari,
            "wdrvi": wdrvi,
            "ndre": ndre,
            "arvi": arvi,
            "sipi": sipi,
            "pvi": pvi,
            "ndri": ndri}
