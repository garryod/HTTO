from cupy import isinf, isnan, log, mean, ndarray, float32


def normalize_data(data: ndarray, flats: ndarray, darks: ndarray) -> ndarray:
    """Performs image normalization with reference to flatfields and darkfields.

    Returns:
        ndarray: A cupy array of normalized projections.
    """
    dark0 = mean(darks, axis=0, dtype=float32)
    flat0 = mean(flats, axis=0, dtype=float32)

    # same as tomopy implementation
    denom = (flat0 - dark0)
    denom[denom<1e-6] = 1e-6
    data = (data - dark0) / denom
    data[data > 10] = 10.
    data[data <= 0.0] = 1e-6
    data = -log(data)

    # old version
    # -----------
    # data = (data - dark0) / (flat0 - dark0 + 1e-3)
    # data[data<=0] = 1
    # data  = -log(data)
    # data[isnan(data)] = 6.0
    # data[isinf(data)] = 0

    return data
