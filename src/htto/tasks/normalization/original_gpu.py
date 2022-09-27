from cupy import isinf, isnan, log, mean, ndarray, float32


def normalize_data(data: ndarray, darks: ndarray, flats: ndarray) -> ndarray:
    """Performs image normalization with reference to flatfields and darkfields.

    Returns:
        ndarray: A cupy array of normalized projections.
    """
    dark0 = mean(darks, axis=0, dtype=float32)
    flat0 = mean(flats, axis=0, dtype=float32)
    data = (data - dark0) / (flat0 - dark0)
    data[data <= 0] = 1e-9
    data[isnan(data)] = 10.0
    data = -log(data)
    data[isinf(data)] = 0
    
    return data
