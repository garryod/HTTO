from numpy import ndarray, mean, log, float32


def normalize_data(data: ndarray, flats: ndarray, darks: ndarray) -> ndarray:
    """Performs image normalization with reference to flatfields and darkfields.

    Args:
        data: A numpy array containing the sample projections.
        flats: A numpy array containing the flatfield projections.
        darks: A numpy array containing the dark projections.

    Returns:
        ndarray: A numpy array of normalized projections.
    """
    data = data.astype(float32)
    dark0 = mean(darks, axis=0, dtype=float32)
    flat0 = mean(flats, axis=0, dtype=float32)
    denom = flat0 - dark0
    denom[denom<1e-6] = 1e-6
    data = (data - dark0) / denom
    data[data > 10] = 10.
    data[data <= 0.0] = 1e-09
    data = -log(data)
    
    return data
