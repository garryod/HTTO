import cupy as cp
from cupy import isinf, isnan, log, mean, ndarray, float32


def normalize_data(data: ndarray, flats: ndarray, darks: ndarray) -> ndarray:
    """Performs image normalization with reference to flatfields and darkfields.

    Returns:
        ndarray: A cupy array of normalized projections.
    """
    dark0 = mean(darks, axis=0, dtype=float32)
    flat0 = mean(flats, axis=0, dtype=float32)
    out = cp.zeros(data.shape, dtype=float32)

    norm_kernel = cp.RawKernel(
        """extern "C" __global__ void normalize(const unsigned short* data,
           const float* flat,
           const float* dark,
           float* out, float eps, float cutoff, int A, int B)
           {
             int bid = blockIdx.x;
             int tx = threadIdx.x;
             int ty = threadIdx.y;

             data += bid * A * B;
             out += bid * A * B;
             
             for (int a = ty; a < A; a += blockDim.y)
    	     {
 	     #pragma unroll(4)
	     for (int b = tx; b < B; b += blockDim.x)
	        {

                float denom = flat[a * B + b] - dark[a * B + b];
                if (denom < eps)
                {
                  denom = eps;
                }
                float tmp = (float(data[a * B + b]) - dark[a * B + b]) / denom;
                if (tmp > cutoff)
                {
                  tmp = cutoff;
                }
                if (tmp <= 0)
                {
                  tmp = eps;
                }
	        out[a * B + b] = -log(tmp);
    	        }
	     }
           }""","normalize")

    grids = (32,32,1)
    blocks = (int(data.shape[0]),1,1)
    params = (data,flat0,dark0,out,float32(1e-9),float32(10.),int(data.shape[1]), int(data.shape[2]))
    norm_kernel(grids, blocks, params)
    return out
