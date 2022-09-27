import unittest
import numpy as np

import cupy
from htto.tasks.normalization.original_cpu import normalize_data as normdata_cpu
from htto.tasks.normalization.original_gpu import normalize_data as normdata_gpu
from htto.tasks.normalization.numpy_cpu import normalize_data as normdata_np
from htto.tasks.normalization.kernels_gpu import normalize_data as normdata_gpu_kernels

class NormalizationTest(unittest.TestCase):

    def setUp(self):
        self.ctx = cupy.cuda.Device(0).use()        

    def tearDown(self):
        del self.ctx
        
    def prepare_arrays(self):
        N,X,Y = 10,128,128
        data = np.random.randint(500,1000,size=(N,X,Y), dtype="uint16")
        dark = np.random.randint(50,150,size=(N,X,Y), dtype="uint16")
        flat = np.random.randint(400,800,size=(N,X,Y),dtype="uint16")
        return data, dark, flat

    def test_normalize_numpy_cpu(self):
        data, dark, flat = self.prepare_arrays()
        
        # TomoPy normalisation on CPU
        normalized_tomopy = normdata_cpu(data,dark,flat)

        # Numpy normalisation on CPU 
        normalized_numpy = normdata_np(data, dark, flat)

        print(normalized_tomopy.dtype, normalized_numpy.dtype)
        
        # Assert
        np.testing.assert_array_almost_equal(normalized_tomopy, normalized_numpy,
                                             err_msg="The normalized data array has not been updateed correctly")

    
    def test_normalize_simple_cupy(self):
        data, dark, flat = self.prepare_arrays()
        data_gpu = cupy.asarray(data)
        dark_gpu = cupy.asarray(dark)
        flat_gpu = cupy.asarray(flat)
        
        # TomoPy normalisation on CPU
        normalized_cpu = normdata_cpu(data,dark,flat)

        # Simple CuPy normalisation on GPU 
        normalized_gpu = normdata_gpu(data_gpu, dark_gpu, flat_gpu)

        print(normalized_cpu.dtype, normalized_gpu.get().dtype)
        
        # Assert
        np.testing.assert_array_almost_equal(normalized_cpu, normalized_gpu.get(),
                                             err_msg="The normalized data array has not been updateed correctly")


    def test_normalize_kernels_cupy(self):
        data, dark, flat = self.prepare_arrays()
        data_gpu = cupy.asarray(data)
        dark_gpu = cupy.asarray(dark)
        flat_gpu = cupy.asarray(flat)
        
        # TomoPy normalisation on CPU
        normalized_cpu = normdata_cpu(data,dark,flat)

        # CuPy normalisation with kernels on GPU 
        normalized_gpu = normdata_gpu_kernels(data_gpu, dark_gpu, flat_gpu)

        print(normalized_cpu.dtype, normalized_gpu.get().dtype)
        
        # Assert
        np.testing.assert_array_almost_equal(normalized_cpu, normalized_gpu.get(),
                                             err_msg="The normalized data array has not been updateed correctly")

    
