"""
This file contains the Context class, which defines the CUDA GPU context for the given computing 
environment (Jetson Nano), and the GPUKernels class, which contains functions that handle memory
allocation and data transfer between the CPU and GPU.

E6692 Spring 2022
"""
# initialize the CUDA GPU context
from pycuda import autoinit 
from pycuda.driver import Device

# PyTorch tries reinitialize the context when using GPU functions. The following line
# tells PyTorch not to do that.
Device(0).retain_primary_context() 

import numpy as np
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray


class Context:
    def __init__(self, block_size):
        """
        Initialize the CUDA context.

        params:
            block_size (int): defines the block size for parallel computation            
        """
        self.block_size = block_size
        
        self.block_dims1D = (self.block_size, 1, 1)

        self.block_dims = (self.block_size, self.block_size, 1) # define block and grid dimensions
        
        self.grid_dims1D = lambda length : (int(np.ceil(length / self.block_size)), 1, 1)
        
        # shape = (cols, rows)
        self.grid_dims = lambda shape : (int(np.ceil(shape[1] / self.block_size)), int(np.ceil(shape[0] / self.block_size)), 1)

        
    def getSourceModule(self, kernel_path, multiple_kernels=False):
        """
        Load the CUDA kernels. Can load kernels from a single file or from
        multiple files.

        params:
            kernel_path (str or list of str): the path to the .cu CUDA kernel file
                                              or list of paths. If list of paths then
                                              multiple_kernels must be set to True.

            multiple_kernels (bool): specifies if kernel_path contains one path
                                     or list of paths.

        returns:
            Instance of pycuda.compiler.SourceModule with user parallel kernels loaded.
        """
        if multiple_kernels:
            try:
                is_iterable = iter(kernel_path)
                kernelwrapper = ''
                for kernel in kernel_path: # if kernel_path is a list, iteratively read files
                    kernelwrapper += open(kernel).read() # concatenate files as strings
                return SourceModule(kernelwrapper) # load concatenated kernel code into source module

            except TypeError:
                raise Exception('{} is not iterable. Set multiple_kernels=False or define kernel_path as list of paths.'.format(kernel_path))
        else:
            kernelwrapper = open(kernel_path).read() # read CUDA kernels from .cu file

        return SourceModule(kernelwrapper) # return source module with kernels loaded


class GPUKernels():
    """
    A class that defines calls to CUDA kernel functions on the GPU.
    Functions of this class facilitate memory allocation, block and grid
    dimension calculations, type casting, kernel function calls, and transfer
    of data between the CPU and GPU.
    """
    
    def __init__(self, context, source_module):
        """
        Initialize the GPUKernels class.
        
        params:
            context (Context): a GPU context object (defined above)
            source_module (pycuda.compiler.SourceModule): a source module object with loaded CUDA kernels
        """
        self.context = context
        self.source_module = source_module
        
    
    def add(self, input_array_a, input_array_b):
        """
        Calls the add() CUDA function. 
        
        params:
            input_array_a (np.array): input array A
            input_array_b (np.array): input array B
            
        returns:
            cuda_output (np.array): the output array A + B
            
        NOTE: Your arrays need only be 1D. Since we are performing 
              elementwise addition, and multidimensional arrays 
              are represented lineraly in memory, 1D and 2D addition
              are functionally equivalent.
        """
        # require matching dimensions
        assert input_array_a.shape == input_array_b.shape
        
        # get CUDA function
        add_cuda = self.source_module.get_function('add')
        
        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################
        
        # multi-dimention shape of matrics
        shape = input_array_a.shape
        
        # size of the flatten array
        size = shape[0]
        for i in range(1,len(shape)):
            size *= shape[i]

        A_d = gpuarray.to_gpu(input_array_a.astype(np.float32))
        B_d = gpuarray.to_gpu(input_array_b.astype(np.float32))
        add_cuda(
            A_d, 
            B_d, 
            np.int32(size), 
            block=self.context.block_dims1D, 
            grid=self.context.grid_dims1D(size)
        )
        
        cuda_output = A_d.get()
    
        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################
        
        return cuda_output
    
    
    def relu(self, input_array):
        """
        Calls the relu() CUDA function. This function needs 
        to be able to handle 1D and 2D input arrays.
        
        params:
            input_array (np.array): input array A, 1D or 2D
            
        returns:
            cuda_output (np.array): the output array, or relu activated input array.
                                    cuda_output should have the same shape as input_array
        """
        # get CUDA relu function
        relu_cuda = self.source_module.get_function('relu')
        
        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################
        
        shape = input_array.shape
        # height 
        yLen = shape[0]
        # width
        xLen = 1
        if len(shape) != 1:
            xLen = shape[1]
        
        input_d = gpuarray.to_gpu(input_array.astype(np.float32))
        
        block_size = self.context.block_dims
        grid_size = self.context.grid_dims((xLen, yLen))
        
        relu_cuda(input_d, np.int32(yLen), np.int32(xLen), block=block_size, grid=grid_size)
        
        cuda_output = input_d.get()
    
        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################
        
        return cuda_output

        
    def conv2d(self, input_array, mask):
        """
        Calls the conv2d() CUDA function.
        
        We make the following assumptions to simplify the implementation:
        - input_array is 2D (H, W) where H == W
        - mask is a 2D odd and square array (i.e. 3x3, 5x5, 7x7, etc.)
        - no padding is required
        - the output shape is the same as the input shape. 
          This corresponds to setting padding='same' in
          torch.nn.functional.conv2d.
        - stride = 1
        - dilation = 1
                
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        
        params:
            input_array (np.array): the input array to be convolved
            mask (np.array): the convolutional mask/filter/kernel
            
        returns:
            cuda_output (np.array): the result of the convolution
        """
        # require square inputs and odd kernel
        assert input_array.shape[0] == input_array.shape[1]
        assert mask.shape[0] == mask.shape[1] # square assertion
        assert mask.shape[0] % 2 == 1 # odd dimension assertion
        
        # get CUDA conv2d function
        conv2d_cuda = self.source_module.get_function('conv2d')
        
        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################
    
        output_array = np.zeros_like(input_array)
        
        input_array_d = gpuarray.to_gpu(input_array.astype(np.float32))
        output_array_d = gpuarray.to_gpu(output_array.astype(np.float32))
        mask_d = gpuarray.to_gpu(mask.astype(np.float32))
        
        mask_size = mask.shape[0]
        shape = input_array.shape
        
        block_size = self.context.block_dims
        grid_size = self.context.grid_dims(shape)
        
        conv2d_cuda(
            input_array_d, 
            mask_d, 
            output_array_d, 
            np.int32(shape[1]), 
            np.int32(shape[0]), 
            np.int32(mask_size),
            block=block_size,
            grid=grid_size
        )

        cuda_output = output_array_d.get()
    
        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################

        return cuda_output
        
        
    def MaxPool2d(self, input_array, kernel_size=2):
        """
        Calls the MaxPool2d() CUDA function.
        
        We assume:
        - square pooling kernel
        - kernel_size == stride
        - square input array
        
        params:
            input_array (np.array): the input array
            kernel_size=2 (int): the length of the pooling kernel
            
        returns:
            cuda_output (np.array): the max-pooled array
        """
        # require square input array
        assert input_array.shape[0] == input_array.shape[1]
        
        # get CUDA MaxPool2d function
        MaxPool2d_cuda = self.source_module.get_function('MaxPool2d')
        
        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################
    
        input_array_d = gpuarray.to_gpu(input_array.astype(np.float32))
        h, w = input_array.shape
        h_out = h//kernel_size
        w_out = w//kernel_size
        
        output_array_d = gpuarray.empty((h_out, w_out), dtype=np.float32)
        
        block_size = self.context.block_dims
        grid_size = self.context.grid_dims((h_out,w_out))
        
        MaxPool2d_cuda(
            input_array_d, 
            output_array_d, 
            np.int32(w), 
            np.int32(h), 
            np.int32(kernel_size), 
            np.int32(kernel_size),
            block=block_size,
            grid=grid_size
        )
        
        cuda_output = output_array_d.get()
        
        
        
    
        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################

        return cuda_output
        
        
    def linear(self, input_array, weights, bias):
        """
        This function implements the forward pass of a dense, 
        fully connected layer of a neural network using CUDA 
        kernel functions.
        
        cuda_output = input_array * weights^T + bias
        
        Your implementation should roughly be:
        
        1. Transpose the weight matrix
        2. Matrix multiply the input and the transposed weights
        3. Add the bias term to the result of the matrix multiplication.
        
        You will need to allocate memory buffers for the intermediate 
        results of the operations above. For example, to store the 
        transposed weights, initialize a pycuda.gpuarray with the 
        transposed shape of the weights to store the result and use 
        it for the next operation.
        
        NOTE: The intermediate result buffers do NOT need to be 
              transferred from GPU to CPU in between GPU operations.
        
        params:
            input_array (np.array): the input matrix of shape (1, N)
            weights (np.array): the weight matrix of shape (M, N)
            bias (np.array): the bias matrix of shape (1, M)
        """
        # require dimensions
        assert len(input_array.shape) == 1
        assert len(weights.shape) == 2
        assert len(bias.shape) == 1
        
        # require matching lengths for dot products and additions
        assert input_array.shape[0] == weights.shape[1]
        assert weights.shape[0] == bias.shape[0]
        
        # get CUDA functions
        transpose_cuda = self.source_module.get_function('transpose')
        dot_cuda = self.source_module.get_function('dot')
        add_cuda = self.source_module.get_function('add')
        
        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################
    
        len_A = input_array.shape
        shape_B = weights.T.shape
        len_C = shape_B[1]
          
        input_d = gpuarray.to_gpu(input_array.astype(np.float32))
        weights_d = gpuarray.to_gpu(weights.astype(np.float32))
        weights_t = gpuarray.empty(shape_B, dtype=np.float32)
        output_d = gpuarray.empty(len_C, dtype=np.float32)
        bias_d = gpuarray.to_gpu(bias.astype(np.float32))
        
        # calculate block and grid dimensions
        block_size = self.context.block_dims
        max_length = max(shape_B)
        grid_size = self.context.grid_dims((max_length, max_length))

        # call CUDA function
        transpose_cuda(
            weights_d, 
            weights_t, 
            np.int32(shape_B[0]), 
            np.int32(shape_B[1]),
            block=block_size, 
            grid=grid_size
        )
        
        block_size = self.context.block_dims1D
        grid_size = self.context.grid_dims1D(len_C)
        
        dot_cuda(
            input_d, 
            weights_t, 
            output_d, 
            np.int32(1),
            np.int32(len_A),
            np.int32(shape_B[0]),
            np.int32(shape_B[1]),
            np.int32(1),
            np.int32(shape_B[1]),
            block=block_size,
            grid=grid_size
        )
        
        add_cuda(
            output_d,
            bias_d,
            np.int32(len_C),
            block=block_size,
            grid=grid_size
        )
        
        cuda_output = output_d.get()
    
        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################

        return cuda_output
    
    
    def flatten(self, input_array):
        """
        flattens the given array to 1D. See:
        
        https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
        
        for details.
        """
        return input_array.flatten()
