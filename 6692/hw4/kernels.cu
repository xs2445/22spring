/* 
This file contains CUDA functions for implementing deep learning network layers.

E6692 Spring 2022
*/

#define BLOCK_SIZE 32


__global__ void transpose(float* input, float* output, int outWidth, int outHeight){
    /* 
        Transposes input and writes the result to output.
    
        params:
           input: pointer to the input matrix to be transposed
           output: pointer to the output transposed matrix
           outWidth: the output width in number of columns
           outHeight: the input height in number of rows
    */

    __shared__ float block[BLOCK_SIZE][BLOCK_SIZE+1];
    
    // read the matrix tile into shared memory
    // load one element per thread from device memory (input) and store it
    // in transposed order in block[][]
    
    unsigned int xIndex = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    
    if(xIndex < outWidth && yIndex < outHeight){
        unsigned int index_in = yIndex * outWidth + xIndex;
        block[threadIdx.y][threadIdx.x] = input[index_in];
    }

    // synchronize to ensure all writes to block[][] have completed
    __syncthreads();

    // write the transposed matrix tile to global memory (output) in linear order
    xIndex = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    
    if(xIndex < outHeight && yIndex < outWidth){
        unsigned int index_out = yIndex * outHeight + xIndex;
        output[index_out] = block[threadIdx.x][threadIdx.y];
    }
}


__global__ void add(float* A, float* B, int size){
    /*
        Performs addition between A and B. The result
        of the addition is written to A.
        
        A = A + B
    
        A and B should have the same shapes (size).
        
        params:
            A: pointer to input matrix A
            B: pointer to input matrix B
            size: the length of A and B
    */
    
    //////////////////////////////////////////////////////////////////
    /******************* YOUR IMPLEMENTATION HERE *******************/
    //////////////////////////////////////////////////////////////////

    
    //////////////////////////////////////////////////////////////////
    /******************* END YOUR IMPLEMENTATION ********************/
    //////////////////////////////////////////////////////////////////
    
}



__global__ void relu(float* input, int xLen, int yLen){

    /* 
        Calculates the nonlinear ReLU activation function.
        
        params:
            input: pointer to input matrix
            xLen: the size of the input in the x direction (number of columns)
            yLen: the size of the input in the y direction (number of rows)
    */
    
    //////////////////////////////////////////////////////////////////
    /******************* YOUR IMPLEMENTATION HERE *******************/
    //////////////////////////////////////////////////////////////////

    
    //////////////////////////////////////////////////////////////////
    /******************* END YOUR IMPLEMENTATION ********************/
    //////////////////////////////////////////////////////////////////

}


__global__ void conv2d(float* input, float* mask, 
                       float* output, int inputWidth,
                       int inputHeight, int maskSize){
    /* 
        Performs convolution between input and mask. 
        
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        
        We make the following assumptions:
            - the input is 2D (H, W) where H == W
            - mask is a 2D odd and square matrix (i.e. 3x3, 5x5, 7x7, etc.)
            - no padding is required
            - the output shape is the same as the input shape. 
              This corresponds to setting padding='same' in
              torch.nn.functional.conv2d. See the link below:
            - stride = 1
            - dilation = 1
            
        https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
    
        params:
           input: pointer to the input matrix to be convolved
           mask: pointer to the convolutional mask matrix
           output: pointer to the output convolved matrix
           inputWidth: the input width in number of columns
           inputHeight: the input height in number of rows
           maskSize: the length of the square convolutional mask
    */
    
    //////////////////////////////////////////////////////////////////
    /******************* YOUR IMPLEMENTATION HERE *******************/
    //////////////////////////////////////////////////////////////////

    
    //////////////////////////////////////////////////////////////////
    /******************* END YOUR IMPLEMENTATION ********************/
    //////////////////////////////////////////////////////////////////

}


__global__ void MaxPool2d(float* input, float* output, int inputWidth, int inputHeight, int kernelSize, int stride){
    /* 
        Performs 2D max-pooling on the input array. 
        
        We make the following assumptions:
            - square pooling kernel
            - kernelSize == stride
            - square input array
                
        params:
           input: pointer to the input matrix to be max-pooled
           output: pointer to the output matrix, result of max-pooling
           inputWidth: the input width in number of columns
           inputHeight: the input height in number of rows
           kernelSize: the length of the square pooling window
           stride: the stride of the pooling operation. This is 
                   equal to kernelSize for our purposes.
    */
    
    //////////////////////////////////////////////////////////////////
    /******************* YOUR IMPLEMENTATION HERE *******************/
    //////////////////////////////////////////////////////////////////

    
    //////////////////////////////////////////////////////////////////
    /******************* END YOUR IMPLEMENTATION ********************/
    //////////////////////////////////////////////////////////////////

}


__global__ void dot(float* A, float* B, float* C, int ARows, int ACols, int BRows,
    int BCols, int CRows, int CCols){
    /* 
        Performs matrix multiplication between A and B. 
        
        A x B = C
        
        We make the following assumption:
            - ARows == BCols

        params:
           A: pointer to the input matrix A
           B: pointer to the output matrix B
           ARows: the number of rows in A
           ACols: the number of columns in A
           BRows: the number of rows in B
           BCols: the number of columns in B
           CRows: the number of rows in C
           CCols: the number of columns in C
    */
    
    //////////////////////////////////////////////////////////////////
    /******************* YOUR IMPLEMENTATION HERE *******************/
    //////////////////////////////////////////////////////////////////

    
    //////////////////////////////////////////////////////////////////
    /******************* END YOUR IMPLEMENTATION ********************/
    //////////////////////////////////////////////////////////////////

}
