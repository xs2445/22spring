# Lab-ParallelComputing

This lab contains starter code for: Lab-ParallelComputing.

### Files finished

* Jupyter Notebook `GPUkernels.ipynb`
* Jupyter Notebook `KernelProfiling.ipynb`
* Jupyter Notebook `Train.ipynb`
* Jupyter Notebook `CUDAInference.ipynb`
* Utility file `utils/context.py`
* Utility file `utils/models.py`
* Kernel file  `kernels.cu`

### Functions and usage
```python
from utils.context import Context, GPUKernels
# instantiate CUDA functions 
context = Context(BLOCK_SIZE)
source_module = context.getSourceModule(kernel_path)
cuda_functions = GPUKernels(context, source_module)
# CUDA implementations
cuda_functions.add(input_array_a, input_array_b)
cuda_functions.relu(input_array)
cuda_functions.conv2d(input_array, mask)
cuda_functions.MaxPool2d(input_array, kernel_size)
cuda_functions.linear(input_array, weights, bias)

from utils.models import CUDAClassifier
# load trained parameters (note cuda and pytorch model must have the same structure)
cuda_model = CUDAClassifier(kernel_path)
cuda_model.load_state_dict(torch.load(state_dict_path))
# CUDA inferencing
cuda_x = cuda_model(x)
cuda_prediction = np.argmax(cuda_x)
```

### Organization of the repo
```bash
tree ./ >> README.md
```

```
./
├── CUDAInference.ipynb
├── GPUkernels.ipynb
├── KernelProfiling.ipynb
├── README.md
├── Train.ipynb
├── kernels.cu
└── utils
    ├── context.py
    ├── dataset.py
    ├── models.py
    ├── plot_execution_times.py
    └── train.py

1 directory, 11 files
```