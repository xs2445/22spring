# e6692-2022Spring-assign1-Lab-convolution

Assignment 1, completed assigned files which contains convolution and correlation for 1D and 2D, dilated convolution.

### Files finished
* Jupyter Notebook `1DConvolution.ipynb`
* Jupyter Notebook `2DConvolution.ipynb`
* Utility file `utils/convolution1D.py`
* Utility file `utils/convolution2D.py`

### Functions
```python
create_rect_signal(base_ampl=0, base_length=128, function_ampl=1, function_length=12)
create_1dconv_mask_sawtooth(base_ampl=1, end_ampl=8, steps=1):
calc_conv(conv_mask, input_signal)
calc_corr(corr_mask, input_signal)
dilate_kernel(kernel, dilation_factor)
pad_img(image, dilated_kernel)
calc_conv2d(image, kernel)
```

### Usage
```python
from utils.convolution1D import *
```

### Organization of the repo
```
.
│  1DConvolution.ipynb
│  2DConvolution.ipynb
│  README.md
│
└─utils
        animation.py
        convolution1D.py
        convolution2D.py
        convolution2D_visuals.py
```
