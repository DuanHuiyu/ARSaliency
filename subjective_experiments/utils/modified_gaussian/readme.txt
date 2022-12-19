All saliency map generation methods take a fixation map as input with sigma/kappa parameter, except the viewport method, which takes in a list of fixations in screen viewport coordinates and head orientation data. The screen coordinates are not normalized. Head orientation data is represented as normalized texture coordinates on the image itself (intersection of a vector from the head to the 360 image sphere, with 0,0 as bottom left corner). The gaussian function takes a parameter to indicate if the modified (correct) or naive distriubtion is used.

By default these methods are not enabled for parallel computation. The Kent and distorted gaussian (kent.m and gauss.m) can easily be converted into their parallel form by uncommenting the parfor loop for image rows, and commenting out the existing for loop. For Kent this is line 42, and for gauss this is line 31.

For this code you will need a few libraries, the Matlab Image Processing Toolbox, and Matlab Robotics System Toolbox. Please install them before using the functions.

Included in the code folder is the PanoBasic toolbox. This is credited to 
Y. Zhang, S. Song, P. Tan, and J. Xiao, please read the readme.txt inside PanoBasic for the publication to cite if using this in future work. You must add it to your matlab path before running methods that utilize it.


Links to file exchange libraries:
HPF: For kent(...) and kappa > 707 https://www.mathworks.com/matlabcentral/fileexchange/36534-hpf-a-big-decimal-class
conv_fft2:  For kent and distorted Gaussian if using the faster fft2 filtering method. This is commented out by default
https://www.mathworks.com/matlabcentral/fileexchange/31012-2-d-convolution-using-the-fft