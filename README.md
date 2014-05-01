README
======
 1. About
 2. Prerequisites
 3. Building
 4. Usage
 5. Example Output
 
## 1. About

 This is a simple layered depth of field rendering implementation for synthetic images. It requires an input image (preferably synthetic) and an associated depth map. A sample input image and depth map are present in the images directory. The code has been written using CUDA, so it needs a computer with an NVIDIA CUDA capable graphics card to compile and execute. It works by convolving the image with different sized gaussian filters, where the filter size at each point is determined by its absolute depth distance from the depth of focus. The implementation showcases a few optimizations in terms of the algorithm (separable convolution) and memory management (tiled convolution, caching, etc.). On my computer (MBP Mid-2012 with low-end graphics card), the 900 x 600 test image is processed in a little over 3ms. The implementation for separable convolution is mainly inspired from NVIDIA's CUDA samples, with significant modifications. Also, most of the Makefile is from NVIDIA's CUDA samples, so if you are able to compile the samples, you should be able to compile this code too. However, it still might take some modifications to the Makefile to get the code to compile on your machine - I am still working on trying to make it more general.

## 2. Prerequisites

 This project requires:
  * GNU Make or equivalent.
  * Clang C++ compiler.
  * NVIDIA CUDA 5.5 and above
  * OpenCV 2.4 and above

## 3. Building

  * Clone the repository using:
  
    git clone https://github.com/vatsnv/render_layered_dof.git
  
  * Navigate to the build directory and run make:
  ```
    cd build
    make
  ```

  You might need to set the CUDA_PATH variable on line 3 of the Makefile.
  
## 4. Usage 

  * Run the executable in the build directory with the sample input image and depth map as arguments.
  ```
    cd build
    ./test_dof_renderer ../images/image.png ../images/depth_map.png
  ```

  Click on any point in the displayed image to render the depth of field effect (currently works only in grayscale). Press any key on the keyboard to exit.

## 5. Example Output

![Example Output](images/output.png?raw=true)
