DESIGN
======
 1. Introduction
 2. Algorithm
 3. Memory management


## 1. Introduction

The point of depth-of-field rendering algorithms is to take a synthetic image and modify it to create the illusion of viewing it through an optical lens. When viewing natural images through a lens of non-negligible aperture size, only points at a certain depth (depends on the focal length of the lens) are in proper focus. The further a point is from this depth, the more blurred it appears in the captured image. This effect can be artifically approximated on synthetic images where depth information is available. There are many sophisticated ways to do this, but in this implementation, we focus on a simple method, using layered Gaussian Blurring to render the depth-of-field effect. The idea is to study the CUDA implementation of this algorithm and understand the different ways in which it can be optimized.

## 2. Algorithm

The high-level algorithm is rather simple. Since we have the corresponding depth-map for the synthetic image, given a point on the synthetic image, we simply blur all other points with Gaussian Kernels of various radii. The radius of convolution at a point (x,y) is proportional to the difference in depth of the point from the depth at the point of focus.

    R = k * |Z(x, y) - Z(x_d, y_d)|
    where 
        R           - radius of convolution
        k           - proportionality constant
        Z           - depth-map
        (x_d, y_d)  - point of focus

Since we are dealing with discrete images, we can discretize the kernel radius in steps of 1. In this implementation, we allow the kernel radius to vary from 0(no convolution) to 9(maximum blurring) in steps of 1, corresponding to distance ranges 0 to 1 in steps of 0.1.
To make things faster, we separate the convolution kernel into horizontal and vertical kernels which can be applied to the image independently. This reduces the computational complexity of convolution from O(NR^2) to O(2NR), where N is the number of pixels and R is the radius of convolution (assuming a square kernel). Also, the number of times each pixel is accessed is reduced from ~R^2 to ~2R, which is significant when considering that memory access is often expensive. Separable convolution is possible only when the square kernel can be represented as the dot product of two vectors, which is true for Gaussian kernels.

    K        =   k_x  .  k_y

    [. . .]      [.]  .  [. . .]
    [. . .]  =   [.]
    [. . .]      [.]

## 3. Memory management

NVIDIA's CUDA samples contain an example for performing separable convolution using CUDA, along with a detailed guide. The example code was used as a skeleton for this implementation. The most important consideration while managing GPU memory is restricting global memory reads. Reading shared, constant and local thread memory is much faster than global memory on GPUs. We use two separate GPU kernels for row and column-wise convolution and between the two, we write the intermediate data to global memory. Within each kernel, there are two stages: 

* *the load stage*, where each thread loads some data into the block's shared memory
* *the processing stage* where the data in the shared memory is processed and the output is written to global memory.

Betweent the two stages, there is a call to ```_syncthreads()``` to ensure all the data is available in shared memory before processing.

### 3.1 Alignment
Whenever a byte is accessed in global memory, the GPU caches the entire line (usually 64 bytes, but can be 32 or 128 bytes, depending on the device). Therefore, if consecutive threads access neighboring locations, the cost of reading from the 2nd thread onwards, is much smaller. To exploit this best, we need to make sure that every row of pixels in the image is 64-byte (or 128-byte) aligned. Towards this, we can make use of the functions ```cudaMallocPitch()``` and ```cudaMemcpy2D()``` which introduce the required padding at the end of each row while copying the input image to the GPU, such that each row is aligned in memory.

                                        image pixels              padding
                              <----------------------------------><---->
    row 0 (multiple of 64) -> |   |   |   |   |   |   ...     |   |00000 
    row 1 (multiple of 64) -> |   |   |   |   |   |   ...     |   |00000
    row 2 (multiple of 64) -> |   |   |   |   |   |   ...     |   |00000

### 3.2 Tiling
The image is divided into multiple blocks of pixels, each of which is to be processed by a block of threads. A simple arrangement would be to have the number of pixels equal to the number of threads within a block. However, this might lead to having a lot of idle threads during convolution, because the number of output pixels in convolution is much smaller than the number of input pixels for large kernels. The idle region along the border of the image block is called the 'apron' or the 'halo'. A better alternative is to have a one to many relationship between the threads and the pixels by introducing tiles within the blocks. A horizontal image block can be split into multiple horizontal tiles whose sizes match the thread block. Each thread is responsible for one pixel within each tile. Ensuring that each thread processes multiple pixels improves the throughput.

### 3.3 Shared memory and tiles
Since we need to access each pixel multiple times during the convolution operation, we make use of the shared memory within each thread-block to store the tiles being processed by that block. Shared memory bandwidth is much higher than that of global memory. If there are *n* tiles within the block, each thread loads *n* pixels into the shared memory. An example of the shared memory arrangement is as follows:

     7 Tiles of 8x4 pixels, 8x4 threads in thread-block
        * - pixels handled by thread (1,2)
     left halo                  processed tiles             right halo
     <-------->--------------------------------------------><------->
         0        1         2       3        4        5         6     
     |........|........|........|........|........|........|........| 
     |..*.....|..*.....|..*.....|..*.....|..*.....|..*.....|..*.....|   
     |........|........|........|........|........|........|........| 
     |........|........|........|........|........|........|........| 

The width of the processed tiles region is the width of the output region. The halo regions are accessed only for computing the output for pixels in the processed region. Note that tile 5 in this setup becomes the left halo tile for the next block and tile 6, the right halo becomes tile 0 for the next block.

### 3.4 Horizontal convolution
In our implementation, for the horizontal convolution, we use tiles of size 32x4 bytes, with 20 processed tiles in a block and 2 halo tiles. The higher the number of processed tiles, the higher is the throughput (in terms of pixels) per block. However, this limits the number of parallel blocks that can be deployed simultaneously, increasing the overall time required. The values used in this implementation were determined by experimenting and tuning repeatedly.

### 3.5 Vertical convolution
Vertical convolution is almost the same, except that instead of dividing the block into a single column of tiles, we use two columns. The size of each tile is 10x16 bytes and having two tiles adjacent to each other gives a total width of 32 bytes. Since the image is stored in row-major order, this leads to higher throughput from caching, compared to using just a single column of tiles. One question here is why not simply increase the tile size from 10x16 to 10x32 and use just a single column of tiles? In experiments, it was observed that a single column of wide tiles performs slower than two columns of narrow tiles. The likely reason is that increasing the tile width does take advantage of caching, but also doubles the number of threads in the block, leaving fewer threads for other blocks. Also, it is better to have each thread processing as many pixels as possible within its block, to reduce the percentage of idle threads. 

### 3.6 Constant memory usage
The constant memory area is used to store the convolution kernels of different radii, since their values are not meant to change while execution. The kernels are stored in a single array, one after another as follows:

    index:        0,1...  3,4,5...    8,9,10...       15,16...          
    c_kernel[]: | . . . | . . . . . | . . . . . . . | . . . . . . . . . |
    radius:         1         2             3                 4

The location of the kernel of radius R can be obtained as (R-1)*(R+1)







