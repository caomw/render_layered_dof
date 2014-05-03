
/* dof_gpu.cu.
 *
 * Written by: Srivatsan Varadharajan.
 *
 * 
 * This file contains the definition of the CUDA functions ,
 * for rendering depth of field, based on Gaussian blurring
 * using separable convolution, with depth-dependent kernel size.
*/

#include <helper_cuda.h>
#include <assert.h>
#include <math.h>

#define MAX_KERNEL_RADIUS 9
#define NUM_KERNELS MAX_KERNEL_RADIUS
#define KERNEL_LENGTH(x) (2 * x + 1)
#define MAX_KERNEL_LENGTH KERNEL_LENGTH(MAX_KERNEL_RADIUS)

#define ROW_TILE_WIDTH 32
#define ROW_TILE_HEIGHT 4
#define ROW_TILES_IN_BLOCK 20
#define ROW_BLOCK_WIDTH ROW_TILES_IN_BLOCK * ROW_TILE_WIDTH
#define ROW_BLOCK_HEIGHT ROW_TILE_HEIGHT

__constant__ float c_kernel[NUM_KERNELS * (NUM_KERNELS + 2)];

extern "C" void copyKernel(float *kernel_coefficients, int kernel_index){
	int kernel_radius = kernel_index + 1;
	cudaMemcpyToSymbol(
        c_kernel, 
        kernel_coefficients, 
        KERNEL_LENGTH(kernel_radius) * sizeof(float),
        kernel_index * (kernel_index + 2) * sizeof(float));
}

extern "C" void testKernel(){
    float h_kernel_data[NUM_KERNELS*(NUM_KERNELS+2)];
    cudaMemcpyFromSymbol(h_kernel_data, c_kernel, NUM_KERNELS*(NUM_KERNELS+2) * sizeof(float));
    int i,j;
    for(i = 0; i < NUM_KERNELS; ++i){
        printf("%d: ",i);
        for(j = 0; j < 2*i+3; ++j)
            printf("%f ", h_kernel_data[i*(i+2)+j]);
        printf("\n");
    }
}

__global__ void convolveSeparableRowsKernel(unsigned char* d_dst, unsigned char* d_src, float* d_depth_map, int image_width, int image_height, size_t pitch, size_t depth_map_pitch, float focus_depth){
    __shared__ unsigned char s_data[ROW_TILE_HEIGHT][(ROW_BLOCK_WIDTH + 2 * ROW_TILE_WIDTH)];
    int x = threadIdx.x, y = threadIdx.y;
    int x_image, y_image, x_s, y_s;

    x_image = blockIdx.x * ROW_BLOCK_WIDTH - ROW_TILE_WIDTH + x;
    y_image = blockIdx.y * ROW_BLOCK_HEIGHT + y;
    x_s = x; y_s = y;
    s_data[y_s][x_s] = x_image < 0 ? 0 : d_src[y_image * pitch + x_image];

    for(int i = 1; i < (ROW_TILES_IN_BLOCK + 2); ++i){
        x_s += ROW_TILE_WIDTH;
        x_image += ROW_TILE_WIDTH;
        s_data[y_s][x_s] = x_image >= image_width*3 ? 0 : d_src[y_image * pitch + x_image];
    }
    __syncthreads();

    x_image = blockIdx.x * ROW_BLOCK_WIDTH + x;
    x_s = ROW_TILE_WIDTH + x;

    for(int i = 0; i < ROW_TILES_IN_BLOCK; ++i){
        if (x_image < image_width*3){
            int kernel_radius = (int)floor(10*fabs(d_depth_map[y_image * depth_map_pitch/sizeof(float) + x_image/3] - focus_depth));
            if (kernel_radius > 0){
                float sum = 0;
                int kernel_start = kernel_radius * kernel_radius - 1;
                int kernel_mid = kernel_start + kernel_radius;
                for(int j = -kernel_radius; j <= kernel_radius; ++j){
                    sum += (float)s_data[y_s][x_s + j*3] *  c_kernel[kernel_mid + j];
                }
                d_dst[y_image * pitch + x_image] = (unsigned char)sum;
            }
            else{
                d_dst[y_image * pitch + x_image] = s_data[y_s][x_s];
            }
        }
        x_s += ROW_TILE_WIDTH;
        x_image += ROW_TILE_WIDTH;
    }
}

extern "C" void GpuConvolveSeparableRows(unsigned char *d_dst, unsigned char *d_src, float* d_depth_map, int image_width, int image_height, size_t pitch, size_t depth_map_pitch, float focus_depth){
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int block_grid_width = (int)ceil((float)image_width*3 / (ROW_TILES_IN_BLOCK * ROW_TILE_WIDTH));
    int block_grid_height = (int)ceil((float)image_height / (ROW_TILE_HEIGHT));
    printf("block_grid_width:%d block_grid_height:%d\n", block_grid_width, block_grid_height);
    printf("image_width:%d image_height:%d\n", image_width, image_height);
    dim3 blocks(block_grid_width, block_grid_height);
    dim3 threads(ROW_TILE_WIDTH, ROW_TILE_HEIGHT);
    cudaEventRecord(start, 0);
    
    convolveSeparableRowsKernel<<<blocks, threads>>>(
        d_dst,
        d_src,
        d_depth_map,
        image_width,
        image_height,
        pitch,
        depth_map_pitch,
        focus_depth
    );
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf ("Time for the kernel: %f ms\n", time);

    getLastCudaError("convolveSeparableRowsKernel() execution failed\n");
}

#define COL_TILE_WIDTH 16
#define COL_TILE_HEIGHT 10
#define COL_VTILES_IN_BLOCK 8
#define COL_HTILES_IN_BLOCK 2
#define COL_BLOCK_WIDTH COL_TILE_WIDTH * COL_HTILES_IN_BLOCK
#define COL_BLOCK_HEIGHT COL_TILE_HEIGHT * COL_VTILES_IN_BLOCK 

__global__ void convolveSeparableColsKernel(unsigned char* d_dst, unsigned char* d_src, float* d_depth_map, int image_width, int image_height, size_t pitch, size_t depth_map_pitch, float focus_depth){
    __shared__ unsigned char s_data[COL_BLOCK_HEIGHT + 2 * COL_TILE_HEIGHT][COL_BLOCK_WIDTH];
    int x = threadIdx.x, y = threadIdx.y;
    int x_image, y_image, x_s, y_s;

    x_image = (blockIdx.x * COL_BLOCK_WIDTH) + x;
    y_image = blockIdx.y * COL_BLOCK_HEIGHT - COL_TILE_HEIGHT + y;
    x_s = x; y_s = y;

    for(int k = 0; k < COL_HTILES_IN_BLOCK; ++k){
        if (x_image < image_width*3){
            s_data[y_s][x_s] = y_image < 0 ? 0 : d_src[y_image * pitch + x_image];
            x_image += COL_TILE_WIDTH;
            x_s += COL_TILE_WIDTH;
        }
    }
    for(int i = 1; i < COL_VTILES_IN_BLOCK + 2; ++i){
        x_image = (blockIdx.x * COL_BLOCK_WIDTH) + x;
        x_s = x;
        y_s += COL_TILE_HEIGHT;
        y_image += COL_TILE_HEIGHT;
        for(int k = 0; k < COL_HTILES_IN_BLOCK; ++k){
            if (x_image < image_width*3){
                s_data[y_s][x_s] = y_image < image_height ? d_src[y_image * pitch + x_image] : 0;
                x_image += COL_TILE_WIDTH;
                x_s += COL_TILE_WIDTH;
            }
        }
    }
    __syncthreads();
    x_image = (blockIdx.x * COL_BLOCK_WIDTH) + x;
    x_s = x; 
    for(int k = 0; k < COL_HTILES_IN_BLOCK; ++k){
        if (x_image < image_width*3){
            y_image = blockIdx.y * COL_BLOCK_HEIGHT - COL_TILE_HEIGHT + y;
            y_s = y;

            for(int i = 0; i < COL_VTILES_IN_BLOCK; ++i){
                y_s += COL_TILE_HEIGHT;
                y_image += COL_TILE_HEIGHT;
                if (y_image < image_height){
                    int kernel_radius = (int)floor(10*fabs(d_depth_map[y_image * depth_map_pitch/sizeof(float) + x_image/3] - focus_depth));
                    if (kernel_radius > 0){
                        float sum = 0;
                        int kernel_start = kernel_radius * kernel_radius - 1;
                        int kernel_mid = kernel_start + kernel_radius;
                        for(int j = -kernel_radius; j <= kernel_radius; ++j)
                            sum += (float)s_data[y_s+j][x_s]*c_kernel[kernel_mid + j];
                        d_dst[y_image * pitch + x_image] = (unsigned char)sum;
                    }
                    else
                        d_dst[y_image * pitch + x_image] = s_data[y_s][x_s];
                }
            }
        }
        x_image += COL_TILE_WIDTH;
        x_s += COL_TILE_WIDTH;
    }
}

extern "C" void GpuConvolveSeparableCols(unsigned char *d_dst, unsigned char *d_src, float* d_depth_map, int image_width, int image_height, size_t pitch, size_t depth_map_pitch, float focus_depth){
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int block_grid_width = (int)ceil((float)image_width*3 / (COL_TILE_WIDTH * COL_HTILES_IN_BLOCK));
    int block_grid_height = (int)ceil((float)image_height / ( COL_VTILES_IN_BLOCK * COL_TILE_HEIGHT));
    printf("block_grid_width:%d block_grid_height:%d\n", block_grid_width, block_grid_height);
    printf("image_width:%d image_height:%d\n", image_width, image_height);
    dim3 blocks(block_grid_width, block_grid_height);
    dim3 threads(COL_TILE_WIDTH, COL_TILE_HEIGHT);
    cudaEventRecord(start, 0);
    convolveSeparableColsKernel<<<blocks, threads>>>(
        d_dst,
        d_src,
        d_depth_map,
        image_width,
        image_height,
        pitch,
        depth_map_pitch,
        focus_depth
    );
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf ("Time for the kernel: %f ms\n", time);

    getLastCudaError("convolveSeparableColsKernel() execution failed\n");
}
