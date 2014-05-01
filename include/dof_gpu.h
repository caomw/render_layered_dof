#ifndef DOF_GPU_H
#define DOF_GPU_H

/* dof_gpu.h.
 *
 * Written by: Srivatsan Varadharajan.
 *
 * 
 * This file contains the interface to the CUDA functions ,
 * for rendering depth of field, based on Gaussian blurring
 * using separable convolution, with depth-dependent kernel size.
*/

extern "C" void copyKernel(float *kernel_coefficients, int kernel_index);
extern "C" void testKernel();
extern "C" void GpuConvolveSeparableRows(float *d_dst, float *d_src, float* d_depth_map, int image_width, int image_height, float focus_depth);
extern "C" void GpuConvolveSeparableCols(float *d_dst, float *d_src, float* d_depth_map, int image_width, int image_height, float focus_depth);



#endif //DOF_GPU_H