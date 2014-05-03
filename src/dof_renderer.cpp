
/* dof_renderer.hpp.
 *
 * Written by: Srivatsan Varadharajan.
 *
 * 
 * This file contains the source of the DepthOfFieldRenderer class,
 * which is an implementation of a simple layered depth of field
 * rendering algorithm based on Gaussian blurring.
*/

#include "../include/dof_renderer.hpp"

static void PrintPixelValue(int event, int x, int y, int, void* userdata){
	Mat* image = (Mat*)userdata;
    if( event != EVENT_LBUTTONDOWN )
        return;
    Point seed = Point(x,y);
    cout << image->at<float>(seed) << "\n";
}

static void ProcessSelectedPoint(int event, int x, int y, int, void* userdata){
	DepthOfFieldRenderer* dof_renderer = (DepthOfFieldRenderer*)userdata;
    if( event != EVENT_LBUTTONDOWN )
        return;
    dof_renderer->RenderDoF(Point(x,y));
}

DepthOfFieldRenderer::DepthOfFieldRenderer(float _depth_step):
kDepthStep(_depth_step){
	for(int radius = 1; radius*kDepthStep < 1.0; ++radius)
		filter_sizes_.push_back(2*radius+1);
	const char* empty;
	findCudaDevice(0, &empty);
}


DepthOfFieldRenderer::~DepthOfFieldRenderer(){
    cudaDeviceReset();
}

void DepthOfFieldRenderer::PreprocessDepthMap(){
	if (depth_map_.channels() > 1){
    	vector<Mat> depth_planes;
    	depth_map_.convertTo(depth_map_, CV_32FC3);
    	split(depth_map_, depth_planes);
		multiply(depth_planes[3], depth_planes[0], depth_map_);
    }
    normalize(depth_map_, depth_map_, 0, 1, NORM_MINMAX);
    MatIterator_<float> itr = depth_map_.begin<float>();
    while (itr != depth_map_.end<float>()){
    	if (*itr < 0)
    		*itr = 0;
    	++itr;
    }
}

void DepthOfFieldRenderer::set_input_image(InputArray _input_image){
	input_image_ = _input_image.getMat();
}

void DepthOfFieldRenderer::set_depth_map(InputArray _depth_map){
	depth_map_ = _depth_map.getMat();
}

void DepthOfFieldRenderer::RenderDoF(Point origin){
	if(input_image_.empty()){
		cout << "Input image has not been set. Set input image by calling set_input_image() first!\n";
		return;
	}
	if(depth_map_.empty()){
		cout << "Depth map has not been set. Set input image by calling set_input_image() first!\n";
		return;
	}
	float depth_of_focus = depth_map_.at<float>(origin);
	cout << "Focusing on point " << origin << " at depth " << depth_of_focus << " ...\n";
	cout << "Rendering depth of field...\n";

    unsigned char *h_input_image, *h_output_image;
    unsigned char *d_input_image, *d_output_image, *d_buffer_image;
    float *h_depth_map, *d_depth_map;

    //Simple trick to ensure input_image_ is stored in a continuous chunk of memory
    while (!input_image_.isContinuous())
        input_image_ = input_image_.clone();

    //Generate Gaussian Kernels of radii 1, 2, 3, ... 9 and copy them to the device's constant memory
    vector<Mat> gaussianKernel;
    vector<float*> h_kernel;
    for(int i = 0; i < filter_sizes_.size(); ++i){
    	gaussianKernel.push_back(getGaussianKernel(filter_sizes_[i], -1, CV_32F));
    	h_kernel.push_back(gaussianKernel[i].ptr<float>(0));
        copyKernel(h_kernel[i], i);
    }
    // testKernel();

    printf("\nAllocating memory on GPU ...\n\n");

    size_t pitch; // Adjusted width of image (in bytes) to ensure alignment in GPU memory

    // Copy input image to device global memory
    h_input_image = (unsigned char*)input_image_.ptr<Vec3b>(0);
    checkCudaErrors(cudaMallocPitch(&d_input_image, &pitch, image_width_ * sizeof(Vec3b), image_height_));
    checkCudaErrors(cudaMemcpy2D(d_input_image, pitch, h_input_image, image_width_*sizeof(Vec3b), image_width_*sizeof(Vec3b), image_height_, cudaMemcpyHostToDevice));
    printf("pitch: %lu\n", pitch);

    // Copy depth map into device global memory
    size_t depth_map_pitch;
    h_depth_map = depth_map_.ptr<float>(0);
    checkCudaErrors(cudaMallocPitch(&d_depth_map, &depth_map_pitch, image_width_ * sizeof(float), image_height_));
    checkCudaErrors(cudaMemcpy2D(d_depth_map, depth_map_pitch, h_depth_map, image_width_*sizeof(float), image_width_*sizeof(float), image_height_, cudaMemcpyHostToDevice));
    printf("depth map pitch: %lu\n", depth_map_pitch);

    // Allocate device global memory and host memory for output image
    h_output_image    = (unsigned char*)malloc(image_width_ * image_height_ * sizeof(Vec3b));
    checkCudaErrors(cudaMalloc((void **)&d_output_image,  pitch * image_height_));
    checkCudaErrors(cudaMemset(d_output_image, 0, pitch * image_height_));

    // Allocate device memory for buffer
    checkCudaErrors(cudaMalloc((void **)&d_buffer_image , pitch * image_height_));

    printf("Running row convolution on GPU ... \n");
    checkCudaErrors(cudaDeviceSynchronize());
    GpuConvolveSeparableRows(d_buffer_image, d_input_image, d_depth_map, image_width_, image_height_, pitch, depth_map_pitch, depth_of_focus);
    // GpuConvolveSeparableRows(d_output_image, d_input_image, d_depth_map, image_width_, image_height_, pitch, depth_map_pitch, depth_of_focus);
 
    printf("Running column convolution on GPU ... \n");
    GpuConvolveSeparableCols(d_output_image, d_buffer_image, d_depth_map, image_width_, image_height_, pitch, depth_map_pitch, depth_of_focus);
    checkCudaErrors(cudaDeviceSynchronize());

	printf("\nCopying results ...\n\n");
    checkCudaErrors(cudaMemcpy2D(h_output_image, image_width_ * sizeof(Vec3b), d_output_image, pitch, image_width_ * sizeof(Vec3b), image_height_, cudaMemcpyDeviceToHost));

    Mat output_image_color(input_image_.size(), input_image_.type(), h_output_image);
    output_image_ = output_image_color;

    imshow("output_image", output_image_);
    checkCudaErrors(cudaFree(d_buffer_image));
    checkCudaErrors(cudaFree(d_output_image));
    checkCudaErrors(cudaFree(d_input_image));


}

int DepthOfFieldRenderer::Run(InputArray _input_image, InputArray _depth_map, OutputArray _output_image){
	input_image_ = _input_image.getMat();
	depth_map_ = _depth_map.getMat();
	image_width_ = input_image_.cols;
	image_height_ = input_image_.rows;

	/* Convert depth map to single channel, 0 to 1 floating point image*/
	PreprocessDepthMap();

	string window_name = "Click to select a point in the image, press any key on the keyboard to exit...";
	namedWindow(window_name);
	imshow(window_name, input_image_);
	setMouseCallback(window_name, ProcessSelectedPoint, this);
    waitKey(0);
    output_image_.copyTo(_output_image);
	return 0;
}