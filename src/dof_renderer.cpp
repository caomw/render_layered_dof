
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

    float *h_input_image, *h_buffer_image, *h_output_image, *h_depth_map;
    float *d_input_image, *d_output_image, *d_buffer_image, *d_depth_map;

    Mat input_image_float;
    cvtColor(input_image_, input_image_float, CV_BGR2GRAY);
    input_image_float.convertTo(input_image_float, CV_32FC1);
    input_image_float *= 1./255;

    // input_image_float = input_image_float(Range(0,576),Range(0,896));

    // if (!input_image_float.isContinuous())
    // 	cout << "not continuous\n";

    // Mat input_image_cropped = input_image_float.clone();

    // if (!input_image_cropped.isContinuous()){
    // 	cout << "not continuous\n";
    // }

    int padding = ceil((float)image_width_/32)*32 - image_width_;
    Mat padded_input_image, padded_depth_map;
    copyMakeBorder(input_image_float, padded_input_image, 0, 0, 0, padding, BORDER_CONSTANT, 0);
    copyMakeBorder(depth_map_, padded_depth_map, 0, 0, 0, padding, BORDER_CONSTANT, 0);

    int padded_image_width = padded_input_image.cols;
    int padded_image_height = padded_input_image.rows;

    h_input_image = padded_input_image.ptr<float>(0);
    h_buffer_image    = (float *)malloc(padded_image_width * padded_image_height * sizeof(float));
    h_output_image    = (float *)malloc(padded_image_width * padded_image_height * sizeof(float));
    h_depth_map = padded_depth_map.ptr<float>(0);

    vector<Mat> gaussianKernel;
    vector<float*> h_kernel;
    for(int i = 0; i < filter_sizes_.size(); ++i){
    	gaussianKernel.push_back(getGaussianKernel(filter_sizes_[i], -1, CV_32F));
    	h_kernel.push_back(gaussianKernel[i].ptr<float>(0));
        // debugPrintMat<float>(gaussianKernel[i],"kernel");
        checkCudaErrors(copyKernel(h_kernel[i], i));
    }
    // testKernel();

    printf("\nAllocating memory on GPU ...\n\n");

    checkCudaErrors(cudaMalloc((void **)&d_input_image,   padded_image_width * padded_image_height * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_output_image,  padded_image_width * padded_image_height * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_buffer_image , padded_image_width * padded_image_height * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_depth_map , padded_image_width * padded_image_height * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_input_image, h_input_image, padded_image_width * padded_image_height * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_depth_map, h_depth_map, padded_image_width * padded_image_height * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_output_image, 0, padded_image_width * padded_image_height * sizeof(float)));

    printf("Running row convolution on GPU ... \n");

    checkCudaErrors(cudaDeviceSynchronize());
    // int kernel_ind = 2;
    GpuConvolveSeparableRows(
        d_buffer_image,
        d_input_image,
        d_depth_map,
        image_width_,
       image_height_,
       depth_of_focus
    );
 
    printf("Running column convolution on GPU ... \n");

    GpuConvolveSeparableCols(
        d_output_image,
        d_buffer_image,
        d_depth_map,
        image_width_,
       image_height_,
       depth_of_focus
    );

    checkCudaErrors(cudaDeviceSynchronize());

	printf("\nCopying results ...\n\n");
    checkCudaErrors(cudaMemcpy(h_output_image, d_output_image, padded_image_width * padded_image_height * sizeof(float), cudaMemcpyDeviceToHost));
    Mat output_image_gray(padded_image_height, padded_image_width, CV_32FC1, h_output_image);
    imshow("output_image_gray", output_image_gray);


    checkCudaErrors(cudaFree(d_buffer_image));
    checkCudaErrors(cudaFree(d_output_image));
    checkCudaErrors(cudaFree(d_input_image));

}

int DepthOfFieldRenderer::Run(InputArray _input_image, InputArray _depth_map, OutputArray _output_image){
	input_image_ = _input_image.getMat();
	depth_map_ = _depth_map.getMat();
	output_image_ = _output_image.getMat();
	image_width_ = input_image_.cols;
	image_height_ = input_image_.rows;

	/* Convert depth map to single channel, 0 to 1 floating point image*/
	PreprocessDepthMap();

	string window_name = "Click to select a point in the image, press any key on the keyboard to exit...";
	namedWindow(window_name);
	imshow(window_name, input_image_);
	setMouseCallback(window_name, ProcessSelectedPoint, this);

    waitKey(0);
	return 0;
}