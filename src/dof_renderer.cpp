
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

    float *h_Input, *h_Buffer, *h_OutputCPU, *h_OutputGPU;
    // vector<float*>h_Kernel(filter_sizes_.size());
    float *d_Input, *d_Output, *d_Buffer;

    Mat input_image_float;
    cvtColor(input_image_, input_image_float, CV_BGR2GRAY);
    input_image_float.convertTo(input_image_float, CV_32FC1);
    input_image_float *= 1./255;

    input_image_float = input_image_float(Range(0,576),Range(0,896));

    if (!input_image_float.isContinuous())
    	cout << "not continuous\n";

    Mat input_image_cropped = input_image_float.clone();

    if (!input_image_cropped.isContinuous()){
    	cout << "not continuous\n";
    }

    image_width_ = input_image_cropped.cols;
    image_height_ = input_image_cropped.rows;

    cout << input_image_cropped.size() << "\n";
    h_Input = input_image_cropped.ptr<float>(0);

    h_Buffer    = (float *)malloc(image_width_ * image_height_ * sizeof(float));
    h_OutputGPU    = (float *)malloc(image_width_ * image_height_ * sizeof(float));

    vector<Mat> gaussianKernel;
    vector<float*> h_Kernel;
    for(int i = 0; i < filter_sizes_.size(); ++i){
    	gaussianKernel.push_back(getGaussianKernel(filter_sizes_[i], -1, CV_32F));
    	h_Kernel.push_back(gaussianKernel[i].ptr<float>(0));
    }

    printf("Allocating and initializing CUDA arrays...\n");
    checkCudaErrors(cudaMalloc((void **)&d_Input,   image_width_ * image_height_ * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Output,  image_width_ * image_height_ * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Buffer , image_width_ * image_height_ * sizeof(float)));

    // Testing GPU convolution with single kernel first
    setConvolutionKernel(h_Kernel[5]);

    checkCudaErrors(cudaMemcpy(d_Input, h_Input, image_width_ * image_height_ * sizeof(float), cudaMemcpyHostToDevice));

    printf("Running GPU convolution... \n");

    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    convolutionRowsGPU(
        d_Buffer,
        d_Input,
        image_width_,
       image_height_ 
    );

    convolutionColumnsGPU(
        d_Output,
        d_Buffer,
        image_width_,
       image_height_ 
    );

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);

	printf("\nReading back GPU results...\n\n");
    checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, image_width_ * image_height_ * sizeof(float), cudaMemcpyDeviceToHost));
    Mat output_image_gray(image_height_, image_width_, CV_32FC1, h_OutputGPU);
    imshow("output_image_gray", output_image_gray);

    cout << gaussianKernel[5].size() << "\n";

    checkCudaErrors(cudaFree(d_Buffer));
    checkCudaErrors(cudaFree(d_Output));
    checkCudaErrors(cudaFree(d_Input));

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



    // Mat histogram;
	// int histSize = 10;
	// float range[] = { 0, 1 } ;
	// const float* histRange = { range };
	// bool uniform = true; bool accumulate = false;
    // calcHist(&depth_map_, 1, 0, Mat(), histogram, 1, &histSize, &histRange, uniform, accumulate);
    // debugPrintMat<float>(histogram, "histogram");
    // namedWindow("depth_map_");
    // imshow("depth_map_", depth_map_);
    // setMouseCallback("depth_map_", PrintPixelValue, &depth_map_);




    // imshow("input_image_", input_image_);
    // imshow("depth_map", depth_map);

    waitKey(0);
	return 0;
}