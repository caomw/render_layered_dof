
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
	for(int i = 0; i*kDepthStep < 1.0; ++i)
		circle_of_confusion_radius_.push_back(i);
}


DepthOfFieldRenderer::~DepthOfFieldRenderer(){
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
	if(input_image_.empty())
		cout << "Input image has not been set. Set input image by calling set_input_image() first!\n";
	if(depth_map_.empty())
		cout << "Depth map has not been set. Set input image by calling set_input_image() first!\n";

	float depth_of_focus = depth_map_.at<float>(origin);
	
}

int DepthOfFieldRenderer::Run(InputArray _input_image, InputArray _depth_map, OutputArray _output_image){
	input_image_ = _input_image.getMat();
	depth_map_ = _depth_map.getMat();
	output_image_ = _output_image.getMat();

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