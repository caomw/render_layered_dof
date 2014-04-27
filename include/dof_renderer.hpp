#ifndef DOF_RENDERER_H
#define DOF_RENDERER_H


/* dof_renderer.hpp.
 *
 * Written by: Srivatsan Varadharajan.
 *
 * 
 * This file contains the interface to the DepthOfFieldRenderer class,
 * which is an implementation of a simple layered depth of field
 * rendering algorithm based on Gaussian blurring.
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/debugging_functions.hpp"

using namespace cv;
using namespace std;

class DepthOfFieldRenderer{
	Mat depth_map_, input_image_;
	Mat output_image_;
	vector<int> circle_of_confusion_radius_;
	float kDepthStep;
	void PreprocessDepthMap();
public:
	DepthOfFieldRenderer(float _depth_step);
	~DepthOfFieldRenderer();		
	void set_input_image(InputArray _input_image);
	void set_depth_map(InputArray _depth_map);
	void RenderDoF(Point origin);
	int Run(InputArray _input_image, InputArray _depth_map, OutputArray _output_image);
};

#endif //DOF_RENDERER_H