#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "../include/dof_renderer.hpp"
#include "../include/debugging_functions.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	/* Load the image and convert to Lab colour space. */
    Mat input_image = imread(argv[1]);
    Mat depth_map = imread(argv[2], -1);
    Mat output_image;

    DepthOfFieldRenderer dof_renderer(0.1);
    dof_renderer.Run(input_image, depth_map, output_image);

    // dof_renderer.run(input_image, depth_map, output_image);

    // imshow("output_image", output_image);
    return 0;
    // return dof_renderer.exec();
}

