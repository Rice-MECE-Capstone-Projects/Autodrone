#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

class Depth {
private:
	cv::Size imgSize;
	cv::Mat mapxl, mapyl, mapxr, mapyr;
	cv::Rect roiL, roiR;

	cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();
	int numDisparities = 128, minDisparity = 4;
	cv::Mat disparity;

public:
	Depth(std::string path);
	cv::Mat calibrateImage(cv::Mat frame, bool left);
	void createStereoMatch(int numDisparities, int minDisparity, int blockSize, int speckleRange, int speckleWindowSize);
	cv::Mat disparityMap(cv::Mat imgl, cv::Mat imgr);
};

Depth::Depth(std::string path) {
	// read in calibration detail
	cv::FileStorage fs(path, cv::FileStorage::READ);
	fs["imgSize"] >> this->imgSize;
	fs["mapxl"] >> this->mapxl;
	fs["mapyl"] >> this->mapyl;
	fs["mapxr"] >> this->mapxr;
	fs["mapyr"] >> this->mapyr;
	fs["roiL"] >> this->roiL;
	fs["roiR"] >> this->roiR;
}

cv::Mat Depth::calibrateImage(cv::Mat frame, bool left) {
	cv::Mat res;
	if (left)
		cv::remap(frame, res, mapxl, mapyl, cv::INTER_LINEAR);
	else
		cv::remap(frame, res, mapxr, mapyr, cv::INTER_LINEAR);
	return res;
}

void Depth::createStereoMatch(int numDisparities = 128, int minDisparity = 4, int blockSize = 17, int speckleRange = 16, int speckleWindowSize = 45) {
	this->numDisparities = numDisparities;
	this->minDisparity = minDisparity;
	//stereo->create(numDisparities, blockSize);
	stereo->setNumDisparities(numDisparities);
	stereo->setBlockSize(blockSize);
	stereo->setMinDisparity(minDisparity);
	stereo->setSpeckleRange(speckleRange);
	stereo->setSpeckleWindowSize(speckleWindowSize);
	stereo->setROI1(roiL);
	stereo->setROI2(roiR);
}

cv::Mat Depth::disparityMap(cv::Mat imgl, cv::Mat imgr) {
	cv::Mat grayl, grayr;
	cv::cvtColor(imgl, grayl, cv::COLOR_BGR2GRAY);
	cv::cvtColor(imgr, grayr, cv::COLOR_BGR2GRAY);

	stereo->compute(grayl, grayr, this->disparity);
	this->disparity.convertTo(this->disparity, CV_32F, 1.0);
	this->disparity = (this->disparity / 16.0f - (float)this->minDisparity + 1.0f) / (float)this->numDisparities;

	return this->disparity;
}	