#ifndef YOLO_H
#define YOLO_H

#include <opencv2/opencv.hpp>
#include <fstream>

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};

std::vector<std::string> load_class_list(std::string path);
void load_net(cv::dnn::Net &net, std::string path, bool is_cuda);
void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className);

#endif
