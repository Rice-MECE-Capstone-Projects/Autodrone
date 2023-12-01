#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
// #include <fstream>
#include "yolo.h"
#include <geometry_msgs/Point.h>

static const std::string OPENCV_WINDOW = "Image window";

class ImageConverter {
    ros::NodeHandle nh;
    image_transport::Subscriber image_sub;
    cv::dnn::Net net;
    std::vector<std::string> class_names;
    ros::Publisher position_pub;

public:

    ImageConverter(std::string model_path, std::string class_path) {
        image_transport::ImageTransport it(nh);
        class_names = load_class_list(class_path);
        load_net(net, model_path, true);
        image_sub = it.subscribe("/webcam1/image_raw", 1, &ImageConverter::imageCallback, this);
        position_pub = nh.advertise<geometry_msgs::Point>("/person_position", 10);
    }

    ~ImageConverter() {
        cv::destroyWindow(OPENCV_WINDOW);
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            // return 1;
        }

        cv::Mat image = cv_ptr->image;

        std::vector<Detection> detections;
        detect(image, net, detections, class_names);


        cv::Point person_center(0, 0);

        //draw bounding boxes on the image
        int count = 0;

        for (const auto& detection: detections) {
            if (class_names[detection.class_id] == "person") {
                person_center.x = detection.box.x + detection.box.width / 2;
                person_center.y = detection.box.y + detection.box.height / 2;
                count++;
            }

            cv::rectangle(image, detection.box, colors[detection.class_id % colors.size()], 3);
            std::string label = class_names[detection.class_id] + ": " + std::to_string(detection.confidence);
            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0, 1, &baseLine);
            cv::putText(image, label, cv::Point(detection.box.x, detection.box.y - labelSize.height-5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, colors[detection.class_id % colors.size()], 1);
        }
        std::string headcount = "Headcount: " + std::to_string(count);
        cv::putText(image, headcount, cv::Point(20,20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);

        cv::Point frame_center(image.cols / 2, image.rows / 2);
        cv::Point relative_position = person_center - frame_center;

        // Publish the relative position
        geometry_msgs::Point pos_msg;
        pos_msg.x = static_cast<double>(relative_position.x) / image.cols;
        pos_msg.y = static_cast<double>(relative_position.y) / image.rows;
        pos_msg.z = 0.0;
        position_pub.publish(pos_msg);


        cv::imshow(OPENCV_WINDOW, image);
        cv::waitKey(1);

        

        ROS_INFO("[Image Subscriber] Image Received");
    }
};


int main(int argc, char** argv) {
    std::cout << "Hello, OpenCV version "<< CV_VERSION << std::endl;
    ros::init(argc, argv, "image_subscriber_node");
    std::string model_path = "/home/rev/Autodrone_ws/src/drone_control/models/yolov5s.onnx";
    std::string class_path = "/home/rev/Autodrone_ws/src/drone_control/models/classes.txt";
    ImageConverter ic(model_path, class_path);
    ros::spin();

    return 0;
}