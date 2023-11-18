// opencvtest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "depth.h"

#define CAMERA_HEIGHT 720
#define CAMERA_WIDTH 1920 // left right together

int CHESSBOARD_SIZE[2]{ 5, 8 };

int main() {
    
    int key;
    int i = 0;
    int h, w, w2;
    cv::Mat frame, im0, im1;

    cv::VideoCapture cap(1);

    if (!cap.isOpened()) {
        std::cout << "Cannot open file\n";
        std::cin.get();
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH);

    cap.read(frame);
    
    w2 = frame.cols;
    h = frame.rows;
    w = (int)(w2 / 2);

    /*// capture images for calibration
    while (true) {    
        cap.read(frame);

        im0 = frame(cv::Range(0, h), cv::Range(0, w));
        im1 = frame(cv::Range(0, h), cv::Range(w, w2));

        cv::imshow("left", im0);
        cv::imshow("right", im1);

        key = cv::waitKey(1);
        if (key == 'c') {
            path_left = "data/left" + std::to_string(i) + ".jpg";
            path_right = "data/right" + std::to_string(i) + ".jpg";
            cv::imwrite(path_left, im0);
            cv::imwrite(path_right, im1);
            i++;
        }
        else if (key == 27) { // esc
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();*/

    /*// Calibrate
    std::vector<std::vector<cv::Point3f>> objpoints;
    std::vector<std::vector<cv::Point2f>> imgpoints1, imgpoints2;
    cv::Size imgSize;
    
    std::vector<cv::Point3f> objp;
    for (int i{ 0 }; i < CHESSBOARD_SIZE[1]; i++) {
        for (int j{ 0 }; j < CHESSBOARD_SIZE[0]; j++) {
            objp.push_back(cv::Point3f(j, i, 0));
        }
    }

    std::string path_left, path_right;
    std::vector<cv::String> imagesL, imagesR;
    path_left = "./data/left*.jpg";
    path_right = "./data/right*.jpg";

    cv::glob(path_left, imagesL);
    cv::glob(path_right, imagesR);

    cv::Mat imgl, imgr, grayl, grayr;
    std::vector<cv::Point2f> corners1, corners2;
    bool success1 = false, success2 =  false;
    cv::TermCriteria criteria((cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS), 30, 0.001);


    for (int i{0}; i < imagesL.size(); i++) {
        imgl = cv::imread(imagesL[i]);
        cv::cvtColor(imgl, grayl, cv::COLOR_BGR2GRAY);
        imgr = cv::imread(imagesR[i]);
        cv::cvtColor(imgr, grayr, cv::COLOR_BGR2GRAY);

        success1 = cv::findChessboardCorners(grayl, cv::Size(CHESSBOARD_SIZE[0], CHESSBOARD_SIZE[1]), corners1, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
        success2 = cv::findChessboardCorners(grayr, cv::Size(CHESSBOARD_SIZE[0], CHESSBOARD_SIZE[1]), corners2, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
        
        if (success1) {
            cv::cornerSubPix(grayl, corners1, cv::Size(11, 11), cv::Size(-1, -1), criteria);
            cv::drawChessboardCorners(imgl, cv::Size(CHESSBOARD_SIZE[0], CHESSBOARD_SIZE[1]), corners1, success1);
        }
        if (success2) {
            cv::cornerSubPix(grayr, corners2, cv::Size(11, 11), cv::Size(-1, -1), criteria);
            cv::drawChessboardCorners(imgr, cv::Size(CHESSBOARD_SIZE[0], CHESSBOARD_SIZE[1]), corners2, success2);
        }

        if (success1 && success2) {
            std::cout << i << ". Found corners!\n";
            imgpoints1.push_back(corners1);
            imgpoints2.push_back(corners2);
            objpoints.push_back(objp);
        }
        cv::imshow("img", imgl);
        cv::waitKey(1);
        cv::imshow("img", imgr);
        cv::waitKey(1);
    }
    cv::destroyAllWindows();
    
    imgSize = imgl.size();

    // calibrate camera
    cv::Mat mtxL, distL, mtxR, distR, R, T, E, F, recL, recR, projL, projR, Q;
    std::vector<cv::Mat> rvecs, tvecs;
    cv::Rect roiL, roiR;

    cv::calibrateCamera(objpoints, imgpoints1, imgSize, mtxL, distL, rvecs, tvecs);
    cv::calibrateCamera(objpoints, imgpoints2, imgSize, mtxR, distR, rvecs, tvecs);

    cv::stereoCalibrate(objpoints, imgpoints1, imgpoints2, mtxL, distL, mtxR, distR, imgSize, R, T, E, F);

    // rectify
    cv::stereoRectify(mtxL, distL, mtxR, distR, imgSize, R, T, recL, recR, projL, projR, Q, 1024, 0.1, imgSize, &roiL, &roiR);

    // undistort rectified parameter
    cv::Mat mapxl, mapyl, mapxr, mapyr;
    cv::initUndistortRectifyMap(mtxL, distL, recL, projL, imgSize, CV_32FC1, mapxl, mapyl);
    cv::initUndistortRectifyMap(mtxR, distR, recR, projR, imgSize, CV_32FC1, mapxr, mapyr);

    // save
    cv::FileStorage fs("calibration", cv::FileStorage::WRITE);
    fs << "imgSize" << imgSize;
    fs << "mapxl" << mapxl;
    fs << "mapyl" << mapyl;
    fs << "mapxr" << mapxr;
    fs << "mapyr" << mapyr;
    fs << "roiL" << roiL;
    fs << "roiR" << roiR;

    // draw for reference
    cv::Mat dstl, dstr;
    imgl = cv::imread("./data/left0.jpg");
    imgr = cv::imread("./data/right0.jpg");
    cv::remap(imgl, dstl, mapxl, mapyl, cv::INTER_LINEAR);
    cv::remap(imgr, dstr, mapxr, mapyr, cv::INTER_LINEAR);

    cv::imshow("left", dstl);
    cv::imshow("right", dstr);

    while (true) {
        if (cv::waitKey(1) == 27) break;
    }*/
    
    /*// Depth
    cv::Size imgSize;
    cv::Mat mapxl, mapyl, mapxr, mapyr;
    cv::Rect roiL, roiR;
    cv::Mat imgl, imgr, grayl, grayr;
    cv::Mat disparity;

    cv::Ptr<cv::StereoBM> stereo;
    int numDisparities = 128, minDisparity = 4, blockSize = 17, speckleRange = 16, speckleWindowSize = 45;

    // read in calibration detail
    cv::FileStorage fs("calibration", cv::FileStorage::READ);
    fs["imgSize"] >> imgSize;
    fs["mapxl"] >> mapxl;
    fs["mapyl"] >> mapyl;
    fs["mapxr"] >> mapxr;
    fs["mapyr"] >> mapyr;
    fs["roiL"] >> roiL;
    fs["roiR"] >> roiR;

    stereo = cv::StereoBM::create(numDisparities, blockSize);
    stereo->setMinDisparity(minDisparity);
    stereo->setSpeckleRange(speckleRange);
    stereo->setSpeckleWindowSize(speckleWindowSize);

    while (true) {
        cap.read(frame);

        im0 = frame(cv::Range(0, h), cv::Range(0, w));
        im1 = frame(cv::Range(0, h), cv::Range(w, w2));

        cv::remap(im0, imgl, mapxl, mapyl, cv::INTER_LINEAR);
        cv::remap(im1, imgr, mapxr, mapyr, cv::INTER_LINEAR);

        cv::cvtColor(imgl, grayl, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imgr, grayr, cv::COLOR_BGR2GRAY);

        stereo->compute(grayl, grayr, disparity);
        disparity = (disparity / 16.0 - minDisparity) / numDisparities;
        
        cv::imshow("left", imgl);
        cv::imshow("right", imgr);
        cv::imshow("disparity", disparity);

        if (cv::waitKey(1) == 27)
            break;
    }
    cap.release();
    cv::destroyAllWindows();*/

    // use depth object
    Depth objd("calibration");
    objd.createStereoMatch();

    cv::Mat disparity;
    while (true) {
        cap.read(frame);

        im0 = frame(cv::Range(0, h), cv::Range(0, w));
        im1 = frame(cv::Range(0, h), cv::Range(w, w2));
        cv::imshow("left", im0);
        
        
        disparity = objd.disparityMap(objd.calibrateImage(im0, true), objd.calibrateImage(im1, false));

        cv::imshow("disparity", disparity);

        if (cv::waitKey(1) == 27)
            break;
    }

    cap.release();
    cv::destroyAllWindows(); 

    return 0;
}
