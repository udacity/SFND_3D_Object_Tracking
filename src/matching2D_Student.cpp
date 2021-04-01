#include <numeric>
#include "matching2D.hpp"
#include <opencv2/features2d.hpp>

// #include <iostream>
// #include <fstream>
// #include <string>

// #include "helper.h"

using namespace std;
// using cv::DMatch;
// using cv::BFMatcher;
// using cv::DrawMatchesFlags;
// using cv::Feature2D;
// using cv::ORB;
// using cv::BRISK;
// using cv::AKAZE;
// using cv::FREAK;

// Descriptors & Dectectors
enum Dectector
{
    FAST,
    BRISK,
    ORB,
    AKAZE,
    SIFT,
    HARRIS,
    BRIEF,
    FREAK
};

// reference to descriptor code
// https://github.com/oreillymedia/Learning-OpenCV-3_examples/blob/master/example_16-02.cpp

// void //write_a(string filename, string str)
// {

//     FILE *fp;

//     fp = fopen(filename.c_str(), "a");
//     if (fp == NULL)
//     {
//         perror("Error");
//         exit(1);
//     }
//     else
//     {

//         fprintf(fp, "%s", str.c_str());
//     }

//     fclose(fp);
// }

int stringToIndex(string str)
{

    if (str.compare("FAST") == 0)
    {
        return FAST;
    }
    else if (str.compare("BRISK") == 0)
    {
        return BRISK;
    }
    else if (str.compare("ORB") == 0)
    {
        return ORB;
    }
    else if (str.compare("AKAZE") == 0)
    {
        return AKAZE;
    }
    else if (str.compare("SIFT") == 0)
    {
        return SIFT;
    }
    else if (str.compare("BRIEF") == 0)
    {
        return BRIEF;
    }
    else if (str.compare("FREAK") == 0)
    {
        return FREAK;
    }

    return -1;
}

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        uint k = 2;
        vector<vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descSource, descRef, knnMatches, k);
        float ratio = 0.8;

        for (vector<cv::DMatch> m : knnMatches)
        {
            if (m[0].distance < (m[1].distance * ratio))
            {
                matches.push_back(m[0]);
               
            }
        }

         cout<< "*** Neighbourhood *** " << matches.size() << endl;
    }

    cout << "# keypoints: " << matches.size() << endl;
    // //write_a("results.csv", "# keypoints: ");
    // //write_a("results.csv", ",");
    //write_a("results.csv", to_string(matches.size()));
    //write_a("results.csv", "\n");

}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    // if (descriptorType.compare("BRISK") == 0)
    // {

    //     int threshold = 30;        // FAST/AGAST detection threshold score.
    //     int octaves = 3;           // detection octaves (use 0 to do single scale)
    //     float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

    //     extractor = cv::BRISK::create(threshold, octaves, patternScale);
    // }
    // else
    // {

    //     //...
    // }

    switch (stringToIndex(descriptorType))
    {
    case BRISK:
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);

        break;
    }

    case BRIEF:
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
        break;
    }

    case ORB:
    {
        extractor = cv::ORB::create();
        break;
    }

    case FREAK:
    {
        extractor = cv::xfeatures2d::FREAK::create();
        break;
    }

    case SIFT:
    {
        extractor = cv::xfeatures2d::SIFT::create();
        break;
    }

    case AKAZE:
    {
        extractor = cv::AKAZE::create();
        break;
    }
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    //writeCSV("results.csv", descriptorType);
    //write_file("results.csv", descriptorType);
    // //write_a("results.csv", "descriptor extraction");
    // //write_a("results.csv", ",");
    //write_a("results.csv", to_string(t));
    //write_a("results.csv", ",");
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    // //write_a("results.csv", "detector extraction");
    // //write_a("results.csv", ",");
    //write_a("results.csv", to_string(t));
    //write_a("results.csv", ",");

    int count = 0;
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        imwrite("test_"+ to_string(count++) + ".bmp", visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    //cout << std::stoi( detectorType ) << endl;

    // std::string str = "Hello World";
    // std::hash<std::string> hasher;
    // auto hashed = hasher(detectorType); //returns std::size_t
    // //std::cout << hashed << '\n';
    //std::cout << typeof(hashed) << endl;
    //hashed = hasher("FAST");
    //std::cout << hashed << '\n';

    string windowName;
    // dummy intialize
    cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
    switch (stringToIndex(detectorType))
    {

    case FAST:
    {

        std::cout << "Fast selected" << std::endl;
        int threshold = 30;
        bool bNMS = true;
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
        cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
        break;
    }

    case BRISK:

    {
        std::cout << "BRISK selected" << std::endl;
        cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
        //detector->detect(img, keypoints);
        break;
    }
    case ORB:

    {
        std::cout << "ORB selected" << std::endl;
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
        //detector->detect(img, keypoints);
        break;
    }

    case AKAZE:

    {
        std::cout << "AKAZE selected" << std::endl;
        cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
        //detector->detect(img, keypoints);
        break;
    }

    case SIFT:

    {
        std::cout << "SIFT selected" << std::endl;
        cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SIFT::create();
        //detector->detect(img, keypoints);
        break;
    }
    };

    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " detection with n: " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    // //write_a("results.csv", "detector extraction");
    // //write_a("results.csv", ",");
    //write_a("results.csv", to_string(t));
    //write_a("results.csv", ",");

windowName = detectorType;
int count = 0;
if (bVis)
{
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    imwrite("detect_"+ to_string(count++) + ".bmp", visImage);
    cv::waitKey(0);
}
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    double t = (double)cv::getTickCount();
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // STUDENTS NEET TO ENTER THIS CODE (C3.2 Atom 4)

    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n: " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    // //write_a("results.csv", "detector extraction H");
    // //write_a("results.csv", ",");
    //write_a("results.csv", to_string(t));
    //write_a("results.csv", ",");


    int count = 0;
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        imwrite("harris"+ to_string(count++) + ".bmp", visImage);
        cv::waitKey(0);
    }
}