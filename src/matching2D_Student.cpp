#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // time
    double t = static_cast<double>(cv::getTickCount());

    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = (descriptorType == "DES_BINARY") ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F) descSource.convertTo(descSource, CV_32F);
        if (descRef.type() != CV_32F) descRef.convertTo(descRef, CV_32F);
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    } else {
        throw std::runtime_error("Unsupported matcher type. Supported matchers are: MAT_BF, MAT_FLANN.");
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        static const int k = 2;
        std::vector<std::vector<cv::DMatch>> kMatches;
        matcher->knnMatch(descSource, descRef, kMatches, k);

        // filter matches using descriptor distance ratio test
        bool filtering = true;
        static const double descDistanceRatioThreshold = 0.8;
        for (auto& kMatch : kMatches) {
            if (!filtering) {
                // get the best match
                matches.push_back(kMatch[0]);
            }
            else {
                // get only the best match if it passes descriptor distance ratio test
                double descDistanceRatio = kMatch[0].distance / kMatch[1].distance;
                if (descDistanceRatio < descDistanceRatioThreshold) {
                    matches.push_back(kMatch[0]);
                }
            }
        }
    } else {
        throw std::runtime_error("Unsupported selector type. Supported selectors are: SEL_NN, SEL_KNN.");
    }

    // time
    t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
    std::cout << matcherType << " matcher and " << selectorType << " selector with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType == "BRIEF") {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    } else if (descriptorType == "ORB") {
        extractor = cv::ORB::create();
    } else if (descriptorType == "FREAK") {
        extractor = cv::xfeatures2d::FREAK::create();
    } else if (descriptorType == "AKAZE") {
        extractor = cv::AKAZE::create();
    } else if (descriptorType == "SIFT") {
        extractor = cv::SIFT::create();
    } else {
        throw std::runtime_error("Unsupported descriptor type. Supported descriptors are: BRIEF, ORB, FREAK, AKAZE, SIFT.");
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
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

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // time
    double t = static_cast<double>(cv::getTickCount());

    // Harris detector parameters
    static const int blockSize = 2;     // blockSize × blockSize neighborhood
    static const int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    static const int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    static const double k = 0.04;       // Harris parameter

    // Detect Harris corners
    cv::Mat cornersImg(img.size(), CV_32FC1);
    cv::cornerHarris(img, cornersImg, blockSize, apertureSize, k);
    // then normalize
    cv::Mat cornersImgNorm;
    cv::normalize(cornersImg, cornersImgNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    // corners as keypoints with NMS
    static const double maxOverlap = 0.0; // NMS max overlap
    for (size_t y = 0; y < cornersImgNorm.rows; y++) {
        for (size_t x = 0; x < cornersImgNorm.cols; x++) {
            int response = cornersImgNorm.at<float>(y, x);
            // skip points below response threshold
            if (response < minResponse) {
                continue;
            } 
            cv::KeyPoint newKeypoint;
            newKeypoint.pt = cv::Point2f(x, y);
            newKeypoint.size = 2 * apertureSize;
            newKeypoint.response = response;

            // perform NMS
            bool hasOverlap = false;
            for (auto& keypoint : keypoints) {
                double overlap = cv::KeyPoint::overlap(newKeypoint, keypoint);
                if (overlap > maxOverlap) {
                    hasOverlap = true;
                    if (newKeypoint.response > keypoint.response) {
                        keypoint = newKeypoint;
                        break;
                    }
                }
            }
            if (!hasOverlap) {
                keypoints.push_back(newKeypoint);
            }
        }
    }

    // time
    t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
    std::cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// FAST, BRISK, ORB, AKAZE, SIFT
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    // use default parameters for each detector type
    cv::Ptr<cv::Feature2D> detector;
    if (detectorType == "FAST") {
        detector = cv::FastFeatureDetector::create();
    } else if (detectorType == "BRISK") {
        detector = cv::BRISK::create();
    } else if (detectorType == "ORB") {
        detector = cv::ORB::create();
    } else if (detectorType == "AKAZE") {
        detector = cv::AKAZE::create();
    } else if (detectorType == "SIFT") {
        detector = cv::SIFT::create();
    } else {
        throw std::runtime_error("Unsupported modern detector. Supported detectors are: FAST, BRISK, ORB, AKAZE, SIFT.");
    }

    // time
    double t = static_cast<double>(cv::getTickCount());

    detector->detect(img, keypoints);

    // time
    t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
    std::cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}