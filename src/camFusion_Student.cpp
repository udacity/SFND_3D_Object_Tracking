
#include <iostream>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

namespace {

// helper function to filter the matches in the current frame that are inside a given roi
std::vector<cv::DMatch> getMatchesInsideRoi(const std::vector<cv::KeyPoint>& keypoints, const std::vector<cv::DMatch> &matches, const cv::Rect& roi) {
    std::vector<cv::DMatch> matchesInsideRoi;
    for (auto& match : matches) {
        auto& keypoint = keypoints[match.trainIdx];
        if (roi.contains(keypoint.pt)) {
            matchesInsideRoi.push_back(match);
        }
    }
    return matchesInsideRoi;
}

} // namespace

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // get all matches inside bounding box, bounding box is size is shrunk to remove some keypoints that don't belong to the target object
    cv::Rect smallerBox = boundingBox.roi;
    smallerBox.width = boundingBox.roi.width * 0.9;
    smallerBox.height = boundingBox.roi.height * 0.9;
    const std::vector<cv::DMatch> matchesInsideBox = ::getMatchesInsideRoi(kptsCurr, kptMatches, smallerBox);

    // compute the average euclidean distance of all matches
    std::vector<double> euclideanDistances;
    for (auto& match : matchesInsideBox) {
        auto& currKeypoint = kptsCurr[match.trainIdx];
        auto& prevKeypoint = kptsPrev[match.queryIdx];
        euclideanDistances.push_back(cv::norm(currKeypoint.pt - prevKeypoint.pt));
    }
    const double totalEuclideanDistance = std::accumulate(euclideanDistances.begin(), euclideanDistances.end(), 0.0);
    const double averageEuclideanDistance = totalEuclideanDistance / euclideanDistances.size();

    // filter out matches that are far from the mean (outliers)
    const double thresholdDistance = averageEuclideanDistance * 1.5;
    for (size_t i = 0; i < euclideanDistances.size(); ++i) {
        if (euclideanDistances[i] < thresholdDistance) {
            auto& match = matchesInsideBox[i];
            boundingBox.kptMatches.push_back(match);
            boundingBox.keypoints.push_back(kptsCurr[match.trainIdx]);
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // computes distance of keypoints to each other keypoint
    // then for each distance, compute the ratio between current and previos frame value
    std::vector<double> distanceRatios;
    for (size_t i = 0; i < kptMatches.size(); ++i) {
        auto& currKeypoint = kptsCurr[kptMatches[i].trainIdx];
        auto& prevKeypoint = kptsPrev[kptMatches[i].queryIdx];
        for (size_t j = 0; j < kptMatches.size(); ++j) {
            // skip if same keypoint
            if (i == j) {
                continue;
            }

            // compute distance betweek keypoints
            auto& currKeypointOther = kptsCurr[kptMatches[j].trainIdx];
            auto& prevKeypointOther = kptsPrev[kptMatches[j].queryIdx];
            double currDistance = cv::norm(currKeypoint.pt - currKeypointOther.pt);
            double prevDistance = cv::norm(prevKeypoint.pt - prevKeypointOther.pt);

            // compute ratio between current and previous frame distance
            static const double minDistance = 0;
            if (prevDistance > 0.0 && currDistance >= minDistance) {
                distanceRatios.push_back(currDistance / prevDistance);
            }
        }
    }
    if (distanceRatios.empty())
    {
        TTC = NAN;
        return;
    }

    // compute TTC using median distance ratio
    std::sort(distanceRatios.begin(), distanceRatios.end());
    double medianDistRatio;
    size_t size = distanceRatios.size();
    if (size % 2 == 0) {
      medianDistRatio = (distanceRatios[size / 2 - 1] + distanceRatios[size / 2]) / 2.0;
    } else {
      medianDistRatio = distanceRatios[size / 2];
    }
    double deltaT = 1 / frameRate;
    TTC = -deltaT / (1 - medianDistRatio);
}

namespace {

// Get the median value of the N% nearest points.
double medianOfNearestXs(const std::vector<LidarPoint>& points, double samplePercentage) {
    // sort by nearest X
    auto compareNearerX = [](const LidarPoint& p1, const LidarPoint& p2) {
        return p1.x < p2.x;
    };
    std::vector<LidarPoint> sortedPoints = points;
    std::sort(sortedPoints.begin(), sortedPoints.end(), compareNearerX);

    // resize to remove elements at the end of the vector
    size_t sampleSize = sortedPoints.size() * samplePercentage;
    sortedPoints.resize(sampleSize);

    // get the median value
    return sortedPoints[sampleSize / 2].x;
}

} // namespace

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    bool filterOutliers = true;
    double prevNearestX;
    double currNearestX;

    if (filterOutliers) {
        // Filter outlier points by getting the N% nearest points and using the median value as the nearest point.
        // This does not give the nearest point exactly but it assumes that the cluster of N% nearest points are close enough together,
        // such that any point in that cluster (for this case the median value) can represent the nearest point.
        const double samplePercentage = 0.5;
        prevNearestX = medianOfNearestXs(lidarPointsPrev, samplePercentage);
        currNearestX = medianOfNearestXs(lidarPointsCurr, samplePercentage);
    } else {
        auto compareNearerX = [](const LidarPoint& p1, const LidarPoint& p2) {
            return p1.x < p2.x;
        };
        prevNearestX = std::min_element(lidarPointsPrev.begin(), lidarPointsPrev.end(), compareNearerX)->x;
        currNearestX = std::min_element(lidarPointsCurr.begin(), lidarPointsCurr.end(), compareNearerX)->x;
    }

    // compute TTC using constant velocity model
    const double deltaX = prevNearestX - currNearestX;
    const double deltaT = 1.0 / frameRate;
    const double velocity = deltaX / deltaT;
    TTC = currNearestX / velocity;

    std::cout << "delta T " << deltaT << "s" <<std::endl;
    std::cout << "nearest point prev X " << prevNearestX << std::endl;
    std::cout << "nearest point curr X " << currNearestX << std::endl;
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // for each current frame bounding box...
    for (const auto& currBoundingBox : currFrame.boundingBoxes) {
        std::cout << "Finding best match for current frame bounding box " << currBoundingBox.boxID << std::endl;
        // get matches that are inside the current bounding box
        const std::vector<cv::DMatch> matchesInsideCurrBox = ::getMatchesInsideRoi(currFrame.keypoints, currFrame.kptMatches, currBoundingBox.roi);

        // then try to find the best matching bounding box from the previous frame w/ the most number of matching keypoints inside prev box
        std::unordered_map<int, int> prevBoxIdAndNumberOfMatchesMap;
        for (const auto& prevBoundingBox : prevFrame.boundingBoxes) {
            // count the number of matching keypoints inside prev frame bounding box roi
            int prevKeypointsInsideRoiCounter = 0;
            for (auto& matchInsideCurrBox : matchesInsideCurrBox) {
                int prevKeypointIndex = matchInsideCurrBox.queryIdx;
                auto& prevKeypoint = prevFrame.keypoints[prevKeypointIndex];
                if (prevBoundingBox.roi.contains(prevKeypoint.pt)) {
                    ++prevKeypointsInsideRoiCounter;
                }
            }
            if (prevKeypointsInsideRoiCounter > 0) {
                // prev frame bounding box is candidate as best match
                prevBoxIdAndNumberOfMatchesMap[prevBoundingBox.boxID] = prevKeypointsInsideRoiCounter;
                std::cout << "Candidate found. Previous frame bounding box " << prevBoundingBox.boxID;
                std::cout << " w/ corresponding matches count " << prevKeypointsInsideRoiCounter << std::endl;
            }
        }
        if (!prevBoxIdAndNumberOfMatchesMap.empty()) {
          int bestPrevBoxIndex = std::max_element(prevBoxIdAndNumberOfMatchesMap.begin(),
                                                  prevBoxIdAndNumberOfMatchesMap.end(),
                                                  [](const std::pair<int, int>& p1, const std::pair<int, int>& p2) {
                                                      return p1.second < p2.second;}
                                                  )->first;
          bbBestMatches[currBoundingBox.boxID] = bestPrevBoxIndex;
          std::cout << "BEST MATCH is previous frame bounding box "
                    << bestPrevBoxIndex << std::endl;
        }
        std::cout << "----" << std::endl;
    }
}
