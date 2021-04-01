
#include <iostream>
#include <algorithm>
#include <numeric> // accumulate
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"
#include <typeinfo>
#include <algorithm> // max_element

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

    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
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
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if (bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

double calcMean(std::vector<double> x)
{
    double mean = 0;
    if (x.size() > 1)
        return 0;
    for (int i = 0; i < x.size(); i++)
    {
        mean += x[i];
    }
    return mean / x.size();
}

double calcStddev(double mean, std::vector<double> x)
{
    if (x.size() < 1)
    {
        return 0;
    }
    // cout<< "x before " << x << endl;
    for (int i = 0; i < x.size(); i++)
    {
        x[i] -= mean;
    }
    // cout<< "x after " << x << endl;

    double num = std::pow(std::accumulate(x.begin(), x.end(), x[0]), 2);
    return sqrt(num / x.size());
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
    /**
     * Steps:
     * 1) calc distances
     * 2) calc mean
     * 3) calc standard dev
     * */

    std::map<vector<cv::DMatch>::iterator,double> theMap;
    std::vector<double> euclideanDistances;
    // calc distances
    for (auto it = kptMatches.begin(); it != kptMatches.end(); it++)
    {
        auto &currKpt = kptsCurr[(*it).trainIdx];
        auto &prevKpt = kptsPrev[(*it).queryIdx];

        if (boundingBox.roi.contains(currKpt.pt))
        {
            // https://cppsecrets.com/users/168511510411111711412197115105110104975764103109971051084699111109/C00-OpenCV-cvnorm.php           
            // https://www.techiedelight.com/print-keys-values-map-cpp/
            theMap[it] = cv::norm(currKpt.pt - prevKpt.pt);
            euclideanDistances.push_back(cv::norm(currKpt.pt - prevKpt.pt));
        }
    }

    // calc mean
    double mean = calcMean(euclideanDistances);
    // calc stddev
    double stddev = calcStddev(mean, euclideanDistances);
    // for each distance, if it is below the threshold, push it
    for (auto const& pair: theMap) {
        if ((pair.second - mean) < stddev){
            boundingBox.kptMatches.push_back(*pair.first);
        }
    }

}
// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...

    /**
     * when computing the camera-based TTC, try not to take all the keypoint correspondences. But, try to be more robust
     * because there will be more erroneous matches    
    */


    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }


    // STUDENT TASK (replacement for meanDistRatio)
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    cout << "Camera TTC: " << TTC << " s" << endl;

    // EOF STUDENT TASK
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
    /**
     * your task is to complete the TTC for all matched 3D objects based on lidar measurements
     * alone
     * when you look for the closest point in x, try to be robust against outliers which might
     *  be way too close, and thus lead to faulty estimates of the TTC
     * see Lesson 3.1 `Estimating TTC with Lidar`
    **/

    // auxiliary variables
    double dT = 1 / frameRate; // time between two measurements in seconds
    //this is based on the horizontal FOV
    double laneWidth = 4.0; // assumed width of the ego lane

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); it++)
    {

        if (abs(it->y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
            minXPrev = minXPrev > it->x ? it->x : minXPrev;
        }
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); it++)
    {

        if (abs(it->y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
            minXCurr = minXCurr > it->x ? it->x : minXCurr;
        }
    }

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);

    cout << "Lidar TTC: " << TTC << " s" << endl;
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // ...
    /*
     * This basic idea is to use the keypoint matches between the previous and the current images.
     * So you might want to make an outer loop of those.Then you should try to find out by which
     * bounding boxes key points are enclosed, both on the previous, and in the current currFrame
     * those will be your potential match candidates whose box ids you could store in, let's say,
     * a `multimap`
     * 
     * once you've completed the loop of our key point matches, you could try to find all the match
     * candidates in the multimap, which share the same bounding box id in the previous frame, and
     * just count them
     * 
     * In the end, you could assicuate tge bounding boxes with the highest number of occurrences,
     * but of course, it's up to you how to solve this problem
     * 
     * At the end of match bounding boxes, you must return your results to the main function, by 
     * storing the box ids of all matched pairs, in the map called bbBestMatches 
     * 
     * 
     * ///////////// Steps /////////////////
     *  inner and outer loops (done)
     *  multimap (done)
     *  find all the match candidates in the multimap
     * 
     * 
     struct DataFrame { // represents the available sensor information at the same time instance
    
        cv::Mat cameraImg; // camera image
        
        std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
        cv::Mat descriptors; // keypoint descriptors
        std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
        std::vector<LidarPoint> lidarPoints;

        std::vector<BoundingBox> boundingBoxes; // ROI around detected objects in 2D image coordinates
        std::map<int,int> bbMatches; // bounding box matches between previous and current frame
    };


        struct BoundingBox { // bounding box around a classified object (contains both 2D and 3D data)
        
        int boxID; // unique identifier for this bounding box
        int trackID; // unique identifier for the track to which this bounding box belongs
        
        cv::Rect roi; // 2D region-of-interest in image coordinates
        int classID; // ID based on class file provided to YOLO framework
        double confidence; // classification trust

        std::vector<LidarPoint> lidarPoints; // Lidar 3D points which project into 2D image roi
        std::vector<cv::KeyPoint> keypoints; // keypoints enclosed by 2D roi
        std::vector<cv::DMatch> kptMatches; // keypoint matches enclosed by 2D roi
    };
    */

    // multimap for previous and current frames
    multimap<int, int> boxesMap;

    for (auto match : matches)
    {

        cv::KeyPoint prevKP = prevFrame.keypoints[match.queryIdx];
        cv::KeyPoint currKP = currFrame.keypoints[match.trainIdx];

        int prevIdx, currIdx;
        // initialization
        prevIdx = currIdx = -1;

        for (auto box : prevFrame.boundingBoxes)
        {
            if (box.roi.contains(prevKP.pt))
            {
                prevIdx = box.boxID;
                break;
            }
        }

        for (auto box : currFrame.boundingBoxes)
        {
            if (box.roi.contains(currKP.pt))
            {
                currIdx = box.boxID;
                break;
            }
        }

        boxesMap.insert({currIdx, prevIdx});
    }

    // now we have a map that contains boxes id in the current
    // frames and its correponding in the previous frame
    int currSize = currFrame.boundingBoxes.size();
    int prevSize = prevFrame.boundingBoxes.size();

    // let's find the best matches for each
    for (int i = 0; i < currSize; i++)
    {
        // https://en.cppreference.com/w/cpp/algorithm/equal_range
        auto similarBoxes = boxesMap.equal_range(i);
        // https://www.geeksforgeeks.org/initialize-a-vector-in-cpp-different-ways/
        vector<int> count(prevSize, 0);

        for (auto pair = similarBoxes.first; pair != similarBoxes.second; pair++)
        {
            if ((*pair).second != -1)
            {
                count[(*pair).second] += 1;
            }
        }

        // get the mode, the highest number of occurrences
        // https://www.geeksforgeeks.org/stddistance-in-c/
        auto maxCount = std::max_element(count.begin(), count.end());
        // cout<< "check type" << endl;
        // cout<< typeid(std::max_element(count.begin(), count.end())).name() << endl;

        int mode = std::distance(count.begin(), maxCount);

        bbBestMatches.insert({mode, i});

        //print data

        cout << "BoxID: " << i << "& Mode: " << mode << endl;
    }
}
