# SFND Camera Course - Final Project

This file contains documentation for the SFND camera course final project rubric points.

The specifications of the system used to build and run the program are:

- Intel® Core™ i7-8550U w/ integrated UHD 620 graphics and 16GB of RAM

- Ubuntu 20.04

- g++ 9.3

- OpenCV 4.5.1

## FP.1 Match 3D Objects

The general idea to find matching bounding boxes between successive frames is to get all the keypoints inside a specific bounding box in the current frame, then look for the bounding box in the previous frame that contains the most number of corresponding keypoint matches, as that is the most likely matching bounding box.

Below is the pseudo-code of the implementation:

```cpp
for each currBoundingBox{
    get the keypoints matches that are inside currBoundingBox's 2D ROI
    then for each prevBoundingBox {
        count the number of corresponding keypoint matches that are inside the prevBoundingBox's 2D ROI
    }
    select the prevBoundingBox with the most keypoint matches contained as the matching pair of currBoundingBox
}
```

## FP.2 Compute Lidar-based TTC

The implementation is based from the previous lesson `Estimating TTC with Lidar`.

The lidar-based TTC is computed using the constant-velocity model.

The general algorithm is to get the nearest x value in the current and previous frames to compute `delta distance`. Then the inverse of the `frameRate` is the `delta time`.

From the above, the `velocity` is computed as `delta distance / delta time`.

Finally, using the constant-velocity equation `distance = velocity * time`, `TTC` is computed as `distance / velocity`, where the `distance` used is the current frame's nearest x value.

Two implementations are available inside `computeTTCLidar()` function. One does not filter for outliers, and the other does filtering.

Filtering outlier points is done by getting the N% nearest points based on x value, and then using the median value of the sampled points as the nearest point.

This method does not give the nearest point exactly, but it assumes that the cluster of N% nearest points are close enough together such that any point in that cluster (for this case the median value) is a good representative of the nearest. Also, this filtering method assumes that outliers a very few such as in the image below wherein there is only one outlier point.

![lidar point outlier example](doc/lidar_outlier.png)

As improvement, more robust outlier filtering can be done such as clustering using point-to-point distance, and then considering clusters other than the biggest one as outliers.

## FP.3 Associate Keypoint Correspondences with Bounding Boxes

Assigning keypoints associated with bounding boxes is straightforward. Basically, if the keypoint, is inside the 2D ROI of the bounding box, that keypoint is added to the list of associated keypoints.

In `clusterKptMatchesWithROI()`, keypoint association is made more robust by...

- shrinking the 2D ROI to remove some keypoints that don't belong to the target object, such as keypoints on the road

- and by computing the average euclidean distance between current and previous frame keypoint matches, then removing keypoint matches that have a euclidean distance that is not near the average value

## FP.4 Compute Camera-based TTC

The implementation is based from the previous lesson `Estimating TTC with Camera`.

After, associating the keypoint matches to the bounding boxes in FP.3, TTC can now be computed using distances between all keypoints across successive frames.

First, all distances between keypoints in the current frame are calculated.

Then, distances between keypoints in the previous frame are also calculated.

For each corresponsing keypoint pair, the computed distance in the current frame is divided by the computed distance in the previuos frame to create a list of distance ratios.

To remove the influence of outliers, the median value of the distance ratios is used to compute for the TTC.

## FP.5 Performance Evaluation 1

### Example 1

For data frame index 6 (previous) and 7 (current), a long TTC value of `47s` was calculated using the lidar data.

Looking at the top view lidar data in the image below, it can be seen that frame 6 lidar points along the middle of the cluster is more spread out compared to other lidar data where the points are compact.

The spread-out points in frame 6 caused the distance of the nearest point to be almost the same with the distance of the nearest point in the next frame, frame 7.

This resulted in a long TTC value since it seemed that the target object did not move much. This could be because of sensor error/noise during those frames.

![long TTC lidar](doc/long_ttc_lidar.jpeg)

### Example 2

Another example, that has already been discussed in previous lessons, is that lidar TTC estimation will become incorrect, if outlier points that are much nearer than the actual nearest point of the object are not filtered out.

Since those points, are outliers, they will appear only in one frame, and not on the next one. The effect is that it will seem that the target object moved farther away causing a negative TTC value.

![lidar point outlier again](doc/lidar_outlier.png)

![lidar point outlier ttc](doc/outlier_ttc.jpeg)

## FP.6 Performance Evaluation 6

The following detector + descriptor extractor combinations were analyzed:

- `FAST + BRIEF`

    - From the midterm project, this combination was evaluated as the best one due to the fast execution speed and high number of detected keypoint matches, though with the trade-off of resulting in more false positives.

- `AKAZE + AKAZE`

    - This combination was selected since it had good distribution of keypoints along the edges of the target object which is expected to give more accuracy.

- `HARRIS + BRIEF`

    - `HARRIS` detector results in few detected keypoints. This combination is to show the adverse effect of having few keypoints for TTC estimation.

- `SIFT + SIFT`

    - Lastly, all the other combinations are binary descriptors, this combination is to show the performance of HOG-based descriptor.

The table and graph below shows the TTC values for the different detector + descriptor combinations, along with the lidar TTC as reference.

| Frame Number | Lidar TTC (s) | FAST + BRIEF TTC (s) | AKAZE + AKAZE TTC (s) | HARRIS + BRIEF TTC (s) | SIFT + SIFT TTC (s) |
| ------------ | ------------- | -------------------- | --------------------- | ---------------------- | ------------------- |
| 1            | 13.9          | 11.8                 | 13.6                  | nan                    | 12.1                |
| 2            | 11.2          | 9.5                  | 17.9                  | inf                    | 12.4                |
| 3            | 19.5          | 11.7                 | 14.3                  | 405.7                  | 12.4                |
| 4            | 12.1          | 21.4                 | 18                    | inf                    | 17                  |
| 5            | 13.5          | 16.7                 | 17.6                  | 11                     | 14.9                |
| 6            | 7.1           | 12.6                 | 17.8                  | 50.5                   | 12                  |
| 7            | 47.3          | 12.8                 | 16                    | 13.1                   | 13.4                |
| 8            | 22.2          | 12.3                 | 16.7                  | 9                      | 14.8                |
| 9            | 12.5          | 12.8                 | 17.8                  | 25.8                   | 13.2                |
| 10           | 14.8          | 12.5                 | 15.2                  | 8.5                    | 10.7                |
| 11           | 10.1          | 10                   | 13.2                  | inf                    | 10.8                |
| 12           | 10.2          | 9.7                  | 12.6                  | 7.2                    | 10.4                |
| 13           | 9.1           | 10.1                 | 13.3                  | 13.9                   | 9                   |
| 14           | 11.3          | 8.8                  | 10.9                  | 6.4                    | 8.8                 |
| 15           | 8             | 9.4                  | 11.8                  | inf                    | 8.8                 |
| 16           | 9.1           | 8.6                  | 10.4                  | 7.4                    | 8.2                 |
| 17           | 9.5           | 10.6                 | 9.9                   | 8                      | 8.5                 |
| 18           | 9.4           | 11.6                 | 10.2                  | nan                    | 8.5                 |

![lidar camera ttc graph](doc/lidar_camera_ttc_graph.png)

From the graph above, it can be clearly seen that the trend of all TTC estimations are all decreasing, which is expected since the ego vehicle is getting closer to the target vehicle in front.

`AKAZE + AKAZE` resulted in a trend line that didn't have large variations. This could indicate that indeed the keypoint matches are accurate for this combination.

`FAST + BRIEF` resulted in a trend line w/ larger but still acceptable variations than `AKAZE + AKAZE`. Looking back at the midterm project, this combination had the advantage of faster execution time which could be more valuable depending on the target application.

As expected `HARRIS + BRIEF` resulted in a lot of frames where TTC could not be calculated. This is due to the low number, to even possibly no keypoint matches.

The HOG-based combination of `SIFT + SIFT` resulted in a stable TTC trend line. Although really slow, detected keypoint matches are reliable resulting in good TTC estimation.