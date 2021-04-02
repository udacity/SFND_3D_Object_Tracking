
# FP.1 Match 3D Objects

I looped over the matches, checked the bounding boxes if contain the matches. Got the mode, the highest number of occurrences, and returned the bestmatches in a map.


# FP.2 Compute Lidar-based TTC

It is based on the exercise in lesson 3. I have added more accurate criterion, which is the median. Before using the median, I got negative time.

# FP.3 Associate Keypoint Correspondences with Bounding Boxes

I have calculated the euclidean distance (as recommended by the instructor) between previous and current keypoints. Evaluated the mean, and the standard devition. I compared the difference based on the stddev to neglect the outliers.

# FP.4 Compute Camera-based TTC

This code is the same as in the excercise of lesson 3

# FP.5 Performance Evaluation 1

I faced negative TTC with the lidar, and high TTC with the camera. I solved these problems by using median technique as recommended. 
I guess this is because of the outliers, and my poor clustering (which I have improved)


### Lidar & Camera TTC (Brisk-Brisk) before median improvement
| Lidar TTC | Camera TTC |
|--------|------|
| **-10.85** | 8.95 |
| 9.22   | 9.96 |
| 10.967 | 9.56 |
| 8.094  | 8.57 |
| 3.175  | 9.5  |
| **-9.9**   | 9.54 |

---

# FP.6 Performance Evaluation 2

I tried all the possible combinations. Put them in a table, sorted them.

I have attached the CSV file.

The Top 3:
1. HARRIS - BRISK
1. HARRIS - BRIEF
1. HARRIS - ORB

My criterion is based on the difference between lidar TTC & cam TTC. The less difference, the more accurate.

<img src="images/graph.png">
<img src="images/3d_demo.gif">


# Extra Notes
* I am submitting the repo with (report) branch
* I created bugs&problems md which includes some of the bugs I encountered as well as the references I used.

