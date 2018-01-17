# Advanced Lane Finding Project
This report is made to report the results of the Advanced lane finding project,
P4 of Udacity's Self-driving car Nanodegree course.

## 1. Goals
The goals / steps of this project are stated as follows,

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./doc_images/1_camera_calibrate.png "Undistorted"
[image2]: ./doc_images/2_distort_correct.png "Road Transformed"
[image3]: ./doc_images/3_binary_image.png "Binary Example"
[image4]: ./doc_images/4_perspective_transform.png "Warp Example"
[image5]: ./doc_images/5_sliding_window.png "Fit Visual"
[image6]: ./doc_images/6_lane_image.png "Output"

## 2. Contents
This section states briefly the contents of the report.

* Rubic points
* Discussion
* Conclusion

## 3. Rubric Points
This section states the rubic requirements of this project and how they are
satisfied briefly.

### 3.1. Provide a Writeup / README that includes all the rubric points and how you addressed each one
This document satisfies the criteria 3.1.

### 3.2. Camera calibration
The camera used for lane detection is calibrated initially with the provided
chessboard images.

The calibration matrix and the distortion coefficients are computed using the
functions provided by the OpenCV library. Initially, the object points are
computed based on the size of the chessboard, by considering the intersection of
blacks and blacks or whites and whites.

The provided calibration images are transformed to grayscale and then with the
help of function _findChessboardCorners()_, the image points of each of the
intersection points are computed and checked with _drawChessboardCorners()_.

It is worth mentioning that the size of the chessboard given in the lessons are
of different size from the chessboard size for calibrating the considered
camera. The size of the intersections of the given chessboard is 9x6.

The object and image points of each of the calibration images were appended to
a list and with the help of function calibrateCamere()_, the calibration matrix
and distortion coefficients were computed. These coefficients were used to
undistort a test image and visually verified. The undistorted image is shown in
the image below.

![Camera calibration][image1]
**Camera Calibration**

The camera calibration is performed in a separate python file 
`camera_calibration.py`.

### 3.3. Pipeline - test images

#### 3.3.1. Distortion corrected image
The computed distortion coefficients are dumped to a pickle file and are just
loaded to find the undistorted images. One such image corrected for distortion
is shown in the image below.

![Undistorted Image][image2]
**Undistorted Image**

The code for distortion correction can be found in file `fine_lanes.py` between
lines 9 through 13. A separate function _undistortImage()_ has been defined to
perform undistortion.

#### 3.3.2.Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.
Various techniques were adopted to create the binary image where the lanes are
clearly shown. A list of considered thresholding methods is mentioned below.

- sobel x threshold.
- sobel magnitude threshold.
- sobel gradient threshold.
- Thresholds on H and S channel of HSL transformation.
- Threshold on R-channel on RGB image.

| Method		| Kernel size | Threshold value (min - max)|
|:-------------------------|:--------:|:----------------:|
| Sobel x			| 5 | 20 - 80 |
| Sobel magnitude		| 15 | 30 - 130 |
| Sobel gradient		| 15 | 40 - 75 deg |
| H-channel of HSL image	| - | 20 - 30 |
| S-channel of HSL image 1	| - | 170 - 255 |
| S-channel of HSL image 2	| - | 150 - 255 |
| R-channel of RGB image	| - | 220 - 255 |


The binary image is created with a union or intersection of the above mentioned
thresholding techniques. The method used is mentioned below.

```python
(((sobel_x) & (sobel_mag)) & ((sobel_x) & (sobel_grad))) | (((s_th_1) & (r_th)) | ((h_th) & (s_th_2)))
```

The resulting image is shown below.

![Binary Image][image3]
**Binary Image**

The code for creating a thresholded binary image can be found in the python file
`find_lanes.py` between the lines 15 and 125. Here separate functions
abs_sobelthresh(), mag_sobel_thresh() and gradient_sobel_thresh() are defined to
perform different thresholding techniques. These functions are called by
_createBinaryThresholded()_ function which outputs the actual binary image.

#### 3.3.3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
In order to find the lanes with a suitable algorithm, the available image had to
be transformed to another perspective. This perspective transformation is made
with the help of _warpPerspective()_ method available in the OpenCV library. The
method _warpPerspective()_ needs to take the transformation matrix as input.
This transformation matrix is computed with the function
_getPerspectiveTransform()_, which takes the input of source and destination
points.

The source points are the identified points in the source image and the
destination points tell us where the source points will end up on the warped
image. Since here it is only necessary to identify only the ego's lane, the
source points are defined as those which includes the own lane of the ego
vechicle. The source and destination points used here are mentioned in the
table below.

| Points		| x | y |
|:---------------------------|:------------------------:|:------------------------:|
| Source points		| [275, 600, 685, 1045] | [677, 446, 446, 677] |
| Destination points	| [426, 426, 852, 852] | [720, 0, 0, 720] |

![Perspective Transformation][image4]
**Perspective Transformation**

The code for perspective transformation can be found in file `fine_lanes.py`
from line 254 until line 265.

#### 3.3.4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
The thresholds for the warped binary image was already tuned to pick-up most of
the lane pixels. Now the lanes are initially identified with the help of a
sliding window algorithm.

**Sliding window algorithm**
It was observed that the warped image (perspective transformed) had more
blurriness with increased longitudinal distance from the ego vehicle. Hence to
improve certainity of lane detection, histograms along the 'x' direction is
computed for the bottom half of the image.

The algorithm starts with the identification of a solid line by detecting the
peak in the histogram. Since solid lines are always available in the provided
project video, this was considered as a good starting point. If the peak is
detected left of the mid-point in x-direction, then the left lane is assumed to
have been found. If the peak is to the right, then it is assumed to correspond
to the right lane.

The peak point is considered as a start point for that specific lane by the
algorithm. The start point for the other lane is defined by offsetting the peak
left or right correspondingly by the _pixellanewidth_. Now the parameters for
the sliding window algorithm are defined as follows,

* Number of windows - 9
* Window margin - 60 pixels
* Window height - image_height/number of windows
* Number of qualifiers for realigning the window - 30 pixels

The algorithm now defines a box with the start point of the left and right lanes
with the defined sliding window parameters and counts the number of qualified
pixels (denoted as inliers from now on). The inliers are appended to an empty
list. If the number of inliers are above the defined number of qualifiers, then
the start point of the left and right lanes are updated.

The same procedure is repeated for the defined number of windows (inside a for
loop) and inliers for the left and right lanes are continuously appended. For
each iteration the location of the window is slided left or right for the next
iteration and hence the name.

Finally the inliers are used to shortlist the lane candidates for the left and
right lane separately. Now a second order polynimial is fitted to the left and
right lanes using the _polyfit_ function provided by the numpy module.

The code for sliding windows algorithm can be found in file `find_lanes.py`
between lines 138 and 189. Here a separate function by the name
_slidingWindowSearch()_ is defined which performs the sliding window algorithm.

![Sliding Window and Lane Fit][image5]
**Sliding Window and Lane Fit**

#### 3.3.5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
**Radius of curvature**
The polynomial fit is now in a warped pixel frame. A radius of curvature
calculated on this fit doesn't correspond to the actual radius of curvature in
the real world. Hence the x and y pixels have to be scaled to real world
frame before calculating the radius of curvature.

The scaling factors used for radius of curvature calculation is as follows,

* x scaling factor - 3.7/(pixel width + 24)
* y scaling factor - 40/image height

The x scaling factor defines the distance covered (in meters) by each pixel in
x direction. Since the project video was recorded on a highway, the normal lane
width is assumed to be 3.7 meters. Hence a division of 3.7m by the average
distance (in pixels) that separates the lanes gives us the distance (in meters)
covered by each pixel. Here, 24 pixels is added to the pixel lane width for
tuning and generalizing.

The y scaling factor defines the distance covered by each pixel in y-direction.
The numerator was found, by tuning, to be around 40m in the longitudinal
direction. So a division of 40 meters by the total height of the warped images
will correspond to the distance covered by each of the pixel in y-direction.

Now, the inliers corresponding to the left and right lanes are scaled in x
and y with the defined scaling factors. A second order polynimial fit can now
be calculated by using the _polyfit()_ and will correspond to the lane fit in
real world. The curvature can now be calculated by using the found coefficients
of the second order polynomial fit. This computation is performed in the
function _computeRadiusOfCurvature()_.

The radius of curvature is computed in file `fine_lanes.py` between lines 191
and 208. A separate function is defined by the name _computeRadiusOfCurvature()_
to calculate the radius of curvature. The formula used here is exactly the same
formula mentioned in the lesson.

**Lane offset**
The lane offset is a measure of how much is the center of the ego vehicle offset
from the lane center. Firstly, left and right offset are computed seaparetely
and compared with the center of the image, since the camera is assumed to be
mounted in the center of the ego vehicle.

The left lane position is computed by using the point close to the ego vehicle
(which is the image height itself) with the computed second order polynomial
fit. The output will be the position of the lane in pixels and in warped
perspective and hence has to be subtracted with the location of the camera on
the ego vehicle (which is half the width of the image) to give the offset.

The found offset will have to be scaled as it has to be converted from
pixel frame to real world frame (the scaling factor used here is same as that
mentioned in the radius of curvature calculation). Now the offset of each of the
lanes are compared to the lane center (which is half of the lane width). Now,
the left and right offsets are compared with the lane center and text are
displayed accordingly on the images.

The code for lane offset can be found in file `fine_lanes.py` beween lines 405
and 437. This includes finding the lane offset from the mounting position of
the camera in the ego vehicle and calculating the actual offset in real world
coordinates. Here a small function _computeLaneOffset()_ is defined to return
the position of the lane in pixel frame. For visualization purposes, the
computed offset is also written on to the output image.

#### 3.3.6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
Once the procedure until the lane offset is completed, a ploygon is defined is
filled using the method _fillpoly()_ available in the OpenCV library. This
polygon is still in the warped perspective and has to be transformed back to the
undistorted image frame. For this the inverse of the transformation matrix is
computed by swapping the input to the _getPerspectiveTransform()_ method in the
OpenCV library.

This inverse transformation matrix is used in the _warpPerspective()_ method to
unwarp the warped image.

After unwarping, the curvature and lane offset information is written as text on
the images.

![Lane Image][image6]
**Lane Image**

### 3.4. Pipeline - video
A link to the output video is provided here,

[Project output video](https://drive.google.com/file/d/1-50uKdLmAg_R5ZNv3glGaUZdudognM_g/view?usp=sharing "My project video")

In order to find the lane lines on the video, the _fl_image()_ method available
in the VideoClip module is used. A separate function is defined, which takes
each frame (image) of the video as input and processes the image as follows,

* Distortion correction
* Binary threshold image creation
* Perspective transformation
* Identifying lane pixels
* Radius of curvature and lane offset calculation
* Unwarping

#### 3.4.1. Distortion correction
The distortion coeffecients are computed separately as described in section
3.3.1 and are dumped to a pickle file. Before processing the video clip, the
distortion coefficients are loaded from pickle file and each frame in the video
is undistorted in the function _undistortImage()_.

#### 3.4.2. Binary thresholded image creation
Once the images are undistorted, the binary image is created by using the same
techniques mentioned in section 3.3.2. This is done in the function
_createBinaryThresholded()_


#### 3.4.3. Perspective transformation
The perspective transformation is done in a similar fashion as stated in the
section 3.3.3.


#### 3.4.4. Identifying lane pixels
This is where the pipeline differs for the images and videos. The sliding window
algorithm takes a handful of computation time. It may be unnecessary to perform
the sliding window algorithm for every frame from the video (as the ego is
continuously driving). Instead, when a good fit is available, that can be used
as a good starting point in the next frame.

A region around the already computed fit is defined where inliers for the new
lanes are found. These inliers can then be used to update the fit parameters.

There may be situations where the lane lines are not very clearly identified
by all the above techniques. In such a situation it may be necessary to use
the already found fits as a starting point for lane search. Hence, a Line class
is defined which holds the information of previous detection.

If say the sliding window technique finds a good lane fit, then the line class
for the left and right lane can be separately updated (parameter updated:
detected) to be detected. In the next frame, we can just read in the information
from the left and right line classes and decide whether to perform sliding
window detection or creating a margin on the available fit.

Also, in order to produce a smoother output, it may be necessary to average the
detections over a few samples. For each valid found (validity is checked based
on the left and right lanes being parallel), its fit parameter is appended to
a list and a best fit is computed by averaging the last 5 fit parameters. It
will be this best fit parameter that will be used for lane detection with
margin technique, radius of curvature and lane offset calculation.

If the computed fit is not found to be valid, then a reset counter is
incremented. When the counter reaches a value of `25`, the corresponding
detected parameter will be set to _False_ and the next frame of image will
initiate the sliding window technique.


#### 3.4.5. Radius of curvature and lane offset
The radius of curvature and lane offset are calculated with the available best
fit parameter. The procedure of obtaining and scaling is exactly same as
mentioned in section 3.3.5.


#### 3.4.6. Unwarping
Unwarping is done in the same fashion as mentioned in section 3.3.6.

## 4. Discussion
This section states some advantages and disadvantages for the implemented
method.

**Advantages**
- The implemented method is robust for highways.
- This method works best when the ego vehicle is driving close to the road
shoulder.

**Disadvantages**
- The main disadvantage is that the algorithm works only for highways.
- There is some jittering in the lane detection which could be improved by
decreasing the look ahead distance.
- This method always expects solid line (lane shoulder) on one side of the ego
vehicle. The algorithm's effect was not tested when the ego vehicle was driving
between dotted lanes.

**Problems faced**
The main problem faced was to detect lanes in those patchy areas (areas with
reworked road surface). It was observed that the length of the detected lane
decreased by half the usual length in those sections.

The problem was mainly overcome by setting a reset threshold. There was a reset
counter incremented which would make the algorithm to use the already estimated
parameters for a specified number of frames.

## 5. Conclusion
A method for detecting the lane lines left and right of the ego vehicle was
investigated, implemented and tested. The procedure involved covers most topics
mentioned in the lessons.

