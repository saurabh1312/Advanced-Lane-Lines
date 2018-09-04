## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/camera_calibration/calibration7.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera calibration code is written up in `camera_calibration.py`

I open each file (chessboard images) in the camera_cal directory. I detect the chessboard corners to be used as the image points as shown here. 

![](/output_images/camera_calibration/calibration7.jpg){:width="200px"}


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Using the object and image points I calibrate the camera and save its data in a pickle file `calibration_pickle.p`

![](/test_images/test6.jpg){:width="200px"} ![](/output_images/undistorted_images/test6.jpg){:width="200px"} 

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I defined two functions abs_sobel_threshold() and color_threshold() at the beginning of the file `image_generation.py`. I am using both x and y gradient thresholds and HLS + HSV thresholds in lines 64 - 68. Here is one of the binary images:

![](/output_images/binary_images/test3.jpg){:width="200px"}

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

This is my perspective transform code:

```python
img_size = (img.shape[1], img.shape[0])
trap_height_pct = .62
middle_trap_width_pct = .08
bottom_trap_width_pct = .76
avoid_hood_height_pct = .935

src = np.float32([[img_size[0] * (.5 - middle_trap_width_pct / 2), img_size[1] * trap_height_pct],
                    [img_size[0] * (.5 + middle_trap_width_pct / 2), img_size[1] * trap_height_pct],
                    [img_size[0] * (.5 + bottom_trap_width_pct / 2), img_size[1] * avoid_hood_height_pct],
                    [img_size[0] * (.5 - bottom_trap_width_pct / 2), img_size[1] * avoid_hood_height_pct]])
offset = img_size[0] * .25
dest = np.float32([[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]],
                    [offset, img_size[1]]])

m = cv2.getPerspectiveTransform(src, dest)
m_inv = cv2.getPerspectiveTransform(dest, src)
warped = cv2.warpPerspective(processedImage, m, img_size, flags=cv2.INTER_LINEAR)
```

I experimented and manually defined a trapezoid area of interest in the middle of the original image which corresponds to a rectangle area in the warped image. I also trimmed a small area at the bottom to crop the car's hood from the image. The results were satisfactory.

![](/output_images/warped_images/test2.jpg){:width="200px"}


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I have a python class called Slider in `sliding_window.py`. It uses a sliding window logic using a convolution of the window and the vertical slice of the image. I look for peaks in left and right side of the image to detect lane lines.

The Slider class is used in lines 96 - 165 in `image_generation.py`, specifically lines 141 - 147. I am fitting a second order polynomial curve of this equation: 

<img src="https://latex.codecogs.com/gif.latex?f(y)=Ay&space;2&space;&plus;By&plus;C" title="f(y)=Ay 2 +By+C" />

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The position was calculated simply in lines 177 - 182 in `image_generation.py` using the variables from the polynomial. The radius of curvature is calculated on line 185 of `image_generation.py` using this formula: 

<img src="https://latex.codecogs.com/gif.latex?R_{curve}&space;=&space;\frac{(1&space;&plus;&space;(2Ay&plus;B)^{2})^{\frac{3}{2}}&space;}{|2A|}" title="R_{curve} = \frac{(1 + (2Ay+B)^{2})^{\frac{3}{2}} }{|2A|}" />

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

All this resulted in the lane lines detected on the original image:

![](/output_images/final_output/test6.jpg){:width="200px"}

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline worked well on the project video, but it didn't fare so well on the challenge videos. The main reasons for this:

1. I am only considering the trapezoid area for lane lines. When there is a sharp curve (like in harder_challenge_video), they don't get detected.
2. The color of the road is changing in the middle of the lane in the challenge_video. This is messing up the lane detections.
3. Other motorists coming in your lane may be causing a few issues.

Perhaps using a inverted trapezoid (starting from the car's hood and getting widest in the middle of the video) would consider the sharp curves and hopefully the the convolution process for centroid detection would still find the lane lines among the extra noise added.

