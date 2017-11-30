## **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/feature.png
[image2]: ./examples/window.png
[image3]: ./examples/predict.png
[image4]: ./examples/siximages.png
[image5]: ./examples/outcome1.png
[image6]: ./examples/outcome2.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### List of files

I have three main working files for this project:

* ./Proj.ipynb: My development of pipeline was mostly done in this file, where I preserved intermediate process steps for further tuning.
* ./process.py: After I completed my tuning, I wrapped the pipeline code in this file so I can easily call it wherever I like. The main feature extraction and the video processing are concluded in this file.
* ./utility.py: Some useful functions for plotting or data processing.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cell titled `1. Features and Channels` of the IPython notebook (and in lines 22 through 51 of the file called `process.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images in the small dataset provided using `smalldata_list(filepath = [])` in lines 39 through 48 of the `utility.py`.  

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image1]

#### 2. Explain how you settled on your final choice of HOG parameters.

It is more like an iterative process. I first choose `HSV` as I thought saturation channel could be a good feature map separating cars and noncars. So I implemented the pipeline and tested with the video. The results were not very satisfactory, so I was then curious about how other color spaces perform. After some iterations, I discovered that the `YUV` color-space seems to be robust against noise so I finally decided to settle on this choice.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the code cell titled `2. SVM Training` of the IPython notebook. I first trained a linear SVM using the kernel 'rbf' via the grid searching function provided by `scikit-learn`. However, it is taking too much time of searching and I found that the performances is not much off from a simple linear SVM. So I then switched back to the linear kernel and using a default C and the performance seems fine. It really reduces the turnover time to allow better debugging and tuning.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in `def find_cars_multi_scales(img, coords, svc, X_scaler, config):` and in lines 110 through 232 of the file called `process.py`.  I did multi-scale window searching from scale 1 to 4 with the base size of 64 x 64 and the overlap of 0.75 (2 cells). The sample image can be seen:

![alt text][image2]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The details of how I optimize the entire pipeline and the classifier is explained in the later section as the improvement is an iterative process involving testing the pipeline with a short duration of "difficult" video clip.

The code for this step is contained in `def find_cars_multi_scales(img, coords, svc, X_scaler, config):` and in lines 110 through 232 of the file called `process.py`. Ultimately I searched on four scales using YUV 3-channel HOG features plus spatially binned color at the size of 32 x 32 and histograms of color in the feature vector, which provided a nice result. The details are given:

```python
config['color_space'] = 'YUV'
config['spatial_size']= (32, 32)
config['hist_bins'] = 32
config['orient'] = 9
config['pix_per_cell'] = 8
config['cell_per_block'] = 2      
config['hog_channel'] = 'ALL'
config['spatial_feat'] = True
config['hist_feat']  = True
config['hog_feat']  = True
```
The HOG map is obtained once and sub-sampled to fit different scales of windows (thanks to the provided code!).
Here are some example images:

![alt text][image3]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained in lines 234 through the end of the file called `process.py`. Details of the implementation are:

* SVM: For the SVM classifier, I use the confidence score from `svc.decision_function()` to determine whether the prediction should be taken or not. I tried with different thresholds from 0.0~0.8 but I ended up using 0.0, which is the same from using `svc.predict()`. This is because setting a threshold here seems redundant to the later processes, but it was a good trick to tune and judge the performance of a classifier.

* Heatmap: After a successful prediction of a box, the coordinate of the box is appended into a box list along with the confidence score. I then use the confidence score to reflect how 'hot' a predicted box should be. This is quite useful as weighting each positive prediction equally seems to be vulnerable to the noises. Using the confidence score can allows the pipeline to be less sensitive to the predictions of low confidence, which occur frequently.
##### Here are six frames and their corresponding heatmaps:

![alt text][image4]

* Temporal filtering: The low-pass filter is applied for the heatmap with the formula describing the low-pass filtering: `heatmap_new = (1-tao) * heatmap_old + tao * energy`. The energy is by using the confidence score and the tao determines the filtering strength (low tao implying stronger temporal filtering). I use 0.5 for the tao, which approximately corresponds to the weighted average of the past ~8 frames. Then the threshold was set to 1.5 to remove any heat points cooler than this value from the heatmap. I then used `scipy.ndimage.measurements.label()` to identify the car blocks in the heatmap and constructed bounding boxes around them.

##### Here the resulting bounding boxes and the integrated heatmap from  `scipy.ndimage.measurements.label()` are drawn onto the last frame in the series:

![alt text][image5]

##### More results from a different set six frames:
![alt text][image6]

* Negative Mining: I also used the negative mining approach by including some false positive samples into the non-vehicle dataset. This method is particularly useful against the false positives. I ended up collecting 88 more negative samples from the project video. In fact, I spent quite a lot of time on this method.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  There are some concerns and issues I would like to explain in details:

* Classifier:
 A linear SVM classifier is helpful in this project with a small dataset, but its performance is not very satisfactory. A more powerful classifier like NN-based one or CNNs should be used along with more sample images.

* Feature tuning:
I spent quite lots of time testing different color-spaces and finally settle on `YUV`. But I really feel like CNN based classifiers could easily resolve this pain as this type of nonlinear mapping could be less sensitive to the CNNs (since CNNs have great nonlinearity).

* Negative mining:
This approach is super "useful" for the project video, but I suspect that it will help at the same level for other unseen samples; however, it does assist in generalizing the model better. The problem is the work is quite tedious and time-consuming.

* Temporal filtering:
Although the filtering helps to mitigate false positives, it does slow down the prediction and may fail to capture some fast-moving object. Specifically, we may wanna detect oncoming vehicles that drive reversely. Thus, tuning of the time constant (inverse of tao used in the project) should consider the objects of interests.
