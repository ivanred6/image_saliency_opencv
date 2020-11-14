# Image Saliency Detection using OpenCV
In this mini-project, used as a demonstration of quick use of OpenCV for a Python course I am (at the time of writing) running, we explore Image Saliency Detection.

![Example Saliency Detection Image](https://pyimagesearch.com/wp-content/uploads/2018/07/opencv_saliency_finegrained_players.jpg)

Saliency Detection is the process where image processing and computer vision algorithms are used to locate the most "_salient_" regions of an image. What does this mean? Well, saliency is defined as "_the quality of being particularly noticeable or important_" i.e. the prominent parts of an image, in this case. Our brains and visual systems (eyes), through evolutionary processes, have adapted to rapidly (and rather unconsciously) focus on the most important regions within our visual field. Applying this to computer vision, it allows our systems to pick out key parts of a static image, or video sequence. Use cases of this include the cameras on many cars that pick up road signs, updating the driver of the latest speed limit in force - for example. 

Applications of saliency detection may also be applied to other aspects of computer vision and image processing including: 
* **Object Detection**: Rather than using the _somewhat_ brute-force approach classically deployed (sliding window and image pyramid), only apply the (_admittedly computationally expensive_) detection algorithm to the image's most salient regions. More salient regions should hopefully be more likely to have objects present.
* **Advertising and Marketing**: Apply techniques to validate design logos and advertisements intended to "stand out" and "pop" from a quick or passing glance.
* **Robotics**: Design robotics and potentially autonomous systems with visual / environmental recognition systems similar to our own.

### Saliency Detection Algorithm Variants
1. **Static Saliency**: Relies on image features and statistics to focus and localise on the most prominent image regions.
1. **Motion Saliency**: Typically these rely on video or frame-by-frame input data. Frames are processed, tracking objects that appear to "_move_", with these considered salient.
1. **Objectness**: Saliency Detection algorithms computing "_objectness_" generate a set of "_proposals_" - these are fundamentally just 'bounding boxes' of where objects are thought to exist within an image. 

**Bear in mind - object detection is `not` the same as computing saliency.** 
> The SD algorithm has _no idea_ if the image contains an object of a given type, or not. Rather, it reports areas where it "_thinks_" objects reside within, meaning other processing systems (such as humans, or other algorithms) are responsible for classifying and making any decisions based on this classification/prediction. One benefit of SDs are their speed - useful for real-time applications where we wouldn't be able, or want, to run computationally expensive algorithms over all pixels in all image frames.

#### Checking for OpenCV Saliency Detection Installation
We can check if the `saliency` module has been installed by opening a Python shell and trying to import it...
```python
$ python
>>> import cv2
>>> cv2.saliency
<module 'cv2.saliency'>
```
