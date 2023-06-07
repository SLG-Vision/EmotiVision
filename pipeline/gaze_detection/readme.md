# Gaze Tracking

> "Robot: ≪Are you looking me?≫"

This repo is part of the Computer Vision course final project. The goal of this first part of the pipeline is to detect faces present in the input of the image stream and analyze each of them to learn if the eyes are gazing into the camera. To achieve this result, we studied a process composed of four different steps: face detection, facial landmark detection to get eye corner position, precise pupil localization and finally gaze estimation using the PnP method.

## Face detection

To realize the first step, needed not only for the gaze tracking but also for the next parts of the project pipeline, we have studied how to detect faces using one of the many different approaches present in the literature. According to [Zhao et Al, 2003] there are three different models: 1) feature-based, where the goal is to detect distinguished features (like eyes and nose) and compute from their geometrical position if a face is present; 2) template-based, where the input is compared with an already present pattern using an SVM, for instance, and 3) appearance-based, where a small rectangular-shaped patch is overlapped on several windows of the input image. Many different authors have proposed methods based on this approach, with the most cited being the one from [Viola and Jones, 2001].
During our first tests, we used the Haar Cascade classifier that is based on the above-mentioned Viola-Jones. The AdaBoost algorithm is applied over a set of Haar features, the differences between the sum of pixels within adjacent rectangular windows of the image scaled with several factors. We applied the OpenCV library and we tested it detecting faces in the same dataset (currently Fer2013, even if in the future we could use AffectNet) that we then will use to train the emotion deep network. We made the same test using a HOG-based algorithm present in dlib, another C++ library that offers several interesting machine learning and computer vision algorithms. It may not be the best testing scenario since the faces in the images are often partially hidden and so undetected, moreover, it does not count false negatives and false positive cases, however, it well represents a realistic real-life situation. The results of the second method were slightly better as shown in the table.  In our opinion, we could reach a better accuracy with some adjustments that we will try in the next weeks.

|  | No face detected | One face detected | Two faces detected | Time needed to do all the detection |
|---|---|---|---|---|
| Hog Dlib | 10932 | 24955 | 0 | 34.18 s |
| Haar Cascade OpenCv | 15272 | 20608 | 7 | 38.38 s |

_Comparison between two different classical face detection algorithms. They were tested on 35887 images containing a single image. They reach an accuracy of 0,70 and 0,57 respectively._

## Facial landmarks detection

For what concern the second step of the process, we get the facial landmarks detection simply using the dlib method. The algorithm is based on a paper from CVPR [Kazemi and Sullivam, 2014], which is based on an ensemble of regression trees to do shape invariant feature selection and to find 68 facial points (as shown [here](https://ibug.doc.ic.ac.uk/media/uploads/images/annotpics/figure_68_markup.jpg)) with a very low error rate. In particular, we look for the position of the eye corners.

## Precise eye center localization

As the third step, we have read in detail the method proposed by [Timm and Barth, 2011]. This method lets us get the exact center of the pupil also in images with low resolution and bad lighting. The searched point can be found by comparing the gradient vector $g_i$ at position $x_i$ with the normalized displacement vector of a possible center $d_i$. Preprocessing and postprocessing are done to obtain optimal results.



$$  c^*=\underset{c}{\arg\min} \frac{1}{N} \sum_{i=1}^{N}(d_i^\top g_i)^2 , $$ 

$$ d_i = \frac{x_i-c}{||x_i-c||_2}, \forall{i}: ||g_i||_2=1 $$

This method requires an input of a cropped image of an eye, so we used the information from the second step to get the needed processing. We then developed a script of the above-presented method, partially following an already existing work [Trishume] and its Python implementation [abhisuri97]. In the future, we will review the already working script to achieve maximum performance.

At this point, having both the eye corners and the pupil center we can compute the final step.  

![Example of the above-reported results](rdg_detected.jpg) 
*Figure 1. Eye corners have been detected and marked in pink, the cropped regions where to find each eye center are contoured in yellow, and, finally, detected pupils positions is the green dot. Please note that the left eye has not been perfectly computed since
it is partially occluded*


## Pose computation

We have not already finished the fourth and last step of this first part but we achieved some theoretical detail. We are going to solve the so-called Perspective-n-Point (PnP) pose computation problem, computing the direction of the gaze starting from the found point. We already found some interesting methods to achieve our goal and in the next weeks, we will try to develop the best one.


# References

- Timm, F., & Barth, E. (2011). Accurate eye centre localisation by means of gradients. Visapp, 11, 125-130
- Vahid Kazemi, Josephine Sullivan; Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014, pp. 1867-1874
- Viola, P., & Jones, M. J. (2004). Robust real-time face detection. International journal of computer vision, 57, 137-154.
- abhisuri97. (n.d.). Abhisuri97/pyeLike: Basic pupil tracking with gradients in Python (based on Fabian Timm's algorithm). GitHub. Retrieved May 1, 2023, from https://github.com/abhisuri97/pyeLike 
- Trishume. (n.d.). Trishume/eyeLike: A webcam based pupil tracking implementation. GitHub. Retrieved May 1, 2023, from https://github.com/trishume/eyeLike 
- Zhao, W., Chellappa, R., & Phillips, P. J. (2003). A. Rosenfeld. Face recognition: a literature survey. ACM Computing Surveys, 35(4), 399-458.
