### What did you learn after looking on our dataset?

There are 1080 images with different width, size and aspect ratio. List of unique image dimensions and aspect ratios respectively per  `camera_id` (separated by `|`):

- `c10  =>  (1520, 2688, 3), (480, 640, 3) | 1.768421052631579:1, 1.3333333333333333:1`
- `c20  =>  (1080, 1920, 3) | 1.7777777777777777:1`
- `c21  =>  (6, 10, 3), (619, 1100, 3), (675, 1200, 3), (1080, 1920, 3) | 1.6666666666666667:1, 1.7770597738287561:1, 1.7777777777777777:1, 1.7777777777777777:1`
- `c23  =>  (1080, 1920, 3) | 1.7777777777777777:1`

Aspect ratio is the width of the image divided by the height of the image. There was also one corrupted file, namely `c21_2021_03_27__10_36_36.png` file, where `cv2.imread` call for this image returns `None`.

Another observation from the dataset:


- `c10  =>  defaultdict(<class 'int'>, {(480, 640): 114, (1520, 2688): 12})`
- `c20  =>  defaultdict(<class 'int'>, {(1080, 1920): 324})`
- `c21  =>  defaultdict(<class 'int'>, {(1080, 1920): 142, (6, 10): 1, (675, 1200): 1, (619, 1100): 1})`
- `c23  =>  defaultdict(<class 'int'>, {(1080, 1920): 484})`

`c10  =>  defaultdict(<class 'int'>, {(480, 640): 114, (1520, 2688): 12})` means that for images captured by camera with id `c10`, there are 114 images with (height, width) of (480, 640) and 12 images with (1520, 2688). As you see, there is only one image with (6, 10) dimension, we can discard it (or consider it as anomalous) when we define paremetes such as `min_contour_area` and `similrity_threshold`.


### How does you program work?

Since the smallest meaningful dimension in my dataset (ignoring the anomalous 6x10 image) starts from 480x640, the program removes all images smaller than 480x640. However, I generalized the code considering possible CCTV resolutions. If the height is smaller than 120 or the width is smaller than 176, the image is discarded. To my mind, these numbers are reasonable choices for the images taken by surveillance cameras. Resolutions can range from `120 x 176` to `4K`: https://optiviewusa.com/cctv-video-resolutions/.


`cv2.imread()` method returns a `numpy.ndarray` (NumPy N-dimensional array) if the loading of the image is successful. It returns `None` if the image cannot be read because of any reason (like missing file, improper file permissions, unsupported or invalid format). Therefore, the program directly removes corrupted, unsupported or invalid files. 

In the dataset, filenames use the following formats: `c%camera_id%-%timestamp%.png` and `c%camera_id%_%timestamp%.png`. Example file names: `c21_2021_03_26__16_44_04.png`, `c23-1616694872510.png`. I handle both cases separately when I extract `camera_id`.

Since the requirement is to use the `preprocess_image_change_detection` and `compare_frames_change_detection` functions as they are, I did not touch those functions. However,  `compare_frames_change_detection` is not efficient becuase it does pixel-by-pixel image comparisons. `compare_frames_change_detection` function also requires the pair of images to have the same width and height. Therefore, I shrink the image with a larger width and height in the pair to the other image's smaller width and height. In this case, the aspect ratio of the resized image may not be maintained. `cv2.resize()` function has `interpolation=INTER_LINEAR` as a default interpolation method to resize images. According to opencv docs: "To shrink an image, it will generally look best with `INTER_AREA` interpolation, whereas to enlarge an image, it will generally look best with `INTER_CUBIC` (slow) or `INTER_LINEAR` (faster but still looks OK)", thus I use `interpolation=INTER_AREA`. If maintaining the original aspect ratio is crucial, we might need to consider adding padding to the smaller dimensions instead, but that would involve a more complex handling depending on the specific requirements for analyzing the frame difference. Apart from that, since we do comparisons of images grouped by camera_id, there are not much images which differ in their aspect ratio. For example, for cameras  `c23` and `c20`, there is only 1080p images; for `c21` only 3 images have different resolution than 1080p, in `c10`, only 12 images have different resolution out of 126, so resizing 12 images will suffice.

For the task, we need to do a pairwise comparison of images to find out whether they are similarly looking or not. For the brute-force algorithm, if we have `n` images, the number of comparisons that should be performed is `O(n^2)` which is quadratic and is not efficient when `n` gets larger. In other words, for  `n` images, if we do not compare an image with itself and do a comparison from previous to next (not backwards), the number of unique comparisons is `n * (n - 1) / 2 `. In our case, for, `n=1080`, this means  `582660` total comparisons. Since I have to use `compare_frames_change_detection` function (i.e. pixel-by-pixel image comparison), I focused on the strategy which significantly reduces the number of comparisons by focusing on likely duplicates within metadata-based groups (in my case, by  `camera_id`) and removing duplicates early in the process. In other words, only images captured by the same camera are compared and if the image is found as a duplicate, it is removed directly and thus not included in the further comparisons. I also thought about using numpy arrays to achieve the same or better effect, however, I am bordered with `compare_frames_change_detection` function as a requirement and this function is not vectorized to be efficient when used with numpy or pandas.



### What values did you decide to use for input parameters and how did you find these values?

###### Adjusting `min_contour_area`:
Given the significant variation in image dimensions, setting `min_contour_area` relative to the total image area seems more appropriate. Since the smallest meaningful dimension in the dataset (ignoring the anomalous 6x10 image) starts from 480x640, which is relatively low-resolution, and goes up to high-resolution images of 1520x2688, a percentage-based calculation for `min_contour_area` can ensure consistency across different resolutions.
For even lower-resolution images, smaller than 480p (i.e. 120x176 and so on), use `min_contour_area` which is `0.025%` of the image area,
For lower-resolution images (e.g., 480x640, 619x1100, 675x1200), considering the less detailed nature of such images, we might set a smaller `min_contour_area`, such as 0.05% of the image area.
For higher-resolution images (e.g., 1080x1920, 1520x2688), we can afford to set a slightly higher threshold, such as 0.1% of the image area, given the increased level of detail that higher-resolution images can capture.
Reference: https://optiviewusa.com/cctv-video-resolutions/
Using `0.025%`, `0.05%` and `0.1%` of the image area as `min_contour_area` are heuristic-based, derived from general practice in image processing tasks.

###### Adjusting gaussuan_blur_radius_list:

I utilized commonly used radii such as `[3, 5]` here. A kernel size `(3,3)` or `(5,5)` is often a good starting point.

###### Adjusting `similarity_threshold`:




#### Any other comments about your solution?

I also thought about resizing all images to a certain height and width (e.g. 120x214), since smaller images can require less computational power to process. However, the effectiveness of contour area as a metric (defined in  `compare_frames_change_detection`) may be limited at such resolutions, and this parameter might need to be adjusted based on the actual differences observed in the resized images or possibly reconsidered in favor of more global image metrics due to the extreme reduction in resolution.



#### What you would suggest to implement to improve data collection of unique cases in future?

For surveillance camera images from parking garages, optimizing duplicate removal requires strategies that account for both the static nature of many elements in the images (e.g., the garage structure) and the dynamic aspects (e.g., cars moving in and out). Here are several tailored optimization strategies:

###### 1. **Region of Interest (ROI) Selection**
- **Concept**: Focus on specific regions within the images where changes are most relevant to our analysis (e.g., parking spots) and ignore regions where changes do not indicate relevant events.
- **Implementation**: Manually define or automatically detect ROIs in the images. Only compare these regions between images for changes or duplicates.
- **Advantage**: Significantly reduces the computational load by limiting the comparison to smaller, relevant portions of the image.

###### 2. **Temporal Filtering**
- **Concept**: Use time metadata to limit comparisons. Images captured close in time are more likely to contain duplicate or near-duplicate content.
- **Implementation**: Sort images by timestamp and compare each image only with its temporally adjacent neighbors within a certain window (e.g., within 5 minutes of each other).
- **Advantage**: Reduces the number of comparisons and leverages the temporal nature of surveillance footage to identify potential duplicates.

###### 3. **Movement Detection Thresholding**
- **Concept**: In parking garage surveillance, significant changes often involve vehicles moving. Minor changes might not be relevant for duplicate detection.
- **Implementation**: Use motion detection algorithms to identify significant changes between frames. Only consider images for further analysis if detected movement exceeds a certain threshold.
- **Advantage**: Filters out images without significant changes, reducing the need for more computationally intensive comparisons.

###### 4. **Machine Learning-Based Classification**
- **Concept**: Train a machine learning model to classify images as duplicates or non-duplicates based on features extracted from the images.
- **Implementation**: Use a dataset of image pairs labeled as duplicates or non-duplicates to train a classifier (e.g., convolutional neural network) that can predict whether a new pair of images are duplicates.
- **Advantage**: Once trained, a model can quickly classify new images with high accuracy, significantly reducing manual comparison efforts.

###### 5. **Incremental Backups with Comparison Checkpoints**
- **Concept**: Instead of processing the entire dataset repeatedly, create incremental backups and compare only new images against a set of comparison checkpoints.
- **Implementation**: Periodically select a subset of images as baselines or checkpoints. Compare new images against these checkpoints instead of the entire dataset.
- **Advantage**: Reduces the dataset size for comparisons over time, improving efficiency as the dataset grows.

###### 6. **Cloud-Based Parallel Processing**
- **Concept**: Leverage cloud computing resources to parallelize the duplicate detection process.
- **Implementation**: Use cloud services (e.g., AWS Lambda, Azure Functions) to distribute the image comparison workload across multiple instances.
- **Advantage**: Can significantly speed up the processing time by utilizing the scalable computing resources available in the cloud.

Implementing these strategies requires an understanding of the specific requirements and constraints of the surveillance system. It's also crucial to balance optimization efforts with the need to maintain the integrity and completeness of the surveillance data.
