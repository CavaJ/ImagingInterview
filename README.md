#### What did you learn after looking on our dataset?

There are 1080 images with different width, size and aspect ratio. List of unique image dimensions and aspect ratios respectively per  `camera_id` (separated by `|`):

`c10  =>  (1520, 2688, 3), (480, 640, 3) | 1.768421052631579:1, 1.3333333333333333:1`
`c20  =>  (1080, 1920, 3) | 1.7777777777777777:1`
`c21  =>  (6, 10, 3), (619, 1100, 3), (675, 1200, 3), (1080, 1920, 3) | 1.6666666666666667:1, 1.7770597738287561:1, 1.7777777777777777:1, 1.7777777777777777:1`
`c23  =>  (1080, 1920, 3) | 1.7777777777777777:1`

Aspect ratio is the width of the image divided by the height of the image. There was also one corrupted file, namely `c21_2021_03_27__10_36_36.png` file, where `cv2.imread` call for this image returns `None`.

Another observation from the dataset:


- `c10  =>  defaultdict(<class 'int'>, {(480, 640): 114, (1520, 2688): 12})`
- `c20  =>  defaultdict(<class 'int'>, {(1080, 1920): 324})`
- `c21  =>  defaultdict(<class 'int'>, {(1080, 1920): 142, (6, 10): 1, (675, 1200): 1, (619, 1100): 1})`
- `c23  =>  defaultdict(<class 'int'>, {(1080, 1920): 484})`

`c10  =>  defaultdict(<class 'int'>, {(480, 640): 114, (1520, 2688): 12})` means that for images captured by camera with id `c10`, there are 114 images with (height, width) of (480, 640) and 12 images with (1520, 2688). As you see, there is only one image with (6, 10) dimension, we can discard it (or consider it as anomalous) when we define paremetes such as `min_contour_area` and `similrity_threshold`.


### How does you program work?

Since the smallest meaningful dimension in my dataset (ignoring the anomalous 6x10 image) starts from 480x640, the program removes all images smalle than 480x640.

`cv2.imread()` method returns a `numpy.ndarray` (NumPy N-dimensional array) if the loading of the image is successful. It returns `None` if the image cannot be read because of any reason (like missing file, improper file permissions, unsupported or invalid format). Therefore, the program directly removes corrupted, unsupported or invalid files. 

In the dataset, filenames use the following formats: `c%camera_id%-%timestamp%.png` and `c%camera_id%_%timestamp%.png`. Example file names: `c21_2021_03_26__16_44_04.png`, `c23-1616694872510.png`. I handle both cases separately, when I extract `camera_id`.

Since the requirement is to use the `preprocess_image_change_detection` and `compare_frames_change_detection` functions as they are, I did not touch those functions. However,  `compare_frames_change_detection` is not efficient becuase it does pixel-by-pixel image comparisons. `compare_frames_change_detection` function also requires the pair of images to have the same width and height. Therefore, I shrink the image with larger width and height in the pair to the other image's smaller width and height. In this case, aspect ratio of resized image may not be maintained. `cv2.resize()` function has `interpolation=INTER_LINEAR` as a default interpolation method to resize images. According to opencv docs: "To shrink an image, it will generally look best with `INTER_AREA` interpolation, whereas to enlarge an image, it will generally look best with `INTER_CUBIC` (slow) or `INTER_LINEAR` (faster but still looks OK)", thus I use `interpolation=INTER_AREA`. If maintaining the original aspect ratio is crucial, we might need to consider adding padding to the smaller dimensions instead, but that would involve a more complex handling depending on the specific requirements for analyzing the frame difference.

For the task, we need to do pairwise comparison of images to find out whether they are similarly looking or not. For brute-force algorithm, if we have `n` images, number of comparisons which should be performed is `O(n^2)` which is quadratic and is not efficient when `n` gets larger. In other words, for  `n` images, if we do not compare an image with itself and do comparison from previous to next (not backwards), number of unique comprisons is `n * (n - 1) / 2 `. In our case, for, `n=1080`, this means  `582660` total comparisons. Since I have to use `compare_frames_change_detection` function (i.e. pixel-by-pixel image comparison), I focused on the strategy which significantly reduces the number of comparisons by focusing on likely duplicates within metadata-based groups (in my case, by  `camera_id`) and removing duplicates early in the process. In other words, only images captured by the same camera are compared and if the image is found as a duplicate, it is removed directly and thus not included in the further comparisons. I also thought about using numpy arrays to achieve the same or better effect, however, I am bordered with `compare_frames_change_detection` function as a requirement and this function is not vectorized to be efficient when used with numpy or pandas.



### What values did you decide to use for input parameters and how did you find these values?

#### Adjusting `min_contour_area`:
Given the significant variation in image dimensions, setting `min_contour_area` relative to the total image area seems more appropriate. Since the smallest meaningful dimension in the dataset (ignoring the anomalous 6x10 image) starts from 480x640, which is relatively low-resolution, and goes up to high-resolution images of 1520x2688, a percentage-based calculation for `min_contour_area` can ensure consistency across different resolutions.
For lower-resolution images (e.g., 480x640), considering the less detailed nature of such images, we might set a smaller `min_contour_area`, such as 0.05% of the image area.
For higher-resolution images (e.g., 1520x2688), we can afford to set a slightly higher threshold, such as 0.1% of the image area, given the increased level of detail that higher-resolution images can capture.

#### Adjusting `similarity_threshold`:

