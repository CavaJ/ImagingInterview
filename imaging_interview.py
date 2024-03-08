import cv2
import imutils
import os
import shutil
from collections import defaultdict


def draw_color_mask(img, borders, color=(0, 0, 0)):
    h = img.shape[0]
    w = img.shape[1]

    x_min = int(borders[0] * w / 100)
    x_max = w - int(borders[2] * w / 100)
    y_min = int(borders[1] * h / 100)
    y_max = h - int(borders[3] * h / 100)

    img = cv2.rectangle(img, (0, 0), (x_min, h), color, -1)
    img = cv2.rectangle(img, (0, 0), (w, y_min), color, -1)
    img = cv2.rectangle(img, (x_max, 0), (w, h), color, -1)
    img = cv2.rectangle(img, (0, y_max), (w, h), color, -1)

    return img


def preprocess_image_change_detection(img, gaussian_blur_radius_list=None, black_mask=(5, 10, 5, 0)):
    gray = img.copy()
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gaussian_blur_radius_list is not None:
        for radius in gaussian_blur_radius_list:
            gray = cv2.GaussianBlur(gray, (radius, radius), 0)

    gray = draw_color_mask(gray, black_mask)

    return gray


def compare_frames_change_detection(prev_frame, next_frame, min_contour_area):
    frame_delta = cv2.absdiff(prev_frame, next_frame)
    thresh = cv2.threshold(frame_delta, 45, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    score = 0
    res_cnts = []
    for c in cnts:
        if cv2.contourArea(c) < min_contour_area:
            continue

        res_cnts.append(c)
        score += cv2.contourArea(c)

    return score, res_cnts, thresh


def check_frame_size(prev_frame, next_frame):
    if prev_frame.shape != next_frame.shape:
        print("Previous frame shape: ", prev_frame.shape)  # Example output: (1080, 1920, 3)
        print("Next frame shape: ", next_frame.shape)  # Example output: (720, 1280, 3)
        print("Frame sizes are different")


def check_frame_channels(prev_frame, next_frame):
    if (len(prev_frame.shape) == 3 and len(next_frame.shape) == 2) or (
            len(prev_frame.shape) == 2 and len(next_frame.shape) == 3):
        print("Frame channels are different")


def get_unique_frame_sizes_count(folder_path, image_files):
    unique_frames_dict = defaultdict(int)
    for image_file in image_files:
        image_abs_path = os.path.abspath(os.path.join(folder_path, image_file)).replace('\\', '/')
        image = cv2.imread(image_abs_path)
        if image is not None:
            height, width = image.shape[:2]
            unique_frames_dict[(height, width)] = unique_frames_dict[(height, width)] + 1
    return unique_frames_dict


def get_unique_frame_sizes(folder_path, image_files):
    unique_frames = set()
    for image_file in image_files:
        image_abs_path = os.path.abspath(os.path.join(folder_path, image_file)).replace('\\', '/')
        image = cv2.imread(image_abs_path)
        if image is not None:
            unique_frames.add(image.shape)
    return unique_frames


def get_unique_aspect_ratios(folder_path, image_files):
    unique_aspect_ratios = set()
    for image_file in image_files:
        image_abs_path = os.path.abspath(os.path.join(folder_path, image_file)).replace('\\', '/')
        image = cv2.imread(image_abs_path)
        if image is not None:
            height, width = image.shape[:2]
            unique_aspect_ratios.add(width * 1.0 / height)
    return unique_aspect_ratios


def calculate_min_contour_area(height, width, base_percentage=0.001):
    area = height * width
    return area * base_percentage


# handle anomalies or very low-resolution images.
# Anomalies can be found more robustly by statistical analysis of a dataset.
# here we defined it as follows: if the height is smaller than 120 or width is smaller than 176. To my mind, these
# numbers are reasonable choice for the images taken by surveillance cameras. Resolutions can range from
# 120 x 176 to 4K: https://optiviewusa.com/cctv-video-resolutions/
def is_image_very_low_resolution(image, min_height=120, min_width=176):
    height, width = image.shape[:2]
    return height < min_height or width < min_width


def gaussian_blur_radius_list_based_on_resolution(image):
    if is_image_very_low_resolution(image):
        return None

    height, width = image.shape[:2]

    # very low resolution between 120x176 and 480x640
    if height * width < 480 * 640:
        return [1, 3]
    elif height * width < 720 * 1280:  # 480x640, 619x1100, 675x1200
        return [3, 5]
    else:  # they are higher resolution such as 1080x1920 or 1520x2688
        return [5, 7, 9]


# new_folder_path - path to the new folder, e.g. 'path/to/new_folder'
# source_file_path - source file path, e.g. 'path/to/source_file.txt'
def move_file(new_folder_path, source_file_path):
    # Define the destination file path (in the new folder)
    destination_file_path = (os.path.abspath(os.path.join(new_folder_path, os.path.basename(source_file_path)))
                             .replace("\\", "/"))

    # Create the new folder if it doesn't already exist
    os.makedirs(new_folder_path, exist_ok=True)

    # Move the file to the new folder
    shutil.move(source_file_path, destination_file_path)

    print(f"File moved to: {destination_file_path}")


def copy_file(new_folder_path, source_file_path):
    # Define the destination file path (in the new folder)
    destination_file_path = (os.path.abspath(os.path.join(new_folder_path, os.path.basename(source_file_path)))
                             .replace("\\", "/"))

    # Create the new folder if it doesn't already exist
    os.makedirs(new_folder_path, exist_ok=True)

    # copy the file to the new folder
    # Check if the destination file exists
    if not os.path.exists(destination_file_path):
        shutil.copy(source_file_path, destination_file_path)
        print(f"File copied to: {destination_file_path}")


def preprocess_for_lighting_variations(img):
    # Apply Histogram Equalization.
    # Histogram equalization can help by adjusting the contrast and brightness distribution across the image,
    # making it less sensitive to changes in lighting.
    img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img_y_cr_cb[:, :, 0] = cv2.equalizeHist(img_y_cr_cb[:, :, 0])
    processed_img = cv2.cvtColor(img_y_cr_cb, cv2.COLOR_YCrCb2BGR)
    return processed_img


def group_images_by_camera_id(image_files):
    # using set to keep file name, since file names are unique per file system requirements. Apart from that,
    # insert/lookup/delete operation for the set is O(1).
    grouped_images = defaultdict(set)
    for image_file in image_files:
        # Extract camera_id from the filename
        # Adjust the parsing logic based on the filename format
        # In the dataset, filenames use the following formats:
        # `c%camera_id%-%timestamp%.png`
        # `c%camera_id%_%timestamp%.png`.
        # Timestamps could be in two formats as given in the following file names:
        # `c21_2021_03_26__16_44_04.png`, `c23-1616694872510.png`.
        if "-" in image_file:
            parts = image_file.split('-')
        else:
            parts = image_file.split('_')
        camera_id = parts[0]  # Assuming the camera_id is the first part
        grouped_images[camera_id].add(image_file)
    return grouped_images


def remove_duplicates_within_group(grouped_images, folder_path, action='remove'):
    for camera_id, images in grouped_images.items():
        while images:
            base_image_path = images.pop()  # Remove and get an image from a set
            base_image_abs_path \
                = os.path.abspath(os.path.join(folder_path, base_image_path)).replace('\\', '/')
            base_image = cv2.imread(base_image_abs_path)

            # handle corrupted images or anomalies (height or with smaller than 100).
            # anomalies can be found robustly by statistical analysis of the dataset.
            if base_image is None or is_image_very_low_resolution(base_image):
                print("Image is CUI - corrupted/unsupported/invalid: ",
                      base_image_abs_path)
                if action == 'remove':
                    os.remove(base_image_abs_path)
                if action == 'move':
                    move_file(os.path.abspath(os.path.join(folder_path, "CUI")).replace("\\", "/"),
                              base_image_abs_path)
                continue

            prev_frame = preprocess_for_lighting_variations(base_image)
            prev_frame = preprocess_image_change_detection(prev_frame,
                                                           gaussian_blur_radius_list
                                                           =gaussian_blur_radius_list_based_on_resolution(prev_frame))

            for other_image_path in set(images):  # Create a copy of the set for safe removal
                other_image_abs_path = (os.path.abspath(os.path.join(folder_path, other_image_path))
                                        .replace('\\', '/'))
                other_image = cv2.imread(other_image_abs_path)

                # handle corrupted images or anomalies (height or with smaller than 100).
                # anomalies can be found robustly by statistical analysis of the dataset.
                if other_image is None or is_image_very_low_resolution(other_image):
                    print("Image is CUI - corrupted/unsupported/invalid: ",
                          other_image_abs_path)
                    if action == "remove":
                        os.remove(other_image_abs_path)
                    if action == 'move':
                        move_file(os.path.abspath(os.path.join(folder_path, "CUI")).replace("\\", "/"),
                                  other_image_abs_path)
                    images.remove(other_image_path)  # Remove this image from the comparison set
                    continue

                next_frame = preprocess_for_lighting_variations(other_image)
                next_frame = preprocess_image_change_detection(next_frame,
                                                               gaussian_blur_radius_list
                                                               =gaussian_blur_radius_list_based_on_resolution(
                                                                   next_frame))

                # Adjust width and height if necessary,
                # Get the dimensions of both frames
                height_prev, width_prev = prev_frame.shape[:2]
                height_next, width_next = next_frame.shape[:2]

                # Determine the smaller dimensions
                target_height = min(height_prev, height_next)
                target_width = min(width_prev, width_next)

                # Resize prev_frame if it is larger than next_frame
                # If we are shrinking the image, we should prefer to use INTER_AREA interpolation.
                if (height_prev, width_prev) != (target_height, target_width):
                    prev_frame = cv2.resize(prev_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

                # Similarly, resize next_frame if it is larger than prev_frame
                # If we are shrinking the image, we should prefer to use INTER_AREA interpolation.
                if (height_next, width_next) != (target_height, target_width):
                    next_frame = cv2.resize(next_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

                # possible image dimensions (height, width) to consider (ignoring one and only 6x10 image):
                # (480, 640)
                # (619, 1100)
                # (675, 1200)
                # (1080, 1920)
                # (1520, 2688)
                # for resolutions smaller than 480p (i.e. 480, 640), use `min_contour_area` which is
                # `0.025%` of the image area,
                # for resolutions smaller than 720p (i.e. 720, 1280), such as (480, 640), (619, 1100), (675, 1200)
                # use `min_contour_area`, such as `0.05%` of the image area,
                # considering the less detailed nature of such images.
                # for all other dimension use `0.1%`,
                # given the increased level of detail that higher-resolution images can capture.
                # reference: https://optiviewusa.com/cctv-video-resolutions/

                # the image sizes are the same, comparing either prev_frame or next_frame would suffice.
                curr_height, curr_width = prev_frame.shape[:2]
                if curr_height * curr_width < 480 * 640:
                    score, _, _ = compare_frames_change_detection(prev_frame, next_frame,
                                                                  min_contour_area
                                                                  =calculate_min_contour_area(curr_height,
                                                                                              curr_width,
                                                                                              0.00025))
                elif curr_height * curr_width < 720 * 1280:
                    score, _, _ = compare_frames_change_detection(prev_frame, next_frame,
                                                                  min_contour_area
                                                                  =calculate_min_contour_area(curr_height,
                                                                                              curr_width,
                                                                                              0.0005))
                else:
                    score, _, _ = compare_frames_change_detection(prev_frame, next_frame,
                                                                  min_contour_area=calculate_min_contour_area(
                                                                      curr_height,
                                                                      curr_width))

                # adjust similarity threshold based on image resolution
                if curr_height * curr_width < 480 * 640:
                    similarity_threshold = 500
                elif curr_height * curr_width < 720 * 1280:
                    similarity_threshold = 1000
                else:
                    similarity_threshold = 2000

                if score < similarity_threshold:
                    print("===================================")
                    print(f"Base image abs path: {base_image_abs_path}")
                    if action == 'remove':
                        os.remove(other_image_abs_path)
                        print(f"Duplicate removed: {other_image_abs_path}")
                    if action == 'move':
                        copy_file(os.path.abspath(os.path.join(folder_path, "DUP",
                                                               os.path.splitext(os.path.basename(base_image_path))[0]))
                                  .replace("\\", "/"),
                                  base_image_abs_path)
                        move_file(os.path.abspath(os.path.join(folder_path, "DUP",
                                                               os.path.splitext(os.path.basename(base_image_path))[0]))
                                  .replace("\\", "/"),
                                  other_image_abs_path)
                    print("===================================")
                    images.remove(other_image_path)  # Remove this image from the comparison set


def main(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    # print(get_unique_frame_sizes(folder_path, image_files))
    # print(get_unique_aspect_ratios(folder_path, image_files))
    grouped_images = group_images_by_camera_id(image_files)

    # for camera_id, image_file_paths in grouped_images.items():
    #    print(camera_id, " => ", get_unique_frame_sizes(folder_path, image_file_paths))
    # for camera_id, image_file_paths in grouped_images.items():
    #     print(camera_id, " => ", get_unique_aspect_ratios(folder_path, image_file_paths))
    # for camera_id, image_file_paths in grouped_images.items():
    #     print(camera_id, " => ", get_unique_frame_sizes_count(folder_path, image_file_paths))

    remove_duplicates_within_group(grouped_images, folder_path, action='move')


folder_loc_path = 'dataset'
main(folder_loc_path)
