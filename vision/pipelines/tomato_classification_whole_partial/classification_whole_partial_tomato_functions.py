
import cv2
import numpy as np
import matplotlib.pyplot as plt
from vision.misc.help_func import validate_output_path
from sklearn.cluster import KMeans

def _segment_red_objects(image):
    # blur before thresholding:
    blurred = cv2.GaussianBlur(image, (5, 5), 4)
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 50, 5])
    upper_red_1 = np.array([8, 255, 255])

    # Broaden the hue range for purple and adjust saturation and value bounds
    lower_red_2 = np.array([130, 50, 5])  # Start from a bluish-purple shade
    upper_red_2 = np.array([180, 255, 255])

    # Generate masks for red regions
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    # Combine the two masks
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply the mask to extract the red regions in the original image
    result = cv2.bitwise_and(image, image, mask=mask)

    return mask, result


def _segment_kmeans(image):

    # Reshape the RGB image to a list of RGB values
    pixels_cv = image.reshape(-1, 3)

    # Apply the k-means clustering algorithm
    kmeans_cv = KMeans(n_clusters=2)
    kmeans_cv.fit(pixels_cv)

    # Determine the labels for the clusters
    labels_cv = kmeans_cv.labels_.reshape(image.shape[:-1])

    # Identify the cluster with a more distinct color (assumed to be the background)
    background_cluster_cv = np.argmin(kmeans_cv.cluster_centers_.sum(axis=1))

    # Create the mask such that tomatoes are white and the background is black
    mask = np.where(labels_cv == background_cluster_cv, 0, 255).astype('uint8')

    # Apply the mask to extract the red regions in the original image
    result = cv2.bitwise_and(image, image, mask=mask)

    return mask, result

def _get_biggest_contour(mask, image):
    # get contour of the biggest red object:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # draw the contour of the biggest red object:
    image_all_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 1)
    # get the biggest area contour, check if empty first:
    if len(contours) == 0:
        return None, image_all_contours, image_all_contours
    biggest_contour = max(contours, key=cv2.contourArea)
    # draw c on image:
    image_biggest_contour = cv2.drawContours(image.copy(), [biggest_contour], 0, (0, 255, 0), 1)
    return biggest_contour, image_all_contours, image_biggest_contour


def _in_area_range(biggest_contour, cropped, thresh=0.7):
    # screen contour by contour area / whole image area:
    contour_area = cv2.contourArea(biggest_contour)
    image_area = cropped.shape[0] * cropped.shape[1]
    cnt_area_proportion = contour_area / image_area
    in_range = cnt_area_proportion > thresh
    print(f'Area proportion: {round(cnt_area_proportion, 2)}, in_range = {in_range}')
    return in_range


def _in_solidity_range(biggest_contour, solidity_thresh=0.88):
    contour_area = cv2.contourArea(biggest_contour)
    # screen by solidity:
    hull = cv2.convexHull(biggest_contour)
    hullArea = cv2.contourArea(hull)
    if hullArea == 0:
         return False
    solidity = contour_area / float(hullArea)
    in_range = solidity > solidity_thresh
    print(f'Solidity: {round(solidity, 2)}, in_range = {in_range}')
    return in_range


def _in_diameters_range(biggest_contour, cropped, thresh=1.2):
    # screen by min enclosed rect:
    rect = cv2.minAreaRect(biggest_contour)
    ##########################################
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # # draw the rotated bounding box on the original image
    # image_rotated_box = cv2.drawContours(cropped.copy(), [box], 0, (0, 255, 0), 1)
    ##########################################
    # get the proportion of both rediused axis:
    long_axis = max(rect[1])
    short_axis = min(rect[1])
    if short_axis == 0 or long_axis == 0:
        return False
    axis_proportion = round(max(long_axis / short_axis, short_axis / long_axis), 2)
    in_range = axis_proportion < thresh
    print(f'Rect_radius_proportion: {axis_proportion}, in_range = {in_range}')
    #print(f'width {round(rect[1][0],2)}/hight {round(rect[1][1],2)} = {round(rect[1][0]/rect[1][1],2)}')
    return in_range

def predict_whole_or_partial(cropped, kmeans_segment = False, show=False):

    # segmentation method:
    if  kmeans_segment:
        red_mask, red_masked_image = _segment_kmeans(cropped)
    else:
        red_mask, red_masked_image = _segment_red_objects(cropped)

    biggest_contour, image_all_contours, image_biggest_contour = _get_biggest_contour(red_mask, cropped)
    if biggest_contour is None:
        pred = 'partial'
    else:
        # 0 = whole tomato, 1 = partial tomato
        pred = 0 if _in_solidity_range(biggest_contour, solidity_thresh=0.88) & _in_area_range(biggest_contour, cropped,
                                                                                                     thresh=0.7) & _in_diameters_range(
            biggest_contour, cropped, thresh=1.2) else 1

    if show:
        h_concat = np.concatenate(
            (cropped, red_masked_image, cv2.cvtColor(red_mask, cv2.COLOR_GRAY2RGB), image_biggest_contour), axis=1)
        plt.imshow(cv2.cvtColor( h_concat, cv2.COLOR_BGR2RGB) )
        plt.axis('off')
        plt.title(f'Pred: {pred}')
        plt.figure(figsize=(160, 20))
        plt.show()

    return pred

def display_image_with_bbox(image, left, upper, right, lower, class_name):
    """
    Draw bounding box and class name on the image.

    Args:
    - image (numpy.ndarray): The image array.
    - left, upper, right, lower (int): Bounding box coordinates.
    - class_name (str): Class name for the bounding box.
    """
    # Define colors for classes. You can expand this dictionary for more classes.
    class_colors = {
        0: (0, 255, 0),  # Green
        1: (0, 0, 255)  # Red
    }

    # Choose the color based on the class name
    color = class_colors.get(class_name, (255, 255, 255))  # Default to white if class is not found

    # Draw bounding box and class name on the image
    cv2.rectangle(image, (left, upper), (right, lower), color, 1)
    cv2.putText(image, str(class_name), (left, upper - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image


def classify_whole_partial(jai_batch, det_outputs, f_id, kmeans_segment = False, save=False, output_dir=''):
    # Crop bboxes from images
    for img, dets in zip(jai_batch, det_outputs):
        image_annotated = img.copy()

        for detection in dets:
            x1, y1, x2, y2 = map(int, detection[:4])
            cropped_img = img[y1:y2, x1:x2]
            cls_pred = predict_whole_or_partial(cropped_img, kmeans_segment, show=False)
            detection[-1] = cls_pred  # Overwrite the class with cls_pred
            display_image_with_bbox(image_annotated, x1, y1, x2, y2, cls_pred)

        if save:
            validate_output_path(output_dir)
            output_image_path = f'{output_dir}/frame_{f_id}.jpg'  # todo for batches bigger than 1 add the position in batch
            cv2.imwrite(output_image_path, image_annotated)
            print(f'Saved: {output_image_path}')

    return det_outputs

