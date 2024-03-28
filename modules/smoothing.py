import cv2

###
# Smoothes intersection with Gaussian Blur - DOES NOT LOOK GOOD AT ALL
###
def smoothIntersection(image, intersectpoints, k_size):
    for point in intersectpoints:
        start_x = point[0] - 2 * k_size
        if start_x < 0:
            start_x = 0
        end_x = point[0] + 2 * k_size + 1
        if end_x > image.shape[1]:
            end_x = image.shape[1]

        start_y = point[1] - 2 * k_size
        if start_y < 0:
            start_y = 0
        end_y = point[1] + 2 * k_size + 1
        if end_y > image.shape[0]:
            end_y = image.shape[0]

        blurred = cv2.GaussianBlur(image[start_y:end_y, start_x:end_x], (k_size, k_size), 0)
        image[start_y:end_y, start_x:end_x] = blurred
    return image
