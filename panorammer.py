import cv2
import numpy as np
import matplotlib.pyplot as plt


# This function checks in the link function
# Which contains link data for the set
# Link format (y_aline,whether the image at this index in the images folder is above or below this image
#              y_dist, the absolute value of the distance the image is from the middle y-value of this image
#              x_aline,whether the image at this index in the images folder is left or right this image
#              x_dist, the absolute value of the distance the image is from the middle x-value of this image

def anyAboveOrBelow(links):
    for i in range(len(links)):
        if links[i] != None:
            if links[i][0] == 1:
                return True
            if links[i][2] == 1:
                return True
    return False


# Finds the alignmnet of two images respective to each other
# Returns the alignment link data of the images
# Link format (y_aline,whether the image at this index in the images folder is above or below this image
#              y_dist, the absolute value of the distance the image is from the middle y-value of this image
#              x_aline,whether the image at this index in the images folder is left or right this image
#              x_dist, the absolute value of the distance the image is from the middle x-value of this image

def findImageAlignment(imageA, imageB):
    kps1, features1 = detectAndDescribe(imageA)
    kps2, features2 = detectAndDescribe(imageB)

    matches, temp, pts_source, pts_dst = matchKeypoints(kps1, kps2, features1, features2, 1)

    # If the images do not have enough features points matches, assumes that it is a bad match
    if (len(matches) < 200):
        return None

    # Finds the midpoints of height and width for each Image
    h1, w1 = imageA.shape[:2]
    x1_mid = w1 / 2
    y1_mid = h1 / 2

    h2, w2 = imageB.shape[:2]
    x2_mid = w2 / 2
    y2_mid = h2 / 2

    aveDistCenterx1 = 0
    aveDistCentery1 = 0
    aveDistCenterx2 = 0
    aveDistCentery2 = 0

    # Find average distance of feature points from the center
    for i in range(len(pts_source)):
        aveDistCenterx1 = aveDistCenterx1 + pts_source[i][0][0] - x1_mid
        aveDistCentery1 = aveDistCentery1 + pts_source[i][0][1] - y1_mid
        aveDistCenterx2 = aveDistCenterx2 + pts_dst[i][0][0] - x2_mid
        aveDistCentery2 = aveDistCentery2 + pts_dst[i][0][1] - y2_mid

    aveDistCenterx1 = aveDistCenterx1 / len(pts_source)
    aveDistCentery1 = aveDistCentery1 / len(pts_source)
    aveDistCenterx2 = aveDistCenterx2 / len(pts_source)
    aveDistCentery2 = aveDistCentery2 / len(pts_source)

    # If the average keypoint area is within 40% they are considered alined
    # -1 = left/above, 0 = same, 1 = right/below
    x_align, y_align = 0, 0
    x_dist = aveDistCenterx1 - aveDistCenterx2
    if abs(x_dist) < 0.2 * (w1 + w2) / 2:
        x_align = 0
    elif x_dist > 0:
        x_align = -1
    else:
        x_align = 1

    # If the average keypoint location is within 40% they are considered aligned
    # -1 = left/above, 0 = same, 1 = right/below
    y_dist = aveDistCentery1 - aveDistCentery2
    if abs(y_dist) < 0.2 * (h1 + h2) / 2:
        y_align = 0
    elif y_dist > 0:
        y_align = -1
    else:
        y_align = 1

    return y_align, abs(y_dist), x_align, abs(x_dist)


# Recursive Function for finding the closest image to the right of the current image
# Layout, the returned Layout
# Current Index, the current index in the images link
# Current Location, the current location of the panorma ex (0, 1) (2, 3) etc
def layoutGoRight(layout, imageLinks, current_index, current_location):
    min_dist = 9999999
    min_index = current_index
    found_one = False
    for i in range(len(imageLinks[current_index])):

        # Checks if the image being checked is a neighbor (Non-neighbors are none)
        # -1 checks if its to the right, 0 is checking if it is horizontally aligned
        if (imageLinks[current_index][i] is not None
                and imageLinks[current_index][i][2] == -1
                and imageLinks[current_index][i][0] == 0):

            # If the distance between this image and the next is smaller, use it
            if (imageLinks[current_index][i][3] < min_dist):
                min_dist = imageLinks[current_index][i][3]
                min_index = i
                found_one = True

    # If an image was found that is to the right, recursivly tries to find the next to the right
    # and down of it
    if found_one:
        createLayout(layout, imageLinks, min_index, (current_location[0],
                                                     current_location[1] + 1))
    return


# Recursive Function for finding the closest image to the down of the current image
# Layout, the returned Layout
# Current Index, the current index in the images link
# Current Location, the current location of the panorma ex (0, 1) (2, 3) etc
def layoutGoDown(layout, imageLinks, current_index, current_location):
    min_dist = 9999999
    min_index = current_index
    found_one = False
    for i in range(len(imageLinks[current_index])):

        # Checks if the image being checked is a neighbor (Non-neighbors are none)
        # -1 checks if its to the below, 0 is checking if it is horizontally aligned
        if (imageLinks[current_index][i] is not None
                and imageLinks[current_index][i][0] == -1
                and imageLinks[current_index][i][2] == 0):

            # If the distance between this image and the next is smaller, use it

            if (imageLinks[current_index][i][1] < min_dist):
                min_dist = imageLinks[current_index][i][1]
                min_index = i
                found_one = True

    # If an image was found that is to the below, recursivly tries to find the next to the right
    # and down of it
    if found_one:
        createLayout(layout, imageLinks, min_index, (current_location[0] + 1,
                                                     current_location[1]))
    return


# Reversibly creates the layout of the panorama. Starts at the top left image and hops down and right
# to the closest image until it reaches the bottom right, and all images should have been passed through
# This will not work for all panoramas, if a panoramas is shaped like a backwards L there is no objective
# Top left and the algorithm will not go to the left
def createLayout(layout, imageLinks, current_index, current_location):
    layout[current_index] = tuple((current_location[0], current_location[1]))
    # Go right
    layoutGoRight(layout, imageLinks, current_index, current_location)
    # Go down
    layoutGoDown(layout, imageLinks, current_index, current_location)


# This creates the layout to be used by the panorma function
def createImageAlignments(images):
    image_links = []

    # Loops through all the images and find how each of them relates to each other in terms of location
    for i in range(len(images)):
        cur_links = []
        for j in range(len(images)):
            # Finds the image alignment if the images have not been matched with each other
            # Link format (y_aline,whether the image at this index in the images folder is above or below this image
            #              y_dist, the absolute value of the distance the image is from the middle y-value of this image
            #              x_aline,whether the image at this index in the images folder is left or right this image
            #              x_dist, the absolute value of the distance the image is from the middle x-value of this image
            if j > i:
                cur_links.append(findImageAlignment(images[i], images[j]))

            # If an image Link was already calculated, uses the previous entry, just flips the relationship
            # from left to right and top to bottom as this is from the other perspective
            # Distance is absolute and not flipped
            elif j < i:
                if image_links[j][i] is not None:
                    cur_links.append((image_links[j][i][0] * -1,
                                      image_links[j][i][1],
                                      image_links[j][i][2] * -1,
                                      image_links[j][i][3]))
                # If the image is not adjacent the relationship is none
                else:
                    cur_links.append(None)

            # If the image is the same image it is in the same spot
            else:
                cur_links.append((0, 0, 0, 0))

        image_links.append(cur_links)

    # Finds the index in the top left of the panorma
    cur_links = image_links[0]
    top_left_index = 0

    # Loops to find the top left image
    while anyAboveOrBelow(cur_links):
        for i in range(len(cur_links)):
            if cur_links[i] is not None:
                if cur_links[i][0] == 1:
                    cur_links = image_links[i]
                    top_left_index = i
                if cur_links[i][2] == 1:
                    cur_links = image_links[i]
                    top_left_index = i

    # Finds the layout of the image
    layout = np.full(shape=(len(images), 2), fill_value=(0, 0))

    # Recursively creates the layout from the links of the image
    createLayout(layout, image_links, top_left_index, (0, 0))

    # Turns the layout into a list and converts all arrays into tuples which are needed by the stitcher
    layout_list = layout.tolist()
    for i in range(len(layout_list)):
        layout_list[i] = tuple(layout_list[i])
    return layout_list


def detectAndDescribe(image):
    sift = cv2.SIFT_create()
    kps, features = sift.detectAndCompute(image, None)
    return kps, features


def matchKeypoints(kpsA, kpsB, featuresA, featuresB, match_type, ratio=0.75):
    if match_type not in {0, 1}:
        return ValueError
    bfMatcher = cv2.BFMatcher()

    if match_type == 0:
        matches = bfMatcher.match(featuresA, featuresB)
        matches = sorted(matches, key=lambda x: x.distance)
        n = int(0.15 * len(matches))
        good_matches = matches[:n]

    elif match_type == 1:
        knnMatches = bfMatcher.knnMatch(featuresA, featuresB, 2)
        good_matches = []
        for i, j in knnMatches:
            if i.distance < ratio * j.distance:
                good_matches.append(i)

    pts_source = np.float32([kpsA[i.queryIdx].pt for i in good_matches]).reshape(-1, 1, 2)
    pts_dest = np.float32([kpsB[i.trainIdx].pt for i in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_dest, pts_source, cv2.RANSAC, 5.0)

    return good_matches, H, pts_source, pts_dest


def findCropCol(image, column, direction, delta):
    print(column)
    if delta == 1 or delta == 0:
        return column
    outsideImage = True
    for row in range(0,image.shape[0]): # If any pixels are non-zero (not all black)
        if image[row, column] != 0:
            print("["+str(row)+","+str(column)+"]: "+str(image[row,column]))
            outsideImage = False
            break
    if not outsideImage:
        new_col = column + (direction * delta)
    else:
        new_col = column + (-direction * delta)
    new_delta = delta//2
    return (findCropCol(image, new_col, direction, new_delta))


def autoCropper(image):
    center_col = image.shape[1]//2
    center_row = image.shape[0]//2
    print("Center col " + str(center_col))
    print("Center row " + str(center_row))
    start_col = findCropCol(image, center_col, -1, center_col//2,)
    end_col = findCropCol(image, center_col, 1, center_col//2)
    print("Start col " + str(start_col))
    print("End col " + str(end_col))
    result = np.zeros((image.shape[0],end_col-start_col))
    result[:,:] = image[:,start_col:end_col]
    return result



def getLayoutDetails(layout):
    # Find max row and column index, use result to find centermost point
    height = 0
    width = 0
    for point in layout:
        if point[0] > height:
            height = point[0]
        if point[1] > width:
            width = point[1]
    centerPoint = (height // 2, width // 2)

    return (width, height, centerPoint)


'''
Detemines size of panorama image and initializes
'''


def initResult(layout_h, layout_w, image, colour_type):
    # Find height and width of one image, use to find overall h, w and initialize
    base_height = image.shape[0]
    base_width = image.shape[1]
    result_width = base_width * (layout_w + 1)
    result_height = base_height * (layout_h + 1)
    if (colour_type == 'rgb'):
        result = np.zeros((result_height, result_width, 3))
    else:
        result = np.zeros((result_height, result_width))
    return result


'''
Places centermost image in panorama on the canvas
'''


def placeCenterImage(result, img_center, layout_pt_c, colour_type):
    start_row = img_center.shape[0] * layout_pt_c[0]
    end_row = img_center.shape[0] + start_row
    start_col = img_center.shape[1] * layout_pt_c[1]
    end_col = img_center.shape[1] + start_col
    if colour_type == "rgb":
        result[start_row:end_row, start_col:end_col, 0:3] = img_center
    else:
        result[start_row:end_row, start_col:end_col] = img_center
    return result


'''
Stiches rgb (or other 3 colour depth) images into the rgb result
'''


def stitchColour(result, imageB, match_type):
    kpA, fA = detectAndDescribe(np.uint8(result))
    kpB, fB = detectAndDescribe(imageB)
    good_matches, H, temp1, temp2 = matchKeypoints(kpA, kpB, fA, fB, match_type)

    imageBWarped = cv2.warpPerspective(imageB, H, (result.shape[1], result.shape[0]))
    intersect_points = []

    for i in range(result.shape[0]):
        has_intersected = False
        for j in range(result.shape[1]):
            for k in range(result.shape[2]):
                if (imageBWarped[i][j][k] != 0 and result[i][j][k] == 0):
                    if not has_intersected:
                        intersect_points.append((j, i))
                        has_intersected = True
                    result[i][j][k] = imageBWarped[i][j][k]
    return result


'''
Stitches a grayscale image into the grayscale result
'''


def stitchGrey(result, imageB, match_type):
    kpA, fA = detectAndDescribe(np.uint8(result))
    kpB, fB = detectAndDescribe(imageB)
    good_matches, H, temp1, temp2 = matchKeypoints(kpA, kpB, fA, fB, match_type)

    imageBWarped = cv2.warpPerspective(imageB, H, (result.shape[1], result.shape[0]))
    intersect_points = []

    for i in range(result.shape[0]):
        has_intersected = False
        for j in range(result.shape[1]):
            if (imageBWarped[i][j] != 0 and result[i][j] == 0):
                if not has_intersected:
                    intersect_points.append((j, i))
                    has_intersected = True
                result[i][j] = imageBWarped[i][j]
    return result


'''
Main panorama making function
Params: images, a list of images of same type (height/width/colourscheme)
        layout: paired list for images, showing row and column of image in final order
        colour_type: cv2 colour scheme of image (rgb or grayscale)
        match_type: 0 for brute force, 1 for k-nearest neighbours
Returns: Panorama image
'''


def panoram(images, layout, colour_type, match_type):
    # Check for valid colour type (rgb and grayscale only supported right now)
    if colour_type not in ['rgb', 'gray', 'grey']:
        raise ValueError("Invalid colour_type")
    if colour_type == 'rgb':
        stitchFunction = stitchColour
    else:
        stitchFunction = stitchGrey

    # Find number of images wide, high, and center image in layout graph
    # Use these to initalize result canvas and place center image
    layout_w, layout_h, layout_pt_c = getLayoutDetails(layout)
    idx_center = layout.index(layout_pt_c)
    img_center = images[idx_center]
    result = initResult(layout_h, layout_w, img_center, colour_type)
    result = placeCenterImage(result, img_center, layout_pt_c, colour_type)

    # Find layout graph indices going left/right/up/down from center
    first_above = layout_pt_c[0] - 1
    first_below = layout_pt_c[0] + 1
    first_left = layout_pt_c[1] - 1
    first_right = layout_pt_c[1] + 1

    # Stitch images to left of center
    for i in range(first_left, -1, -1):
        layout_pt = (layout_pt_c[0], i)
        idx = layout.index(layout_pt)
        img = images[idx]
        result = stitchFunction(result, img, match_type)

    # Stich images to right of center
    for i in range(first_right, layout_w + 1, 1):
        layout_pt = (layout_pt_c[0], i)
        idx = layout.index(layout_pt)
        img = images[idx]
        result = stitchFunction(result, img, match_type)

    # Stitch images above center
    for i in range(first_above, -1, -1):
        layout_pt = (i, layout_pt_c[1])
        idx = layout.index(layout_pt)
        img = images[idx]
        result = stitchFunction(result, img, match_type)
        for j in range(first_left, -1, -1):
            layout_pt = (i, j)
            idx = layout.index(layout_pt)
            img = images[idx]
            result = stitchFunction(result, img, match_type)
        for j in range(first_right, layout_w + 1, 1):
            layout_pt = (i, j)
            idx = layout.index(layout_pt)
            img = images[idx]
            result = stitchFunction(result, img, match_type)

    # Stitch images below center
    for i in range(first_below, layout_h + 1, 1):
        layout_pt = (i, layout_pt_c[1])
        idx = layout.index(layout_pt)
        img = images[idx]
        result = stitchFunction(result, img, match_type)
        for j in range(first_left, -1, -1):
            layout_pt = (i, j)
            idx = layout.index(layout_pt)
            img = images[idx]
            result = stitchFunction(result, img, match_type)
        for j in range(first_right, layout_w + 1, 1):
            layout_pt = (i, j)
            idx = layout.index(layout_pt)
            img = images[idx]
            result = stitchFunction(result, img, match_type)

    return result


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


def main():
    # MACEWAN IMAGES
    images = []
    mode = 1

    if mode == 0:
        im1 = cv2.imread('images/macew1.jpg')
        im2 = cv2.imread('images/macew3.jpg')
        im3 = cv2.imread('images/macew4.jpg')
        images = [im2, im1, im3]
        # layout = [(0,1),(0,0),(0,2)]
        for i in range(len(images)):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        img_colour = 'rgb'
    
    if mode == 1:
        # BUDAPEST MAP IMAGES
        im1 = cv2.imread('images/budapest1.jpg')
        im2 = cv2.imread('images/budapest2.jpg')
        im3 = cv2.imread('images/budapest3.jpg')
        im4 = cv2.imread('images/budapest4.jpg')
        im5 = cv2.imread('images/budapest5.jpg')
        im6 = cv2.imread('images/budapest6.jpg')
        images = [im4, im5, im2, im6, im1, im3]

    # layout = [(1, 0), (1, 1), (0, 1), (1, 2), (0, 0), (0, 2)]

        for i in range(len(images)):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        img_colour = 'gray'
    

    if mode == 2:
        # # BOAT IMAGES - WARNING: takes a long time to run!
        im1 = cv2.imread('images/boat1.jpg')
        im2 = cv2.imread('images/boat2.jpg')
        im3 = cv2.imread('images/boat3.jpg')
        im4 = cv2.imread('images/boat4.jpg')
        im5 = cv2.imread('images/boat5.jpg')
        im6 = cv2.imread('images/boat6.jpg')
        images = [im2, im5, im1, im3, im6, im4]
        # layout = [(0,1), (0,4), (0,0), (0,2), (0,5), (0,3)]
        for i in range(len(images)):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        img_colour = 'rgb'

    if mode == 3:
        im1 = cv2.imread('images/seoul1.jpg')
        im2 = cv2.imread('images/seoul2.jpg')
        im3 = cv2.imread('images/seoul3.jpg')
        images = [im3, im2, im1]
        for i in range(len(images)):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        img_colour = 'rgb'

    layout = createImageAlignments(images)
    print(layout)
    result = panoram(images, layout, img_colour, 1)
    # result = cv2.imread('./Budapest.png')
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result = autoCropper(result)
    result = np.uint8(result)
    plt.figure(figsize=(15, 10))

    print(result.ndim)
    if (result.ndim == 2):
        plt.imshow(result, 'gray')  # FOR GRAYSCALE IMAGES
    elif (result.ndim == 3):
        plt.imshow(result)  # FOR COLOUR IMAGES
    plt.xticks([]), plt.yticks([])
    plt.title("It's Pantastic!")
    plt.show()

###############################################################################
if __name__ == "__main__":
    main()
