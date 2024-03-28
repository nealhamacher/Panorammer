import numpy as np
from modules.matching import detectAndDescribe, matchKeypoints


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