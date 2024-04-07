import cv2
import numpy as np
import matplotlib.pyplot as plt
from modules.layout import createImageAlignments
from modules.initialization import getLayoutDetails, initResult, placeCenterImage
from modules.stitching import stitch
from modules.cropping import autoCropper


###
# Purpose: Main panorama making function
# Params: images, a list of images of same type (height/width/colourscheme)
#         layout: paired list to images, entries show [row, column] of image 
#                 if no layout is passed, layout will be generated automatically
#         colour_type: cv2 colour scheme of image (rgb or grayscale)
#         match_type: 0 for brute force, 1 for k-nearest neighbours
#         blend_type: blending to use in areas where images overlap:
#                       0 - no blending (pixel values from result where overlap)
#                       1 - average blending (takes average of two images)
#                       2 - weighted blending (weights pixels based on distance
#                           from edge of overlap, and which image is adjacent)
# Returns: Panorama image
###
def panoram(images, colour_type, layout=None, match_type=1, blend_type=0):

    # Check for valid colour type (rgb and grayscale only supported right now)
    if colour_type not in ['rgb', 'gray']:
        raise ValueError("Invalid colour_type ('rgb' or 'gray' accepted)")
    if match_type not in [0,1]:
        raise ValueError("Invalid match_type (0 or 1 accepted)")
    if blend_type not in [0,1,2]:
        raise ValueError("Invalid blend_type (0, 1, or 2 accepted)")
    
    if layout==None:
        layout = createImageAlignments(images)
    
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
        result = stitch(result, img, colour_type, match_type, blend_type)

    # Stich images to right of center
    for i in range(first_right, layout_w + 1, 1):
        layout_pt = (layout_pt_c[0], i)
        idx = layout.index(layout_pt)
        img = images[idx]
        result = stitch(result, img, colour_type, match_type, blend_type)

    # Stitch images above center
    for i in range(first_above, -1, -1):
        layout_pt = (i, layout_pt_c[1])
        idx = layout.index(layout_pt)
        img = images[idx]
        result = stitch(result, img, colour_type, match_type, blend_type)
        for j in range(first_left, -1, -1):
            layout_pt = (i, j)
            idx = layout.index(layout_pt)
            img = images[idx]
            result = stitch(result, img, colour_type, match_type, blend_type)
        for j in range(first_right, layout_w + 1, 1):
            layout_pt = (i, j)
            idx = layout.index(layout_pt)
            img = images[idx]
            result = stitch(result, img, colour_type, match_type, blend_type)

    # Stitch images below center
    for i in range(first_below, layout_h + 1, 1):
        layout_pt = (i, layout_pt_c[1])
        idx = layout.index(layout_pt)
        img = images[idx]
        result = stitch(result, img, colour_type, match_type, blend_type)
        for j in range(first_left, -1, -1):
            layout_pt = (i, j)
            idx = layout.index(layout_pt)
            img = images[idx]
            result = stitch(result, img, colour_type, match_type, blend_type)
        for j in range(first_right, layout_w + 1, 1):
            layout_pt = (i, j)
            idx = layout.index(layout_pt)
            img = images[idx]
            result = stitch(result, img, colour_type, match_type, blend_type)

    #Crop panorama
    result = autoCropper(result)

    return result


def main():
    images = []
    mode = 0
    layout = None

    # MacEwan Images
    if mode == 0:
        im1 = cv2.imread('images/macew1.jpg')
        im2 = cv2.imread('images/macew3.jpg')
        im3 = cv2.imread('images/macew4.jpg')
        images = [im2, im1, im3]
        layout = [(0,1),(0,0),(0,2)]
        for i in range(len(images)):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        img_colour = 'gray'
    
    if mode == 1:
        # BUDAPEST MAP IMAGES
        im1 = cv2.imread('images/budapest1.jpg')
        im2 = cv2.imread('images/budapest2.jpg')
        im3 = cv2.imread('images/budapest3.jpg')
        im4 = cv2.imread('images/budapest4.jpg')
        im5 = cv2.imread('images/budapest5.jpg')
        im6 = cv2.imread('images/budapest6.jpg')
        
        images = [im4, im5, im2, im6, im1, im3]
        layout = [(1, 0), (1, 1), (0, 1), (1, 2), (0, 0), (0, 2)]

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
        # for i in range(len(images)):
        #     images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        # img_colour = 'rgb'
        for i in range(len(images)):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        img_colour = 'gray'

    if mode == 3:
        im1 = cv2.imread('images/seoul1.jpg')
        im2 = cv2.imread('images/seoul2.jpg')
        im3 = cv2.imread('images/seoul3.jpg')
        images = [im3, im2, im1]
        # for i in range(len(images)):
        #     images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        # img_colour = 'rgb'
        for i in range(len(images)):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        img_colour = 'gray'

    result = panoram(
        images = images, 
        colour_type = img_colour, 
        layout = layout, 
        match_type = 1, 
        blend_type = 2)
    result = np.uint8(result)
    plt.figure(figsize=(15, 10))

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