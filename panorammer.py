import cv2
import numpy as np
import matplotlib.pyplot as plt
from modules.layout import createImageAlignments
from modules.initialization import getLayoutDetails, initResult, placeCenterImage
from modules.stitching import stitch
from modules.cropping import autoCropper


###
# Purpose: Main panorama making function
# Params: images: a list of images of same type (height/width/colourscheme)
#         layout: paired list to images, entries show [row, column] of image 
#                 if no layout is passed, layout will be generated automatically
#         match_type: 0 for brute force, 1 for k-nearest neighbours
#         blend_type: blending to use in areas where images overlap:
#                       0 - no blending (pixel values from result where overlap)
#                       1 - average blending (takes average of two images)
#                       2 - weighted blending (weights pixels based on distance
#                           from edge of overlap, and which image is adjacent)
# Returns: Panorama image
###
def panoram(images, layout=None, match_type=1, blend_type=0):

    # Check for valid colour type type
    if images[0].ndim not in [2,3]:
        raise ValueError("Invalid colour depth")
    n_dim = images[0].ndim

    # Check all images are same colour type (or at least all are colour or all are grey)
    for image in images:
        if image.ndim != n_dim:
            raise ValueError("All images must be same colour type (gray or colour)")
    
    if match_type not in [0,1]:
        raise ValueError("Invalid match_type (0 or 1 accepted)")
    
    if blend_type not in [0,1,2]:
        raise ValueError("Invalid blend_type (0, 1, or 2 accepted)")
    
    # If no layout passed, automatically create one
    if layout==None:
        layout = createImageAlignments(images)
    
    # Find number of images wide, high, and center image in layout graph
    # Use these to initalize result canvas and place center image
    layout_w, layout_h, layout_pt_c = getLayoutDetails(layout)
    idx_center = layout.index(layout_pt_c)
    img_center = images[idx_center]
    result = initResult(layout_h, layout_w, img_center)
    result = placeCenterImage(result, img_center, layout_pt_c)

    # Find layout graph indices going left/right/up/down from center
    first_above = layout_pt_c[0] - 1
    first_below = layout_pt_c[0] + 1
    first_left = layout_pt_c[1] - 1
    first_right = layout_pt_c[1] + 1

    # Stitch images to left of center
    for i in range(first_left, -1, -1):
        layout_pt = (layout_pt_c[0], i)
        if layout_pt in layout:
            idx = layout.index(layout_pt)
            img = images[idx]
            result = stitch(result, img, match_type, blend_type)

    # Stich images to right of center
    for i in range(first_right, layout_w + 1, 1):
        layout_pt = (layout_pt_c[0], i)
        if layout_pt in layout:
            idx = layout.index(layout_pt)
            img = images[idx]
            result = stitch(result, img, match_type, blend_type)

    # Stitch images above center
    for i in range(first_above, -1, -1):
        layout_pt = (i, layout_pt_c[1])
        idx = layout.index(layout_pt)
        img = images[idx]
        result = stitch(result, img, match_type, blend_type)
        for j in range(first_left, -1, -1):
            layout_pt = (i, j)
            if layout_pt in layout:
                idx = layout.index(layout_pt)
                img = images[idx]
                result = stitch(result, img, match_type, blend_type)
        for j in range(first_right, layout_w + 1, 1):
            layout_pt = (i, j)
            if layout_pt in layout:
                idx = layout.index(layout_pt)
                img = images[idx]
                result = stitch(result, img, match_type, blend_type)

    # Stitch images below center
    for i in range(first_below, layout_h + 1, 1):
        layout_pt = (i, layout_pt_c[1])
        if layout_pt in layout:
            idx = layout.index(layout_pt)
            img = images[idx]
            result = stitch(result, img, match_type, blend_type)
        for j in range(first_left, -1, -1):
            layout_pt = (i, j)
            if layout_pt in layout:
                idx = layout.index(layout_pt)
                img = images[idx]
                result = stitch(result, img, match_type, blend_type)
        for j in range(first_right, layout_w + 1, 1):
            layout_pt = (i, j)
            if layout_pt in layout:
                idx = layout.index(layout_pt)
                img = images[idx]
                result = stitch(result, img, match_type, blend_type)

    # Crop panorama
    result = autoCropper(result)

    return result


def main():
    images = []
    img_set = 0
    layout = None

    # MacEwan Images
    if img_set == 0:
        im1 = cv2.imread('images/macewan/macew1.jpg')
        im2 = cv2.imread('images/macewan/macew3.jpg')
        im3 = cv2.imread('images/macewan/macew4.jpg')
        images = [im2, im1, im3]
        layout = [(0,1),(0,0),(0,2)]
        for i in range(len(images)):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
    
    if img_set == 1:
        # BUDAPEST MAP IMAGES
        im1 = cv2.imread('images/budapest/budapest1.jpg')
        im2 = cv2.imread('images/budapest/budapest2.jpg')
        im3 = cv2.imread('images/budapest/budapest3.jpg')
        im4 = cv2.imread('images/budapest/budapest4.jpg')
        im5 = cv2.imread('images/budapest/budapest5.jpg')
        im6 = cv2.imread('images/budapest/budapest6.jpg')
        
        images = [im4, im5, im2, im6, im1, im3]
        layout = [(1, 0), (1, 1), (0, 1), (1, 2), (0, 0), (0, 2)]

        for i in range(len(images)):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
    

    if img_set == 2:
        # BOAT IMAGES - WARNING: takes a long time to run!
        im1 = cv2.imread('images/boat/boat1.jpg')
        im2 = cv2.imread('images/boat/boat2.jpg')
        im3 = cv2.imread('images/boat/boat3.jpg')
        im4 = cv2.imread('images/boat/boat4.jpg')
        im5 = cv2.imread('images/boat/boat5.jpg')
        im6 = cv2.imread('images/boat/boat6.jpg')
        images = [im2, im5, im1, im3, im6, im4]
        layout = [(0,1), (0,4), (0,0), (0,2), (0,5), (0,3)]
        # for i in range(len(images)):
        #     images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        for i in range(len(images)):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

    if img_set == 3:
        im1 = cv2.imread('images/seoul/seoul1.jpg')
        im2 = cv2.imread('images/seoul/seoul2.jpg')
        im3 = cv2.imread('images/seoul/seoul3.jpg')
        images = [im3, im2, im1]
        # for i in range(len(images)):
        #     images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        for i in range(len(images)):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

    if img_set == 4:
        im1 = cv2.imread('images/mural1/mural1.jpg')
        im2 = cv2.imread('images/mural1/mural2.jpg')
        im3 = cv2.imread('images/mural1/mural3.jpg')
        im4 = cv2.imread('images/mural1/mural4.jpg')
        images = [im3, im2, im1, im4]
        for i in range(len(images)):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        # for i in range(len(images)):
        #     images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

    if img_set == 5:
        # Auto-layout doesn't run on these
        im1 = cv2.imread('images/mural2/mural1.jpg')
        im2 = cv2.imread('images/mural2/mural2.jpg')
        im3 = cv2.imread('images/mural2/mural3.jpg')
        im4 = cv2.imread('images/mural2/mural4.jpg')
        im5 = cv2.imread('images/mural2/mural5.jpg')
        im6 = cv2.imread('images/mural2/mural6.jpg')
        images = [im1, im2, im3, im5, im6]
        layout = [(1,0), (1,1), (1,2), (0,1) , (0,2)]
        for i in range(len(images)):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        # for i in range(len(images)):
        #     images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

    if img_set == 6:
        # Auto-layout appears to miss top-right image (ed5)
        im1 = cv2.imread('images/ed1/ed1.jpg')
        im2 = cv2.imread('images/ed1/ed2.jpg')
        im3 = cv2.imread('images/ed1/ed3.jpg')
        im4 = cv2.imread('images/ed1/ed4.jpg')
        im5 = cv2.imread('images/ed1/ed5.jpg')
        im6 = cv2.imread('images/ed1/ed6.jpg')
        images = [im3, im2, im1, im4, im6, im5]
        layout = [(0,1),(1,0),(0,0),(1,1),(1,2),(0,2)]
        for i in range(len(images)):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        # for i in range(len(images)):
        #     images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

    if img_set == 7:
        im1 = cv2.imread('images/ed2/ed1.jpg')
        im2 = cv2.imread('images/ed2/ed2.jpg')
        im3 = cv2.imread('images/ed2/ed3.jpg')
        images = [im3, im1, im2]
        for i in range(len(images)):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        # for i in range(len(images)):
        #     images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

    if img_set == 8:
        im1 = cv2.imread('images/bridge_art2/art1.jpg')
        im2 = cv2.imread('images/bridge_art2/art2.jpg')
        im3 = cv2.imread('images/bridge_art2/art3.jpg')
        im4 = cv2.imread('images/bridge_art2/art4.jpg')
        im5 = cv2.imread('images/bridge_art2/art5.jpg')
        images = [im1, im3, im4, im2, im5]
        for i in range(len(images)):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        # for i in range(len(images)):
        #     images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

    if img_set == 9:
        im1 = cv2.imread('images/bridge_art3/art1.jpg')
        im2 = cv2.imread('images/bridge_art3/art2.jpg')
        im3 = cv2.imread('images/bridge_art3/art3.jpg')
        im4 = cv2.imread('images/bridge_art3/art4.jpg')
        im5 = cv2.imread('images/bridge_art3/art5.jpg')
        im6 = cv2.imread('images/bridge_art3/art6.jpg')
        im7 = cv2.imread('images/bridge_art3/art7.jpg')
        images = [im5, im7, im6, im4, im2, im1, im3]
        #layout = [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6)]
        for i in range(len(images)):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        # for i in range(len(images)):
        #     images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

    result = panoram(
        images = images, 
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