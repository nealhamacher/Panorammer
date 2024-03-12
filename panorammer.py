import cv2
import numpy as np
import matplotlib.pyplot as plt

def anyAboveOrBelow(links):
    for i in range(len(links)):
        if links[i] != None:
            if links[i][0] == 1:
                return True
            if links[i][2] == 1:
                return True
    return False
def findImageAlignment(imageA, imageB):
    kps1, features1 = detectAndDescribe(imageA)
    kps2, features2 = detectAndDescribe(imageB)

    matches, temp, pts_source, pts_dst = matchKeypoints(kps1, kps2, features1, features2, 1)

    if (len(matches) < 200):
        return None
    #FIND X ALIGN
    h1, w1 = imageA.shape[:2]
    x1_mid = w1/2
    y1_mid = h1/2

    h2, w2 = imageB.shape[:2]
    x2_mid = w2/2
    y2_mid = h2/2

    aveDistCenterx1 = 0
    aveDistCentery1 = 0
    aveDistCenterx2 = 0
    aveDistCentery2 = 0

    # Find aveage distance of feature points
    for i in range(len(pts_source)):
        aveDistCenterx1 = aveDistCenterx1 + pts_source[i][0][0] - x1_mid
        aveDistCentery1 = aveDistCentery1 + pts_source[i][0][1] - y1_mid
        aveDistCenterx2 = aveDistCenterx2 + pts_dst[i][0][0] - x2_mid
        aveDistCentery2 = aveDistCentery2 + pts_dst[i][0][1] - y2_mid

    aveDistCenterx1 = aveDistCenterx1/len(pts_source)
    aveDistCentery1 = aveDistCentery1/len(pts_source)
    aveDistCenterx2 = aveDistCenterx2/len(pts_source)
    aveDistCentery2 = aveDistCentery2/len(pts_source)

    #If the average keypoint area is within 40% they are considered alined
    # -1 = left/above, 0 = same, 1 = right/below
    x_align, y_align = 0, 0
    x_dist = aveDistCenterx1 - aveDistCenterx2
    if abs(x_dist) < 0.2 * (w1 + w2)/2:
        x_align = 0
    elif (x_dist) > 0:
        x_align = -1
    else:
        x_align = 1

    # If the average keypoint location is within 40% they are considered aligned
    # -1 = left/above, 0 = same, 1 = right/below
    y_dist = aveDistCentery1 - aveDistCentery2
    if abs(y_dist) < 0.2 * (h1 + h2) / 2:
        y_align = 0
    elif (y_dist) > 0:
        y_align = -1
    else:
        y_align = 1

    return y_align, abs(y_dist), x_align, abs(x_dist)


def layoutGoRight(layout, imageLinks, current_index, current_location):
    min_dist = 9999999
    min_index = current_index
    found_one = False
    for i in range(len(imageLinks[current_index])):
        if (imageLinks[current_index][i] is not None
                and imageLinks[current_index][i][2] == -1
                and imageLinks[current_index][i][0] == 0):


            if(imageLinks[current_index][i][3] < min_dist):
                min_dist = imageLinks[current_index][i][3]
                min_index = i
                found_one = True

    if found_one:
        createLayout(layout, imageLinks, min_index, (current_location[0],
                                                                current_location[1] + 1))
    return
def layoutGoDown(layout, imageLinks, current_index, current_location):
    min_dist = 9999999
    min_index = current_index
    found_one = False
    for i in range(len(imageLinks[current_index])):
        if (imageLinks[current_index][i] is not None
                and imageLinks[current_index][i][0] == -1
                and imageLinks[current_index][i][2] == 0):

            if (imageLinks[current_index][i][1] < min_dist):
                min_dist = imageLinks[current_index][i][1]
                min_index = i
                found_one = True

    if found_one:
        createLayout(layout, imageLinks, min_index, (current_location[0] + 1,
                                                     current_location[1]))
    return


# Starts at the the top left and recursively moves down to bottom right
def createLayout(layout, imageLinks, current_index, current_location):
    layout[current_index] = tuple((current_location[0], current_location[1]))
    # Go right
    layoutGoRight(layout, imageLinks, current_index, current_location)
    # Go down
    layoutGoDown(layout, imageLinks, current_index, current_location)


def createImageAlignments(images):
    # This is not efficient, it pointless compares an images key points to itself.
    image_links = []
    for i in range(len(images)):
        cur_links = []
        for j in range(len(images)):

            if j > i:
                cur_links.append(findImageAlignment(images[i], images[j]))
            elif j < i:
                if image_links[j][i] != None:
                    cur_links.append((image_links[j][i][0] * -1,
                                      image_links[j][i][1],
                                      image_links[j][i][2] * -1,
                                      image_links[j][i][3]))
                else:
                    cur_links.append(None)
            else:
                cur_links.append((0, 0, 0, 0))

        image_links.append(cur_links)

    #Find Top Left Image
    cur_links = image_links[0]
    top_left_index = 0
    while anyAboveOrBelow(cur_links):
        for i in range(len(cur_links)):
            if cur_links[i] != None:
                if cur_links[i][0] == 1:
                    cur_links = image_links[i]
                    top_left_index = i
                if cur_links[i][2] == 1:
                    cur_links = image_links[i]
                    top_left_index = i

    layout = np.full(shape=(len(images), 2), fill_value=(0, 0))
    createLayout(layout, image_links, top_left_index, (0, 0))
    layout_list = layout.tolist()
    for i in range(len(layout_list)):
        layout_list[i] = tuple(layout_list[i])
    return layout_list


def detectAndDescribe(image):
    sift = cv2.SIFT_create()
    kps, features = sift.detectAndCompute(image, None)
    return kps, features


def matchKeypoints(kpsA, kpsB, featuresA, featuresB, match_type, ratio=0.75):
    if match_type not in {0,1}:
        return ValueError
    bfMatcher = cv2.BFMatcher()
    
    if match_type == 0:
        matches = bfMatcher.match(featuresA, featuresB)
        matches = sorted(matches, key = lambda x:x.distance)
        n = int(0.15*len(matches))
        good_matches = matches[:n]
    
    elif match_type == 1:
        knnMatches = bfMatcher.knnMatch(featuresA, featuresB, 2)
        good_matches = []
        for i,j in knnMatches:
            if i.distance < ratio*j.distance:
                good_matches.append(i)

    pts_source = np.float32([kpsA[i.queryIdx].pt for i in good_matches]).reshape(-1,1,2)
    pts_dest = np.float32([kpsB[i.trainIdx].pt for i in good_matches]).reshape(-1,1,2)
    
    H, mask = cv2.findHomography(pts_dest, pts_source, cv2.RANSAC, 5.0)
    
    return good_matches, H, pts_source, pts_dest

'''
Legacy stitching function for reference
'''
def stitchOLD(imageA, imageB, match_type):
    result = []
    kpA, fA = detectAndDescribe(imageA)    
    kpB, fB = detectAndDescribe(imageB)
    good_matches, H = matchKeypoints(kpA, kpB, fA, fB, match_type)
    
    imageBWarped = cv2.warpPerspective(imageB,H,(imageA.shape[1]*2,imageA.shape[0]))
    result = np.zeros((imageA.shape[0],2*imageA.shape[1],imageA.shape[2]))
    print(result.shape)
    intersect_points = []

    for i in range(imageA.shape[0]):
        has_intersected = False
        for j in range(imageA.shape[1]):
            if(imageBWarped[i][j][0] != 0):
                if not has_intersected:
                        intersect_points.append((j,i))
                        has_intersected = True
            for k in range(imageA.shape[2]):
                if(imageBWarped[i][j][k] != 0):
                    result[i][j][k] = imageA[i][j][k]/2 + imageBWarped[i][j][k]/2
                else:
                    result[i][j][k] = imageA[i][j][k]
                
    for i in range(imageA.shape[0]):
        for j in range(imageA.shape[1],imageBWarped.shape[1]):
            if(j==imageA.shape[1] and imageBWarped[i][j][0]!=0):
                intersect_points.append((j,i))
            for k in range(imageA.shape[2]):
                result[i][j][k] = imageBWarped[i][j][k]

    imMatches = cv2.drawMatches(imageA, kpA, imageB, kpB, good_matches, None, flags=2)
    return (result, imMatches, imageBWarped, intersect_points)


def getLayoutDetails(layout):
    #Find max row and column index, use result to find centermost point
    height = 0
    width = 0
    for point in layout: 
        if point[0] > height:
            height = point[0]
        if point[1] > width:
            width = point[1]
    centerPoint = (height//2, width//2)
    
    return(width, height, centerPoint)
    
'''
Detemines size of panorama image and initializes
'''
def initResult(layout_h, layout_w, image, colour_type):
    #Find height and width of one image, use to find overall h, w and initialize
    base_height = image.shape[0]
    base_width = image.shape[1]
    result_width = base_width * (layout_w + 1)
    result_height = base_height * (layout_h + 1)
    if(colour_type == 'rgb'):
        result = np.zeros((result_height, result_width, 3))
    else:
        result = np.zeros((result_height, result_width))
    return result

'''
Places centermost image in panorama on the canvas
'''
def placeCenterImage(result, img_center, layout_pt_c, colour_type):
    start_row = img_center.shape[0]*layout_pt_c[0]
    end_row = img_center.shape[0]+start_row
    start_col = img_center.shape[1]*layout_pt_c[1]
    end_col = img_center.shape[1]+start_col
    if colour_type == "rgb":
        result[start_row:end_row,start_col:end_col,0:3]=img_center
    else:
        result[start_row:end_row,start_col:end_col]=img_center
    return result

'''
Stiches rgb (or other 3 colour depth) images into the rgb result
'''
def stitchColour(result, imageB, match_type):
    kpA, fA = detectAndDescribe(np.uint8(result))    
    kpB, fB = detectAndDescribe(imageB)
    good_matches, H, temp1, temp2 = matchKeypoints(kpA, kpB, fA, fB, match_type)
    
    imageBWarped = cv2.warpPerspective(imageB,H,(result.shape[1],result.shape[0]))
    intersect_points = []

    for i in range(result.shape[0]):
        has_intersected = False
        for j in range(result.shape[1]):
            for k in range(result.shape[2]):
                if(imageBWarped[i][j][k] != 0 and result[i][j][k] == 0):
                    if not has_intersected:
                        intersect_points.append((j,i))
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
    
    imageBWarped = cv2.warpPerspective(imageB,H,(result.shape[1],result.shape[0]))
    intersect_points = []

    for i in range(result.shape[0]):
        has_intersected = False
        for j in range(result.shape[1]):
            if(imageBWarped[i][j] != 0 and result[i][j] == 0):
                if not has_intersected:
                    intersect_points.append((j,i))
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
    #Check for valid colour type (rgb and grayscale only supported right now)
    if colour_type not in ['rgb', 'gray', 'grey']:
        raise ValueError("Invalid colour_type")
    if colour_type == 'rgb':
        stitchFunction = stitchColour
    else:
        stitchFunction = stitchGrey


    #Find number of images wide, high, and center image in layout graph
    #Use these to initalize result canvas and place center image
    layout_w, layout_h, layout_pt_c = getLayoutDetails(layout)
    idx_center = layout.index(layout_pt_c)
    img_center = images[idx_center]
    result = initResult(layout_h, layout_w, img_center, colour_type)
    result = placeCenterImage(result, img_center, layout_pt_c, colour_type)

    #Find layout graph indices going left/right/up/down from center
    first_above = layout_pt_c[0] - 1
    first_below = layout_pt_c[0] + 1
    first_left = layout_pt_c[1] - 1
    first_right = layout_pt_c[1] + 1

    #Stitch images to left of center
    for i in range(first_left, -1, -1):
        layout_pt = (layout_pt_c[0], i)
        idx = layout.index(layout_pt)
        img = images[idx]
        result = stitchFunction(result, img, match_type)
    
    #Stich images to right of center
    for i in range(first_right, layout_w+1, 1):
        layout_pt = (layout_pt_c[0], i)
        idx = layout.index(layout_pt)
        img = images[idx]
        result = stitchFunction(result, img, match_type)
    
    #Stitch images above center
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
        for j in range(first_right, layout_w+1, 1):
            layout_pt = (i, j)
            idx = layout.index(layout_pt)
            img = images[idx]
            result = stitchFunction(result, img, match_type)
    
    #Stitch images below center
    for i in range(first_below, layout_h+1, 1):
        layout_pt = (i, layout_pt_c[1])
        idx = layout.index(layout_pt)
        img = images[idx]
        result = stitchFunction(result, img, match_type)
        for j in range(first_left, -1, -1):
            layout_pt = (i, j)
            idx = layout.index(layout_pt)
            img = images[idx]
            result = stitchFunction(result, img, match_type)
        for j in range(first_right, layout_w+1, 1):
            layout_pt = (i, j)
            idx = layout.index(layout_pt)
            img = images[idx]
            result = stitchFunction(result, img, match_type)
     
    return result


def smoothIntersection(image, intersectpoints, k_size):
    for point in intersectpoints:
        start_x = point[0]-2*k_size
        if start_x < 0:
            start_x = 0
        end_x = point[0]+2*k_size+1
        if end_x > image.shape[1]:
            end_x = image.shape[1]


        start_y = point[1]-2*k_size
        if start_y < 0:
            start_y = 0
        end_y = point[1] + 2*k_size+1
        if end_y > image.shape[0]:
            end_y = image.shape[0]

        blurred = cv2.GaussianBlur(image[start_y:end_y, start_x:end_x], (k_size,k_size), 0)
        image[start_y:end_y,start_x:end_x] = blurred
    return image


def main():
    # MACEWAN IMAGES
    images = []
    mode = 4
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
    #BUDAPEST MAP IMAGES 
        im1 = cv2.imread('images/budapest1.jpg')
        im2 = cv2.imread('images/budapest2.jpg')
        im3 = cv2.imread('images/budapest3.jpg')
        im4 = cv2.imread('images/budapest4.jpg')
        im5 = cv2.imread('images/budapest5.jpg')
        im6 = cv2.imread('images/budapest6.jpg')
        images = [im4, im5, im2, im6, im1, im3]

    #layout = [(1, 0), (1, 1), (0, 1), (1, 2), (0, 0), (0, 2)]



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
        img_colour = 'gray'

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
    result = np.uint8(result)
    plt.figure(figsize=(15, 10))

    print(result.ndim)
    if (result.ndim == 2):
        plt.imshow(result, 'gray') #FOR GRAYSCALE IMAGES
    elif (result.ndim == 3):
        plt.imshow(result) #FOR COLOUR IMAGES
    plt.xticks([]), plt.yticks([])
    plt.title("It's Pantastic!")
    plt.show()
    
    #imageA = cv2.imread('images/macew1.jpg')
    #imageB = cv2.imread('images/macew7.jpg')
    
    #imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)
    #imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB)
    #result,imMatches,imageBWarped,intersectPoints = stitchOLD(imageA, imageB, 0)
    
    #LEGACY CODE - SHOWING MATCHES, PRE-STICHED PROJECTIONS - FOR REFERENCE MAY BE USEFUL IN DEBUGGING
    '''
    plt.subplot(2,1,1)
    plt.imshow(im1, 'gray')
    plt.subplot(2,1,2)
    plt.imshow(imageBWarped, 'gray')
    plt.show
    '''

    '''
    plt.figure(figsize=(15, 10))
    plt.subplot(1,2,1)
    plt.imshow(im1,'gray')
    plt.xticks([]), plt.yticks([])
    plt.title("image 1")
    plt.subplot(1,2,2)
    plt.imshow(im2,'gray')
    plt.xticks([]), plt.yticks([])
    plt.title("image 2")
    plt.show()
    '''

    '''
    plt.figure(figsize=(15, 10))
    plt.imshow(imMatches)
    plt.xticks([]), plt.yticks([])
    plt.title("Matches")
    plt.show()
    '''

    '''
    #result = cv2.cvtColor(np.float32(result), cv2.COLOR_GRAY2BGR)
    plt.figure(figsize=(15, 10))
    plt.subplot(2,1,1)

    plt.imshow(result.astype(np.uint8),'gray')
    plt.xticks([]), plt.yticks([])
    plt.title("Panorama")
    #plt.show()
    resultSmooth = smoothIntersection(result, intersectPoints, 3)
    plt.subplot(2,1,2)
    plt.imshow(resultSmooth.astype(np.uint8),'gray')
    plt.xticks([]), plt.yticks([])
    plt.title("Panorama - Smoothed")
    plt.show()
    '''

###############################################################################
if __name__ == "__main__":
    main()