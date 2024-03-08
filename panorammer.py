import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    
    return good_matches, H

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
    width = 0
    height = 0
    for point in layout: 
        if point[0] > height:
            height = point[0]
        if point[1] > width:
            width = point[1]
    centerPoint = (height//2, width//2)
    return(width, height, centerPoint)
    

def initResult(layout_h, layout_w, image):
    base_height = image.shape[0]
    base_width = image.shape[1]
    result_width = base_width * (layout_w + 1)
    result_height = base_height * (layout_h + 1)
    result = np.zeros((result_height, result_width))

    return result

def placeCenterImage(result, img_center, layout_c):
    start_row = img_center.shape[0]*layout_c[0]
    end_row = img_center.shape[0]+start_row
    start_col = img_center.shape[1]*layout_c[1]
    end_col = img_center.shape[1]+start_col
    result[start_row:end_row,start_col:end_col]=img_center
    return result

def stitchLeft(img_base, img_proj, result, match_type):
    return result


def stitch(result, imageB, match_type):
    kpA, fA = detectAndDescribe(np.uint8(result))    
    kpB, fB = detectAndDescribe(imageB)
    good_matches, H = matchKeypoints(kpA, kpB, fA, fB, match_type)
    
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

def panoram(images, layout, match_type):
    layout_w, layout_h, layout_c = getLayoutDetails(layout)
    result = initResult(layout_h, layout_w, images[0])
    idx_center = layout.index(layout_c)
    img_center = images[idx_center]
    result = placeCenterImage(result, img_center, layout_c)
    first_up = layout_c[0] - 1
    first_down = layout_c[0] + 1
    first_left = layout_c[1] - 1
    first_right = layout_c[1] + 1
    if first_up >= 0:
         layout_above = ((layout_c[0]-1, layout_c[1]))
         idx_above = layout.index(layout_above)
         img_above = images[idx_above]
         result = stitch(result, img_above, match_type)
    if first_up <= layout_h:
         layout_below = ((layout_c[0]+1, layout_c[1]))
         idx_below = layout.index(layout_below)
         img_below = images[idx_below]
         result = stitch(result, img_below, match_type)
    if first_left >= 0:
         layout_left = ((layout_c[0], layout_c[1]-1))
         idx_left = layout.index(layout_left)
         img_left = images[idx_left]
         result = stitch(result, img_left, match_type)
    if first_right <= layout_w:
        layout_right = ((layout_c[0], layout_c[1]+1))
        idx_right = layout.index(layout_right)
        img_right = images[idx_right]
        result = stitch(result, img_right, match_type)

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
    
    im1 = cv2.imread('images/budapest1.jpg')
    im2 = cv2.imread('images/budapest2.jpg')
    im3 = cv2.imread('images/budapest3.jpg')
    im4 = cv2.imread('images/budapest4.jpg')
    im5 = cv2.imread('images/budapest5.jpg')
    im6 = cv2.imread('images/budapest6.jpg')
    images = [im1, im2, im3, im4, im5, im6]
    layout = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    for i in range(len(images)):
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
    result = panoram(images, layout, 0)

    plt.figure(figsize=(15, 10))
    plt.imshow(result,'gray')
    plt.xticks([]), plt.yticks([])
    plt.title("It's Pantastic!")
    plt.show()
    

    #imageA = cv2.imread('images/macew1.jpg')
    #imageB = cv2.imread('images/macew7.jpg')
    
    #imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)
    #imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB)
    #result,imMatches,imageBWarped,intersectPoints = stitchOLD(imageA, imageB, 0)
    
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
main()