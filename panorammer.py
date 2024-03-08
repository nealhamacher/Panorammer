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

def stitch(imageA, imageB, match_type):
    result = []
    kpA, fA = detectAndDescribe(imageA)    
    kpB, fB = detectAndDescribe(imageB)
    good_matches, H = matchKeypoints(kpA, kpB, fA, fB, match_type)
    
    imageBWarped = cv2.warpPerspective(imageB,H,(imageA.shape[1]*2,imageA.shape[0]))
    result = np.zeros((imageA.shape[0],2*imageA.shape[1]))

    intersect_points = []

    for i in range(imageA.shape[0]):
        has_intersected = False
        for j in range(imageA.shape[1]):
            if(imageBWarped[i][j] != 0):
                result[i][j] = imageA[i][j]/2 + imageBWarped[i][j]/2
                if not has_intersected:
                    intersect_points.append((j,i))
                    has_intersected = True
            else:
                result[i][j] = imageA[i][j]
                
    for i in range(imageA.shape[0]):
        for j in range(imageA.shape[1],imageBWarped.shape[1]):
                if(j==imageA.shape[1] and imageBWarped[i][j]!=0):
                    intersect_points.append((j,i))
                result[i][j] = imageBWarped[i][j]
    imMatches = cv2.drawMatches(imageA, kpA, imageB, kpB, good_matches, None, flags=2)
    return (result, imMatches, imageBWarped, intersect_points)

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
    im1 = cv2.imread('images/macew1.jpg')
    im2 = cv2.imread('images/macew7.jpg')
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    result,imMatches,imageBWarped,intersectPoints = stitch(im1,im2, 0)
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

main()