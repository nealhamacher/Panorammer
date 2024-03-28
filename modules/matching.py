import cv2
import numpy as np

###
# Purpose: Detects keypoints and features in an image using the SIFT algorithm
# Inputs: image - The image 
# Returns: kps - The detected keypoints
#          features - The detected features
###
def detectAndDescribe(image):
    sift = cv2.SIFT_create()
    kps, features = sift.detectAndCompute(image, None)
    return kps, features

###
# Purpose: Matches keypoints and features from two images (from detectAndDescribe)
# Inputs: kpsA - The keypoints from the first image 
#         kpsB - The keypoints from the second image
#         featuresA - The features from the first image
#         featuresB - The features from the second image
#         match_type - 0 for brute force, 1 for k-nearest neighbours
#         ratio - The distance ratio for good matches in k-nearest neighbours
# Returns: good_matches - The good matches (top 15% for brute force, distance <r atio for knn)
#          H - The homography matrix to take second image to first image
#          pts_source - The points in the source image space
#          pts_source - The corresponding points after homography is applied
###
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

    H, _ = cv2.findHomography(pts_dest, pts_source, cv2.RANSAC, 5.0)

    return good_matches, H, pts_source, pts_dest