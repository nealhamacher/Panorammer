import cv2
import numpy as np
from math import floor

# (kps, features) = detectAndDescribe(image)
# input : a grayscale image
# output : keypoints and feature descriptors
# This function can be modified to take an extra argument _feat_type
# that indicates the type of the features (SIFT, ORB ...)
def detectAndDescribe(image):
    SIFT = cv2.SIFT.create()

    kps, features = SIFT.detectAndCompute(image, None)
    return kps, features


# matches, H  = matchKeypoints(kpsA, kpsB, featuresA, featuresB, match_type, ratio=0.75)
# input : two sets of keypoints and features as detected by detectAndDescribe
#          match_type = 0 brute force matching
#          match_type = 1 ration distance using param ration
# output : matches - a list of matched of indeces of matched features
#          (like the one returned by cv2.DescriptorMatcher class)
def matchKeypoints(kpsA, kpsB, featuresA, featuresB, match_type, ratio=0.75):
    matcher = cv2.BFMatcher()
    good_matches = []
    if match_type == 0:
        matches = matcher.match(featuresA, featuresB)
        good_matches = sorted(matches, key=lambda x: x.distance)
    elif match_type == 1:
        matches = matcher.knnMatch(featuresA, featuresB, k=2)
        for f1, f2 in matches:
            if f1.distance < ratio * f2.distance:
                good_matches.append(f1)

        good_matches = sorted(good_matches, key=lambda x: x.distance)

    good_matches = good_matches[:floor(len(good_matches) * 0.15)]

    # Src: https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
    src_pts = np.float32([kpsA[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpsB[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H = cv2.findHomography(dst_pts, src_pts, method=cv2.RANSAC)

    return good_matches, H


def stitch(imageA, imageB):
    kpts1, feats1 = detectAndDescribe(imageA)
    kpts2, feats2 = detectAndDescribe(imageB)

    good_matches, H = matchKeypoints(kpts1, kpts2, feats1, feats2, 1)
    H = H[0]

    imMatches = cv2.drawMatches(imageA, kpts1, imageB, kpts2, good_matches, None, flags=2)

    result = cv2.warpPerspective(imageB, H, dsize=(imageA.shape[1] * 2, imageA.shape[0]))
    result[:imageA.shape[0], :imageA.shape[1]] = imageA

    return result, imMatches