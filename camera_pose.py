import numpy as np
import cv2

MIN_MATCH_COUNT = 10


def get_feature_match(camFrame, projFrame):
    #Get the images and convert them to gray scale
    img1 = cv2.imread(camFrame)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img2 = cv2.imread(projFrame)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #initialise sift
    sift = cv2.SIFT()
    #find the keypoints and descriptors using sift
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    #FLANN based matcher
    #FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    #ratio test which can be found in D.Lowe's paper to store only good matches
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)


    return good, kp1, kp2


def get_homography(featurePoints, kp1, kp2):
    #check that there is enough matches
    if len(featurePoints)>MIN_MATCH_COUNT:
        src_pts = np.float32([kp2[m.projIdx].pt for m in featurePoints]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.cameraIdx].pt for m in featurePoints]).reshape(-1,1,2)

        #get the homography matrix
        matrix = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)

        return matrix
    else:
        #if a homography matrix could not be found return 0
        return 0







    #Use sift to find the common features

    #call the homography function to return the homography matrix between the camera and the projector

    #RETURN the homography matrix



#initialise the webcam
cap = cv2.VideoCapture(0)

while (True): #if this is an infinite loop do all other functions have to be called from here.
    # Capture frame-by-frame
    ret, frame = cap.read()

    #Get the best matching features from the images and the key points from each image
    featurePoints, kp1, kp2 = get_feature_match(frame, projFrame)
    #get the homography matrix from these features
    get_homography(featurePoints, kp1, kp2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 0x1b:  # ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()








#def get_feature_match(camFrame, projFrame):
    #Get the images and convert them to gray scale
    #img1 = cv2.imread(camFrame)
    #gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    #img2 = cv2.imread(projFrame)
    #gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #initialise sift
    #sift = cv2.SIFT()
    #find the keypoints and descriptors using sift
    #kp1, des1 = sift.detectAndCompute(gray1, None)
    #kp2, des2 = sift.detectAndCompute(gray2, None)


    #BFMatcher with default params
    #bf = cv2.BFMatcher()
    #gives k best matches
    #matches = bf.knnMatch(des1, des2, k=2)

    #Apply ratio test from paper by D.Lowe
    #good = []
    #for m, n in matches:
    #    if m.distance < 0.75*n.distance:
    #        good.append([m])

    #return good