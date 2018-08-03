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


    #BFMatcher with default params
    bf = cv2.BFMatcher()
    #gives k best matches
    matches = bf.knnMatch(des1, des2, k=2)

    #Apply ratio test from paper by D.Lowe
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    return good

def get_homography(camFrame, projFrame):

    #The matching points in the 
    featurePoints =




    #Use sift to find the common features

    #call the homography function to return the homography matrix between the camera and the projector

    #RETURN the homography matrix



#initialise the webcam
cap = cv2.VideoCapture(0)

while (True): #if this is an infinite loop do all other functions have to be called from here.
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 0x1b:  # ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()