import cv2
import pygame
import numpy as np
import math
import xml.etree.ElementTree as ET
from pygame import mixer # Load the required library
from PIL import Image
from PIL import ImageFont, ImageDraw
import glob
from collections import OrderedDict

# for the sound stuff
pygame.mixer.init()

# Settings
projectedImageHeight = 1080
projectedImageWidth = 1920
MIN_MATCH_COUNT = 6
MAX_MATCH_COUNT = 20
CAM_WIDTH = 1600
CAM_HEIGHT = 896
DISPLAY_INFO_LOCATION_X = projectedImageWidth * 0.2
DISPLAY_INFO_LOCATION_Y = projectedImageHeight * 0.7
MATRIX_SMOOTHENING_FACTOR = 0.2
DELTA_T = 1 # time interval
trackedCenterPoint = [0, 0]
trackingVelocity = [0, 0]
smoothenedMatrix = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Global variable to hold all celestial bodies
planets = stars = planet_list = []

# images to be loaded
imageToBeProjected = 'solar_system2.png'
shuttleToBeDrawn = 'shuttleIcon.png'

# marker stuff
marker_file_name = ["markers/marker_one_small.png", "markers/marker_two_small.png", "markers/marker_three_small.png", "markers/marker_four_small.png"]
marker_points = [[0, 0], [0, projectedImageHeight - 100], [projectedImageWidth - 100, 0], [projectedImageWidth - 100, projectedImageHeight - 100]]

# initialize the feature detector
# we use orb, make it ready
orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = []
matchesMask = []

"""Specific class which is used to read template images with filenames associated with it"""
class PlanetTemplateImage:
    def __init__(self, img_name):
        self.img = cv2.imread(img_name, 0)
        self.__name = img_name

    def __str__(self):
        return self.__name


"""
Planet class to make things easier to handle.
"""
class Planet(object):
    name = ""
    distanceFromEarth = 0 #in lightyears
    surfaceTemperature = 0 #in celcius
    size = 0 # multiplier only. x times of earth's
    gravity = 0 # multiplier only. x times of earth's
    moons = [] # only the names
    compoundFound = []
    orbitTime = 0 # in days (earth)
    dayTime = 0 # in days (earth)

    def __init__(self, name, distanceFromEarth, size, numberOfMoons, gravity, compoundFound, orbitTime, dayTime, surfaceTemperature):
        self.name = name
        self.distanceFromEarth = distanceFromEarth
        self.size = size
        self.gravity = gravity
        self.numberOfMoons = numberOfMoons
        self.compoundFound = compoundFound
        self.orbitTime = orbitTime
        self.dayTime = dayTime
        self.surfaceTemperature = surfaceTemperature


# Text to display about the celestial body
def prepare_info(planet):

    info = "----------Celestial Body Info" \
           "\n--Name: " + str(planet.name) + \
           "\n--Distance from the Earth: " + str(planet.distanceFromEarth) + " light years" + \
           "\n--Size: " + str(planet.size) + " x of Earth" + \
           "\n--Gravity: " + str(planet.gravity) + " x of Earth" + \
           "\n--Number of Moons: " + str(planet.numberOfMoons) + \
           "\n--Compounds Found: " + str(planet.compoundFound) + \
           "\n--Orbit Time: " + str(planet.orbitTime) + " Earth days" + \
           "\n--Day Time: " + str(planet.dayTime) + " Earth days" + \
           "\n--Surface Temperature: " + str(planet.surfaceTemperature) + " Celcius Degrees"

    return info


def getPlanetPixelLocations(backgroundImage, templates):

    # container to carry the virtual planet boundries
    planetRects = []

    # work with a copy
    backgroundImage_Gray = cv2.cvtColor(backgroundImage.copy(), cv2.COLOR_BGR2GRAY)

    # Apply template Matching
    # loop over the list of templates and draw bounding boxes around them in the image
    for i in range(len(templates)):
        w, h = templates[i].img.shape[::-1]
        name = str(templates[i])
        res = cv2.matchTemplate(backgroundImage_Gray, templates[i].img, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        #cv2.rectangle(backgroundImage, top_left, bottom_right, 255, 2) # comment out to see the bounding boxes
        planetDict = OrderedDict()
        planetDict['name'] = name
        planetDict['tl-br'] = (top_left, bottom_right)

        planetRects.append(planetDict)

    return planetRects


def init_webcam(mirror=False):
    cam = []
    camera_height = []
    camera_width = []
    cam = cv2.VideoCapture(1) # try the external camera first
    cam.set(cv2.CAP_PROP_FPS,50)
    cam.set(cv2.CAP_PROP_EXPOSURE,10)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_HEIGHT)
    try:
        ret_val, camera_image = cam.read()
        if len(camera_image) == 0:
            print('Camera not connected')
        camera_height,camera_width,d = camera_image.shape
    except:
        print("No external camera found, attempting to default to  internal camera")
        cam = cv2.VideoCapture(1)
        ret_val, camera_image = cam.read()
        if len(camera_image) == 0:
            raise
        camera_height,camera_width,d = camera_image.shape
    return cam, camera_height, camera_width


def get_feature_matches(projectionImage_des, cameraImage):
    # Find the keypoints and descriptors with ORB
    cameraImage_kp = orb.detect(cameraImage, None)

    # Compute matches
    matches = []
    if len(cameraImage_kp) > 0:
        cameraImage_kp, cameraImage_des = orb.compute(cameraImage, cameraImage_kp)
        matches = bf.match(projectionImage_des, cameraImage_des)
        if len(matches) > MAX_MATCH_COUNT:
            matches = sorted(matches, key=lambda x: x.distance)[0:MAX_MATCH_COUNT]

    return matches, cameraImage_kp

# Function to find the homography matrix which transforms from the camera image to the projector image
def get_homography(matches, projectionImage_kp, cameraImage_kp):

    homographyMatrix = []

    # Only perform this if there are enough matches
    if len(matches) > MIN_MATCH_COUNT:
        # Taken from the tutorial
        src_pts = np.float32([projectionImage_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([cameraImage_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Get the homography matrix with RANSAC
        homographyMatrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    else:
        print("Not enough matches are found - %d/%d" % (len(matches), MIN_MATCH_COUNT))
        matchesMask = None

    return homographyMatrix, matchesMask


def show_matches(projectionImage, cameraImage, projectionImage_kp, cameraImage_kp, homographyMatrix):
    h,w,d = projectionImage.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts,homographyMatrix)
    cameraImage = cv2.polylines(cameraImage, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    visualizationImage = cv2.drawMatches(projectionImage, projectionImage_kp, cameraImage, cameraImage_kp, matches, None, **draw_params)
    cv2.imshow('Debug', visualizationImage)


# Returns the location of the centre of the camera image in the projector image
def virtual_point(homographyMatrix):
    pts = np.float32([ [round(CAM_WIDTH/2),round(CAM_HEIGHT/2)] ]).reshape(-1,1,2)
    m = cv2.invert(homographyMatrix)
    # find the location of this same point in the projector image
    dst = cv2.perspectiveTransform(pts, m[1])
    return dst


def smoothenMatrix(homographyMatrix):
    for j in range(3):
        for i in range(3):
            smoothenedMatrix[i][j] = smoothenedMatrix[i][j] * (MATRIX_SMOOTHENING_FACTOR) + homographyMatrix[i][j] * (1 - MATRIX_SMOOTHENING_FACTOR)

"""
We need to ensure that we are able to match the features while having a bounding box around the projected image.
"""
def isImageFullyVisible(homographyMatrix, background_height, background_width):
    pts = np.float32([ [0,0],[0,background_height-1],[background_width-1,background_height-1],[background_width-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,homographyMatrix)
    area = cv2.contourArea(dst)
    if area > 100000:
        x,y,w,h = cv2.boundingRect(dst)
        metric = w*h
        error_ratio = abs(metric - area) / area
        if error_ratio < 0.8:
            return True
    else:
        return False

"""
We need to smoothen the center point motion of the camera on the projected image. 
Therefore, we measure how much distance the point takes in a given delta_t time interval.
And we scale it so that it looks smooth and reasonable.
"""
def smoothenCenterMotion(measured_position, delta_t):
    new_point = [0,0]
    for i in range(2):
        new_point[i] = int(round((trackedCenterPoint[i] + trackingVelocity[i] * delta_t) * MATRIX_SMOOTHENING_FACTOR + measured_position[i] * (1 - MATRIX_SMOOTHENING_FACTOR)))
        trackingVelocity[i] = new_point[i] - trackedCenterPoint[i]
        trackedCenterPoint[i] = new_point[i]


# function to overlay a transparent image on background.
def transparentOverlay(backgroundImage, overlayImage, pos=(0, 0), scale=1):
    overlayImage = cv2.resize(overlayImage, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlayImage.shape  # Size of foreground
    rows, cols, _ = backgroundImage.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlayImage image

    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlayImage[i][j][3] / 255.0)  # read the alpha channel
            backgroundImage[x + i][y + j] = alpha * overlayImage[i][j][:3] + (1 - alpha) * backgroundImage[x + i][y + j]
    return backgroundImage


'''Takes 2 vectors and returns the rotation matrix between these 2 vectors'''
def get_camera_rotation(homographyMatrix):
    # Points in the camera frame
    camera_pts = np.float32([[round(CAM_WIDTH / 2), round(CAM_HEIGHT / 2)],
                             [round(10 + CAM_WIDTH / 2), round(10 + CAM_HEIGHT / 2)]]).reshape(-1, 1, 2)
    # Find these points in the projector image
    proj_pts = cv2.perspectiveTransform(camera_pts, homographyMatrix)

    # Find the vectors between the sets of points
    camera_vector = (camera_pts[0][0][0] - camera_pts[1][0][0], camera_pts[0][0][1] - camera_pts[1][0][1])
    proj_vector = (proj_pts[0][0][0] - proj_pts[1][0][0], proj_pts[0][0][1] - proj_pts[1][0][1])

    # change the vectors to unit vectors
    camera_vector = camera_vector / np.absolute(np.linalg.norm(camera_vector))
    proj_vector = proj_vector / np.absolute(np.linalg.norm(proj_vector))

    # calculate the angle between the 2 vectors
    # Change the sign of the angle if the rocket is turning the opposite way to desired
    #sine of the angle
    sinAngle = camera_vector[0] * proj_vector[1] - camera_vector[1] * proj_vector[0]

    #angle between the vectors
    angle = np.arcsin(np.clip(sinAngle, -1.0, 1.0))

    # calculate the 2D rotation matrix from this angle
    #rotation_matrix = np.matrix([[np.cos(angle), -1 * np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    return angle

# used to figure if we landed on the planet by means of pixels
def isInsideRect(currentPos, tl, br):
    return currentPos[0] <= br[0] and currentPos[0] >= tl[0] and currentPos[1] <= br[1] and currentPos[1] >= tl[1]


# in order to avoid divide by zero or infinity errors
def clean_asin(asin_angle_in_radians):
    return min(1, max(asin_angle_in_radians, -1))


def main():

    # for template matching and finding their pixels in the image
    planet_templates = []
    for file in glob.glob("templates/*.png"):
        image = PlanetTemplateImage(file)
        planet_templates.append(image)

    # check if templates are loaded
    if len(planet_templates) == 0:
        print("Planet templates could not get loaded. Please check the paths.")
        exit()

    # play the background sound
    mixer.music.load('sounds/background.mp3')
    mixer.music.play(-1)

    # Get the data from the XML
    solarSystem = ET.parse('planet_info.xml')
    celestialBodies = solarSystem.getroot()

    # Parse everything from XML into global variables
    for cBodies in celestialBodies:
        planets = cBodies.findall("planet")
        stars = cBodies.findall("star")
        if planets:
            for planet in planets:
                planet_list.append(
                    Planet(planet[0].text, planet[1].text, planet[2].text, planet[3].text, planet[4].text,
                           planet[5].text, planet[6].text, planet[7].text, planet[8].text))
        elif stars:  # since there is only one star (sun), we just add it into the list of planets
            for star in stars:
                planet_list.append(
                    Planet(star[0].text, star[1].text, star[2].text, star[3].text, star[4].text, star[5].text,
                           star[6].text, star[7].text, star[8].text))
        else:
            print("Nothing was read from the XML.")

    # Set up the camera
    cam, camera_height, camera_width = init_webcam()
    CAM_WIDTH = camera_width
    CAM_HEIGHT = camera_height

    # Load the image that is going to be projected
    projectionImage = cv2.imread(imageToBeProjected)
    shuttleIcon = cv2.imread(shuttleToBeDrawn, cv2.IMREAD_UNCHANGED) # read with the alpha channel

    # get the planet locations
    planetLocations = getPlanetPixelLocations(projectionImage, planet_templates)

    # Draw markers on the image
    for marker_index, cp in enumerate(marker_points):
        marker_image = cv2.imread(marker_file_name[marker_index])
        h, w, d = marker_image.shape
        projectionImage[cp[1]:cp[1] + h, cp[0]:cp[0] + w] = marker_image.copy()
    h, w, d = projectionImage.shape

    # Create an opencv window to display the projection onto
    cv2.namedWindow("Projector", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Projector', projectionImage)
    cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)

    while True:

        # work with a copy
        processedImage = projectionImage.copy()

        # Find keypoints in the projected image
        orb2 = cv2.ORB_create(nfeatures=500)
        projectionImage_kp = orb2.detect(processedImage, None)
        projectionImage_kp, projectionImage_des = orb2.compute(processedImage, projectionImage_kp)

        # Get an image from the camera
        ret_val, cameraImage = cam.read()

        # Get the matching features in the camera image using the descriptors from the projection image
        matches, cameraImage_kp = get_feature_matches(projectionImage_des, cameraImage)

        # if we can't find any matches, just keep displaying the image and inform the user
        if len(matches) <= MIN_MATCH_COUNT:
            cv2.imshow('Projector', processedImage)
            print('Could not find matches.')
            cv2.waitKey(10)
            continue

        # Now compute the Homography
        homographyMatrix, matchesMask = get_homography(matches, projectionImage_kp, cameraImage_kp)

        # Visualize!
        show_matches(processedImage, cameraImage, projectionImage_kp, cameraImage_kp, homographyMatrix)

        # Get a virtual point on the image
        # Check first if the image is fully visible
        # Update the position to the new found position, otherwise keep the old one
        if isImageFullyVisible(homographyMatrix, projectedImageWidth, projectedImageHeight):
            smoothenMatrix(homographyMatrix)
            virtualPoint = virtual_point(smoothenedMatrix)
            updatedPoint = virtualPoint[0][0]
        else:
            updatedPoint = [p for p in trackedCenterPoint]

        # we don't want scattering or abrupt weird moves, so smoothen the motion
        smoothenCenterMotion(updatedPoint, DELTA_T)

        # get the planet names and respective locations
        for d in planetLocations:
            name, pos = d.values()
            tl = pos[0]
            br = pos[1]
            if isInsideRect(updatedPoint, tl, br):  # if we are inside the boundaries of any
                tmp = name.replace('templates/', '')  # remove the path-related part of the string
                planetname = tmp.replace('.png', '')  # remove the file-related part of the string
                # loop over the objects of planets
                for planet in planet_list:
                    if planet.name == planetname:  # find the one that matches the one we landed
                        # prepare the information of the planet we land
                        info = prepare_info(planet)

                        # get the font
                        fontsize = 20
                        font = ImageFont.truetype("spacefont.ttf", fontsize)

                        # load the image to PIL format
                        img_pil = Image.fromarray(processedImage)

                        # draw the text
                        draw = ImageDraw.Draw(img_pil)
                        draw.text((DISPLAY_INFO_LOCATION_X, DISPLAY_INFO_LOCATION_Y), info, font=font, fill=(0, 255, 255, 0))  # color BGR

                        # back to opencv format
                        processedImage = np.array(img_pil)
                        break

        # rotate the shuttle as the camera does
        # first though, get a copy
        toBeRotatedShuttle = shuttleIcon.copy()
        rows, cols, w = toBeRotatedShuttle.shape
        angle = get_camera_rotation(smoothenedMatrix)
        angleInDegrees = round(math.degrees(clean_asin(angle)), 2)  # convert radian to degrees
        rotationMatrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angleInDegrees, 1)
        toBeRotatedShuttle = cv2.warpAffine(toBeRotatedShuttle, rotationMatrix, (cols, rows), cv2.INTER_LANCZOS4)

        # Overlay transparent images at desired position(x,y) and scale.
        result = transparentOverlay(processedImage, toBeRotatedShuttle, tuple(trackedCenterPoint), 0.7)

        # Display the resulting projector image with a dot for the camera location
        cv2.imshow('Projector', processedImage)
        if cv2.waitKey(20) == ord('a'):
            break

    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # execute only if run as a script
    main()