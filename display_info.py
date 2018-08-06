import cv2
import pygame
import numpy as np
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import xml.etree.ElementTree as ET
from pygame import mixer # Load the required library


# global variable to hold everything
planets = stars = planet_list = []

# for the sound stuff
pygame.mixer.init()

"""
Planet class to make things easier to handle.

"""
class Planet(object):
    name = ""
    distanceFromEarth = 0 #in lightyears
    size = 0 # multiplier only. x times of earth's
    gravity = 0 # multiplier only. x times of earth's
    moons = [] # only the names
    elementsFound = []
    orbitTime = 0 # in days (earth)
    dayTime = 0 # in days (earth)

    def __init__(self, name, distanceFromEarth, size, moons, gravity, elementsFound, orbitTime, dayTime):
        self.name = name
        self.distanceFromSun = distanceFromEarth
        self.size = size
        self.gravity = gravity
        self.moons = moons
        self.elementsFound = elementsFound
        self.orbitTime = orbitTime
        self.dayTime = dayTime

def click_and_display(event, x, y, flags, param):
    # grab references to the global variables
    global img

    #TODO: We need to get a new image frame every time the user clicks,
    #TODO: otherwise we keep writing on the same image

    # if the left mouse button was clicked, record the starting
    if event == cv2.EVENT_LBUTTONUP:
        # print(x,y)
        # draw a circle on where we clicked
        cv2.circle(img, (x,y), 3, (0,0,255), thickness=-1, lineType=8) # color BGR

        # display info
        #TODO: Here we need to find out which planet is selected,
        #TODO: by getting the coordinates from the image.
        #TODO: However, we need to find a clever way, unlike what is shown below
        """if(x == 100 && y == 150):
            planet = planet_list.find('Mercury')
        elif(x == 200 && y == 250):
            planet = planet_list.find('Venus')"""

        # play the info effect
        #click_effect = pygame.mixer.Sound('info.wav')
        #click_effect.play()


        # for the time being we only print the info of the sun, which is the 0th element
        # once we figure out which celestial body is selected, we need to pass that specific planet with:
        # info = prepare_info(planet)
        info = prepare_info(planet_list[0]) # index 0 is the sun

        # get the font
        fontsize = 10
        font = ImageFont.truetype("spacefont.ttf", fontsize)

        # load the image to PIL format
        img_pil = Image.fromarray(img)

        # draw the font
        draw = ImageDraw.Draw(img_pil)
        clickingOffset = 50 # how far should the information be displayed (in pixels)
        draw.text((x + clickingOffset, y), info, font=font, fill=(0,0,255,0)) # color BGR

        # back to opencv format
        img = np.array(img_pil)

        # add a line to the info
        textOffset = 90 # enough long to cover the text body on X axis
        cv2.line(img, (x, y), (x+clickingOffset + textOffset, y), (0, 0, 255), 1)

        # display it
        cv2.imshow("image", img)

def prepare_info(planet):

    info = "-Celestial Body Info" \
           "\n--Name: " + str(planet.name) + \
           "\n--Distance from the Earth: " + str(planet.distanceFromEarth) + " lightyears" + \
           "\n--Size: " + str(planet.size) + " x of Earth" + \
           "\n--Gravity: " + str(planet.gravity) + " x of Earth" + \
           "\n--Moons: " + str(planet.moons) + \
           "\n--Elements Found: " + str(planet.elementsFound) + \
           "\n--Orbit Time: " + str(planet.orbitTime) + " Earth days" + \
           "\n--Day Time: " + str(planet.dayTime) + " Earth days"

    return info

if __name__ == "__main__":

    # play the background sound
    mixer.music.load('background.mp3')
    mixer.music.play(-1)

    img = cv2.imread('solar_system.jpg',1)
    clone = img.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_display)

    # Get the data from the XML
    solarSystem = ET.parse('planet_info.xml')
    celestialBodies = solarSystem.getroot()

    # parse everything from XML into global variables
    for cBodies in celestialBodies:
        planets = cBodies.findall("planet")
        stars = cBodies.findall("star")
        if planets:
            for planet in planets:
                planet_list.append(
                    Planet(planet[0].text, planet[1].text, planet[2].text, planet[3].text, planet[4].text,
                           planet[5].text, planet[6].text, planet[7].text))
        elif stars: #since there is only one star (sun), we just add it into the list of planets
            for star in stars:
                planet_list.append(
                    Planet(star[0].text, star[1].text, star[2].text, star[3].text, star[4].text, star[5].text,
                           star[6].text, star[7].text))
        else:
            print("Nothing")

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF

        # if the 'c' key is pressed, break from the loop
        if key == ord("c"):
            break

    # close all open windows
    cv2.destroyAllWindows()