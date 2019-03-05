import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
 
# TODO: add exceptions for when no contours are found
# TODO: rewrite beginning for raspberry pi
# TODO: add auto exposure and gain functions
 
CIRCLE_COLORS = {
    # TODO: better color ranges
    'red1': {'min': (0, 30, 30), 'max': (15, 255, 255), 'draw': (0, 0, 255)},
    'red2': {'min': (245, 30, 30), 'max': (255, 255, 255), 'draw': (0, 0, 255)},
    'green': {'min': (35, 30, 30), 'max': (70, 255, 255), 'draw':(0, 255, 0)},
    'blue':  {'min': (80, 30, 30), 'max': (140, 255, 255), 'draw': (255, 0, 0)},
}
 
 
def start_video(cap):
    return cap.read()
 
 
def start_image(cap):
    return False, cv2.imread(r"C:\Users\Z1\Desktop\eurobot2019.png")
 
 
def check_contour_area(contour, min_area=0, max_area=200000):
    return max_area > cv2.contourArea(contour) > min_area
 
 
def is_perspective_circle(contour, max_ratio=5, min_ratio=0.5):
    _, _, w, h = cv2.boundingRect(contour)
    return max_ratio * w >= h >= min_ratio * w
 
 
def filter_contours(color_mask):
    filtered_contours = []
    _, contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if check_contour_area(contour) and is_perspective_circle(contour):
            filtered_contours.append(contour)
    filtered_contours = sorted(filtered_contours, key=cv2.contourArea)
    return filtered_contours[0], True
 
 
def find_center(color_mask):
    contour, found = filter_contours(color_mask)
    x, y, w, h = cv2.boundingRect(contour)
    return int(x+w/2), int(y+h/2)
 
 
def init_picamera():
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))
    time.sleep(0.1)
    return camera, rawCapture
 
 
camera, rawCapture = init_picamera()
 
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
 
    # read & split images
    image = frame.array
 
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
 
    blue_mask = cv2.inRange(image_hsv, CIRCLE_COLORS["blue"]["min"], CIRCLE_COLORS["blue"]["max"])
    green_mask = cv2.inRange(image_hsv, CIRCLE_COLORS["green"]["min"], CIRCLE_COLORS["green"]["max"])
    red1_mask = cv2.inRange(image_hsv, CIRCLE_COLORS["red1"]["min"], CIRCLE_COLORS["red1"]["max"])
    red2_mask = cv2.inRange(image_hsv, CIRCLE_COLORS["red2"]["min"], CIRCLE_COLORS["red2"]["max"])
    red_mask = cv2.bitwise_or(red1_mask, red2_mask)
 
    # remove noise
    kernel = np.ones((7, 7), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
 
    # blur
    kernel = np.ones((3, 3), np.uint8)
    MORPH_ITERATIONS = 2
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_DILATE, kernel, iterations=MORPH_ITERATIONS)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_DILATE, kernel, iterations=MORPH_ITERATIONS)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel, iterations=MORPH_ITERATIONS)
 
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_ERODE, kernel, iterations=MORPH_ITERATIONS)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_ERODE, kernel, iterations=MORPH_ITERATIONS)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_ERODE, kernel, iterations=MORPH_ITERATIONS)
 
    # crop original
    blue_image = cv2.bitwise_and(image, image, mask=blue_mask)
    green_image = cv2.bitwise_and(image, image, mask=green_mask)
    red_image = cv2.bitwise_and(image, image, mask=red_mask)
 
    try:
        # find centers of pucks
        blue_center = find_center(blue_mask)
        green_center = find_center(green_mask)
        red_center = find_center(red_mask)
 
        # paint centers to original
        cv2.circle(image, blue_center, 5, CIRCLE_COLORS["blue"]["draw"], -1)
        cv2.circle(image, green_center, 5, CIRCLE_COLORS["green"]["draw"], -1)
        cv2.circle(image, red_center, 5, CIRCLE_COLORS["red1"]["draw"], -1)
 
        cv2.circle(blue_image, blue_center, 5, CIRCLE_COLORS["blue"]["draw"], 2)
        cv2.circle(green_image, green_center, 5, CIRCLE_COLORS["green"]["draw"], 2)
        cv2.circle(red_image, red_center, 5, CIRCLE_COLORS["red1"]["draw"], 2)
 
    except:
        pass
 
    # show images
    """cv2.imshow("orig",image)
    cv2.imshow("blue", blue_image)
    cv2.imshow("green", green_image)
    cv2.imshow("red", red_image)
"""
    print("test\n")
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord("q"):
        break
 
cv2.waitKey(1)
cv2.destroyAllWindows()
