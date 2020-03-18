from imutils.object_detection import non_max_suppression
import pytesseract
import sys
import math
import cv2
import numpy as np
import argparse
import cv2 as cv


#   ***************************** % *************************  %  ***************************************

circleCord = []
stateNameCord = []
LineCord = []
inputCord = []
NoOfStates = 0
NoOfInput = 0

table = []


def Sort_Tuple(tup):
    return(sorted(tup, key=lambda x: x[0]))



# DETECTING LINES -------------------------------------------------------------------------------------------------------------------------------


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])


# argv=sys.argv[1:]
default_file = args["image"]
filename = default_file
# Loads an image
src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
# Check if image is loaded fine
if src is None:
    print('Error opening image!')
    print(
        'Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')


dst = cv.Canny(src, 50, 200, None, 3)
# circlesCannyImage = dst
# cv.imshow("Canny Edge Detector", dst)

# Copy edges to the images that will display the results in BGR
cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
#cv.imshow("CDST", cdst)
cdstP = np.copy(cdst)
#cv.imshow("CDSTP", cdstP)

lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)


if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

#cv.imshow("linesCDST", cdst)
linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

textBoxes = []
#textBoxes.append([800, 247, 890, 360])
lineTextBoxes = []
# cdstP=image
print("\n\n\n\d")
print(linesP)
print("\n\n\n\d")
if linesP is not None:
    print(len(linesP))
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
        startx = l[0]
        starty = l[1]
        endx = l[2]
        endy = l[3]
        if(starty - endy > 0):  # angle greater than 0 degree 1st quadrant
            print('Hey')
            li = [startx, endy-3, endx, starty+2]
            print(li)
            lineTextBoxes.append(li)
        else:
            if startx - endx > 0:
                print('hi')
                li = [endx, starty, startx, endy]
                print(li)
                lineTextBoxes.append(li)
            else:
                lineTextBoxes.append([startx+7, starty-60, endx+4, endy-8])
        #cv.imshow("linesCDSTP" + str(i), cdstP)

print(lineTextBoxes)
# cv.imshow("Source", src)
# cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
# cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

# cv.waitKey()

# copy output of line detection into final output variable "output"


output = cdstP.copy()
cv2.imwrite('lines.png', output)


# detect circles in the image
#   ***************************** % *************************  %  ***************************************


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
dst = cv.Canny(image, 50, 200, None, 3)
# cv2.imshow("Canny",dst)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.21, 7)
# returns x,y,radius most probably
# ensure at least some circles were found


print(circles)
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        circleCord.append((x, y, r))
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        #cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        pad = r//2
        li = []
        li.append(x-r+pad)
        li.append(y-r+pad)
        li.append(x+r-pad)
        li.append(y+r-pad)
        textBoxes.append(li)

        #cv.imshow("linesCDSTP" + str(x),output)
        # show the output image
        #cv2.imshow("output", np.vstack([image, output]))
    cv2.imshow("CirclesOp", output)
    cv2.waitKey(0)

cv2.imwrite('CircleAndLine.png', output)

print(circleCord)
circleCord = Sort_Tuple(circleCord)
print(circleCord)
outputCircleLine = output


#   ***************************** % *************************  %  ***************************************
# Detecting Text


def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < args["min_confidence"]:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                help="path to input image")
ap.add_argument("-east", "--east", type=str,
                help="path to input EAST text detector", default="frozen_east_text_detection.pb")
ap.add_argument("-c", "--min-confidence", type=float, default=0.1,
                help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
                help="nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height", type=int, default=320,
                help="nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding", type=float, default=0.0,
                help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())


# load the input image and grab the image dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(origH, origW) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (args["width"], args["height"])
rW = origW / float(newW)
rH = origH / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                             (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

# decode the predictions, then  apply non-maxima suppression to
# suppress weak, overlapping bounding boxes
(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)

# initialize the list of results
results = []
print(boxes)
print(type(boxes))
print(textBoxes)
print(type(textBoxes))


# Create a bounding box for text inside circles and try extracting the text from it--------------------------------------------------------

# loop over the bounding boxes
# for (startX, startY, endX, endY) in boxes:
for (startX, startY, endX, endY) in textBoxes:
    # scale the bounding box coordinates based on the respective
    # ratios
    # startX = int(startX * rW)
    # startY = int(startY * rH)
    # endX = int(endX * rW)
    # endY = int(endY * rH)
    print([startX, startY, endX, endY])
    # print(results)

    # in order to obtain a better OCR of the text we can potentially
    # apply a bit of padding surrounding the bounding box -- here we
    # are computing the deltas in both the x and y directions
    # dX = int((endX - startX) * args["padding"])
    # dY = int((endY - startY) * args["padding"])

    # apply padding to each side of the bounding box, respectively
    # startX = max(0, startX - dX)
    # startY = max(0, startY - dY)
    # endX = min(origW, endX + (dX * 2))
    # endY = min(origH, endY + (dY * 2))

    # extract the actual padded ROI

    roi = orig[startY:endY, startX:endX]

    # in order to apply Tesseract v4 to OCR text we must supply
    # (1) a language, (2) an OEM flag of 4, indicating that the we
    # wish to use the LSTM neural net model for OCR, and finally
    # (3) an OEM value, in this case, 7 which implies that we are
    # treating the ROI as a single line of text
    config = ("-l eng --oem 1 --psm 10")
    text = pytesseract.image_to_string(roi, config=config)

    # add the bounding box coordinates and OCR'd text to the list
    # of results
    results.append(((startX, startY, endX, endY), text))

# print(orig.shape[:2])
# print(lineTextBoxes)
print("\n\n\n\nDrawing Line Text Boxes . . .")

# Create a bounding box for text above lines and try extracting the text from it--------------------------------------------------------



for (startX, startY, endX, endY) in lineTextBoxes:

    print([startX, startY, endX, endY])
    roi = orig[startY:endY, startX:endX]
    config = ("-l eng --oem 1 --psm 10")
    text = pytesseract.image_to_string(roi, config=config)
    results.append(((startX, startY, endX, endY), text))








# sort the results bounding box coordinates from top to bottom
results = sorted(results, key=lambda r: r[0][1])
output = outputCircleLine.copy()
# loop over the results
for ((startX, startY, endX, endY), text) in results:
    # display the text OCR'd by Tesseract
    print("OCR TEXT")
    print("========")
    print("{}\n".format(text))
    

    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV, then draw the text and a bounding box surrounding
    # the text region of the input image
    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()

    cv2.rectangle(output, (startX, startY), (endX, endY),
                  (255, 51, 51), 2)
    textCoordinates = [(startX+endX)/2, (startY+endY)/2]
    print(textCoordinates[0], end=" ")
    print(textCoordinates[1])
    cv2.putText(output, text, (startX, startY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    stateNameCord.append([textCoordinates[0],textCoordinates[1],text])
    # show the output image


#creating Table - - - -- - - -- - - - - - - -   -----  ---------------------------- - - - -- ----- - -- - -   -- -- ------------- --- ---  -- - 

FinalState =[]
print(stateNameCord)
stateNameCord = Sort_Tuple(stateNameCord)
print(stateNameCord)
print(circleCord)
stateName = []
stateNameIndx = 0 
NoOfInput=0
for textcord in stateNameCord:
    # print( textcord[0])
    # print(circleCord[i][0])
    # print(circleCord[i][2])
    if textcord[0] > circleCord[NoOfInput][0]+circleCord[NoOfInput][2]:
        #then it might be input text
        inputCord.append(textcord)
        NoOfInput+=1
    elif(textcord[0] >= circleCord[NoOfInput][0] ):
        print( textcord[2])
        if len(stateName)!=0:
            print(stateName)
        if(len(stateName)!=0 and textcord[2]==stateName[-1][2] ):
            #double circle . May be final state
            print(stateName[-1][2])
            stateName[-1][2]="*" + str(stateName[-1][2])
            continue
        if len(stateName)!=0 and  (("*" + textcord[2])== str(stateName[-1][2])):
            stateName.append(textcord)
            continue
        stateName.append(textcord)
print(stateName)
print(inputCord)
print(NoOfInput)
x=[0]
for i in inputCord:
    x.append(i[2])
table.append(x)
for circle in stateName:
    x=[0]*(NoOfInput+1)
    x[0]=circle[2]
    table.append(x)

print("*********************************************")
print("                 INPUTS                          ")
print("\n\n")
for i in table:
    if i==table[len(table)//2]:
        print("STATES       ",end="")
    else:   
        print("             ",end="")
    print(*i)
print("\n\n")
print("*********************************************")
#table.append()
print(table)
cv2.imshow("Text Detection", output)
cv2.waitKey(0)
cv2.imwrite('result.png', output)
