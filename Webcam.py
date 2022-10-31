# Chess piece detection software for CheckM8
# University of Victoria
# Written by Sahel

# The following is the chess piece detection software that is to be used by the overall chess engine.
# As the name suggests, it is used to detect the nature and color of the chess piece present on the chess board being analysed.
# The program makes use of Convolutional Neural Networks (CNN) to split the input into mulitple layers, analyse these layers individually,
# map every layer to every other layer, and finally combine it all together to achieve a single output. 

# The technology used for the detection portion is OpenCV and the dataset has been trained using the YoloV3 model.
# Numpy has been used to carry out the required calculations for accurate detection of pieces.  

# Fun fact about chess: The number of unique chess games possible is 10^120. As a comparison, the number of electrons in the universe is 10^79.

#-------------------------------------------------------------------------------------------------------------------------------------------------#

# Import the OpenCV and numpy libraries.
import cv2
import numpy as np

# Define the path of the weights, configurations, and names folders here.
# This project used openCV version 4, therefore if the files are directly passed to the readNet function, a parsing error is thrown.
# Use r before the file path so that compiler doesn't throw an error.
modelConfiguration = r"C:\Users\Owner\Documents\B.Eng. Computer Engineering\Third Year\3A\ECE 356\Project\chess_yolov3.cfg.txt"
modelWeights = r"C:\Users\Owner\Documents\B.Eng. Computer Engineering\Third Year\3A\ECE 356\Webcam project\yolov3.weights"
modelNames = r"C:\Users\Owner\Documents\B.Eng. Computer Engineering\Third Year\3A\ECE 356\Webcam project\coco_webcam.txt"

# Reading the trained dataset.
net = cv2.dnn.readNet(modelWeights, modelConfiguration)

# Array to store the names of the various pieces that are to be detected.
# PW = Pawn White
# PB = Pawn Black
# RW = Rook White
# RB = Rook Black
# NW = Knight White
# NB = Knight Black
# BW = Bishop White
# BB = Bishop Black
# QW = Queen White
# QB = Queen Black
# KW = King White
# KB = King Black
classes = []
with open(modelNames, 'r') as f:
    classes = f.read().splitlines()

# Uncomment test_video and cap to enable webcam detection.
#test_video = r"C:\Users\Owner\Documents\B.Eng. Computer Engineering\Third Year\3A\ECE 356\New folder\video.mp4"
cap = cv2.VideoCapture(0)

# Comment out test_pic and img to disable digital image detection.
# Change the name of the file (eg. ....\ECE 356\New folder\*file name*) every time a new image is to be analysed.
#test_pic = r"C:\Users\Owner\Documents\B.Eng. Computer Engineering\Third Year\3A\ECE 356\New folder\test.png"
#img = cv2.imread(test_pic)

# This while loop is required only when webcam is on so that the detection is continuous.
while True:
    _, img = cap.read()
    height, width, _ = img.shape

    # Change the image specifications to fit the configurations required by the yolov3 model.
    # OpenCV provides a function to do this. Essentially, this is simply reducing the pixel density of the original image
    # and converting it from BGR to RGB order of colours.
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB = True, crop = False)
    net.setInput(blob)

    # This simply obtains the names of the output layers and passes them as the required output.
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    # Arrays to hold the bounding boxes, confidence of detection, and the predicted class by the algorithm.
    boxes = []
    confidences = []
    class_ids = []

    # First for loop extracts the obtained ouput from the output layers.
    # Second for loop extracts the information that each output holds.
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]                                              # Array to hold the predictions made.
            class_id = np.argmax(scores)                                        # Class with the highest value of scores.
            confidence = scores[class_id]                                       # Isolate the confidence the predicted class.
            
            # Check if the confidence of prediction is more than 50%.
            if confidence > 0.5:

                # Convert the output image back to the scaling of the original input image.
                # Essentially, this portion is trying to cancel out the modifications made by the "blob" variable above.
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
            
                # Passing the information obtained back to the original arrays.
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    # NMSBoxes function is used to eliminate multiple detections of the same object and avoid redundancy.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Font of output class and color of boxes.
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size = (len(boxes), 3))

    #if statement added to avoid tuple attribute error
    if len(indexes) > 0:
        # Combine all the extracted information from each detected object back together and display it as the output.
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)                                                        # Use cv2.waitKey(1) for video and webcam files.
    if key == 27:                                                              # If Escape key has been pressed, exit the while loop.
        break

cap.release()                                                                  # Stop processing the video.
cv2.destroyAllWindows()                                                         # Kill all running processes.