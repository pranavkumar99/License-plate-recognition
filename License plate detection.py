#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import imutils
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\prana\anaconda3\Lib\site-packages\pytesseract\__pycache__"


# In[ ]:


image = cv2.imread('car image.jpg')


# In[ ]:


# Resize the image - change width to 500
image = imutils.resize(image, width=500)


# In[ ]:


cv2.imshow("Original Image", image)
cv2.waitKey(0)


# In[ ]:


# RGB to Gray scale conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("1 - Grayscale Conversion", gray)
cv2.waitKey(0)


# In[ ]:


# Noise removal with iterative bilateral filter(removes noise while preserving edges)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("2 - Bilateral Filter", gray)
cv2.waitKey(0)


# In[ ]:


# Find Edges of the grayscale image
edged = cv2.Canny(gray, 170, 200)
cv2.imshow("3 - Canny Edges", edged)
cv2.waitKey(0)


# In[ ]:


# Find contours based on Edges
cnts, new  = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


# In[ ]:


# Create copy of original image to draw all contours
img1 = image.copy()
cv2.drawContours(img1, cnts, -1, (0,255,0), 3)
cv2.imshow("4- All Contours", img1)
cv2.waitKey(0)


# In[ ]:


#sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
NumberPlateCnt = None #we currently have no Number plate contour


# In[ ]:


# Top 30 Contours
img2 = image.copy()
cv2.drawContours(img2, cnts, -1, (0,255,0), 3)
cv2.imshow("5- Top 30 Contours", img2)
cv2.waitKey(0)


# In[ ]:


# loop over our contours to find the best possible approximate contour of number plate
count = 0
idx =7
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # print ("approx = ",approx)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx #This is our approx Number Plate Contour

            # Crop those contours and store it in Cropped Images folder
            x, y, w, h = cv2.boundingRect(c) #This will find out co-ord for plate
            new_img = gray[y:y + h, x:x + w] #Create new image
            cv2.imwrite('license plate/Cropped Images-Text/' + str(idx) + '.png', new_img) #Store new image
            idx+=1

            break


# In[ ]:


# Drawing the selected contour on the original image
#print(NumberPlateCnt)
cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
cv2.imshow("Final Image With Number Plate Detected", image)
cv2.waitKey(0)


# In[ ]:


Cropped_img_loc = 'Cropped Images-Text/7.png'
cv2.imshow("Cropped Image ", cv2.imread(Cropped_img_loc))


# In[ ]:


# Use tesseract to covert image into string
text = pytesseract.image_to_string(Cropped_img_loc, lang='eng')
print("Number is :", text)

cv2.waitKey(0) 
#Wait for user input before closing the images displayed


# In[ ]:




