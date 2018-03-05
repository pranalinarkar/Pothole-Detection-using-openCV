import cv2
import numpy as np
import pygame
import cv2 as cv
import time
import smtplib
from matplotlib import pyplot as plt



im = cv2.imread('3.jpg')
# CODE TO CONVERT TO GRAYSCALE


gray1 = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# save the image
cv2.imwrite('graypothholeresult.jpg', gray1)
plt.subplot(331),plt.imshow(gray1, cmap='gray'),plt.title('GRAY')
plt.xticks([]), plt.yticks([])

#blur
blur = cv2.blur(gray1,(5,5))
#guassian blur 
gblur = cv2.GaussianBlur(gray1,(5,5),0)
#median 
median = cv2.medianBlur(gblur,5)
#erosion
kernel = np.ones((5,5),np.uint8)
kernel1 = np.ones((2,2),np.uint8)
erosion = cv2.erode(median,kernel,iterations = 1)
#erosion followed dilation
dilation = cv2.dilate(erosion,kernel,iterations = 2)
#canny edge detection
edges = cv2.Canny(dilation,9,220)

dilation = cv2.dilate(edges,kernel,iterations = 1)
erosion = cv2.erode(dilation,kernel1,iterations = 4)
dilation = cv2.dilate(erosion,kernel1,iterations = 1)

#contourdetection for sample ellipse
im1 = cv2.imread('index4.jpg')
gray1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
gblur1 = cv2.GaussianBlur(gray1,(5,5),0)
median1 = cv2.medianBlur(gblur1,5)
kernel = np.ones((5,5),np.uint8)
erosion1 = cv2.erode(median1,kernel,iterations = 1)
dilation1 = cv2.dilate(erosion1,kernel,iterations = 2)
edges1 = cv2.Canny(dilation1,9,220)
dilation1 = cv2.dilate(edges1,kernel,iterations = 1)
erosion1 = cv2.erode(dilation1,kernel1,iterations = 4)
dilation1 = cv2.dilate(edges1,kernel1,iterations = 1)
_, contours, hierarchy = cv2.findContours(dilation1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt=contours[0]
#contour detection
_, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_list=[]
if contours is not None:
    contour_list = []
    avg=0
    avg1=0
    i=0
    for c in contours:
        approx = cv2.approxPolyDP(c,0.015*cv2.arcLength(cnt,True),True)
        #print len(approx)
        avg += len(approx)
        avg1+= cv2.contourArea(c)
        i += 1
    avg /= i
    avg1 /= i
    #print avg
    for contour in contours:
        area = cv2.contourArea(contour)
        print area
        #print len(approx)
        ret=cv2.arcLength(contour, True)
        #print cv2.isContourConvex(contour)
        #print ret
        
        if ((len(approx) < avg*2) and (area > 1000) ):
            #print 'working'
            #
            ret1=cv2.matchShapes(contour,cnt,1,0.0)
            #print ret1
            if (ret1 < 0.6):
                contour_list.append(contour)
                print 'working'
                print area
                #print ret1
                ellipse = cv2.fitEllipse(contour)
                if ellipse is not None:
                    imgg=cv2.ellipse(im,ellipse,(0,255,0),2)
                
            
        

    cv2.drawContours(im, contour_list,  -1, (255,0,0), 2)
    
    cv2.imwrite('contourpothholeresult.jpg', im)
    result1 = cv2.imread('contourpothholeresult.jpg')
    

if (contour_list == []):
    print 'No contours'



    
#plotting using matplotlib
plt.subplot(332),plt.imshow(blur,cmap = 'gray'),plt.title('BLURRED')
plt.xticks([]), plt.yticks([])
plt.subplot(333),plt.imshow(gblur,cmap = 'gray'),plt.title('guassianblur')
plt.xticks([]), plt.yticks([])        
plt.subplot(334),plt.imshow(median,cmap = 'gray'),plt.title('Medianblur')
plt.xticks([]), plt.yticks([])
plt.subplot(337),plt.imshow(dilation,cmap = 'gray')
plt.title('dilated Image'), plt.xticks([]), plt.yticks([])

plt.subplot(338),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(335),plt.imshow(erosion,cmap = 'gray'),plt.title('EROSION')
plt.xticks([]), plt.yticks([])
plt.subplot(336),plt.imshow(blur,cmap = 'gray'),plt.title('hough')
plt.xticks([]), plt.yticks([])
plt.subplot(339),plt.imshow(result1),plt.title('Contours')
plt.xticks([]), plt.yticks([])

plt.show()




