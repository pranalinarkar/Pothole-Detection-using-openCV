import cv2
import numpy as np
import pygame
import cv2 as cv
import time
import smtplib
from matplotlib import pyplot as plt



im = cv2.imread('index4.jpg')
# CODE TO CONVERT TO GRAYSCALE


gray1 = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# save the image
cv2.imwrite('graypothholeresult.jpg', gray1)
plt.subplot(331),plt.imshow(gray1, cmap='gray'),plt.title('GRAY')
plt.xticks([]), plt.yticks([])
#CONTOUR DETECTION CODE
#imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#ret,thresh = cv2.threshold(imgray,127,255,0)

#_, contours1, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#_, contours2, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#img1 = im.copy()
#img2 = im.copy()

#out = cv2.drawContours(img1, contours1, -1, (255,0,0), 2)
#out = cv2.drawContours(img2, contours2, -1, (250,250,250),1)
#out = np.hstack([img1, img2])


#img = cv2.imread('index1.jpg',0)
#ret,thresh = cv2.threshold(img,127,255,0)
#_, contours,hierarchy = cv2.findContours(thresh, 1, 2) 
#cnt = contours[0]
#M = cv2.moments(cnt)

#print M
#perimeter = cv2.arcLength(cnt,True)
#print perimeter
#area = cv2.contourArea(cnt)
#print area
#epsilon = 0.1*cv2.arcLength(cnt,True)
#approx = cv2.approxPolyDP(cnt,epsilon,True)
#print epsilon
#print approx
#for c in contours:
    #rect = cv2.boundingRect(c)
    #if rect[2] < 100 or rect[3] < 100: continue
    #print cv2.contourArea(c)
    #x,y,w,h = rect
    #cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),8)
    #cv2.putText(img2,'Moth Detected',(x+w+40,y+h),0,2.0,(0,255,0))

#cv2.waitKey()  
#cv2.destroyAllWindows()


#to check convexity

#blur
blur = cv2.blur(gray1,(5,5))
#guassian blur 
gblur = cv2.GaussianBlur(gray1,(5,5),0)
#median 
median = cv2.medianBlur(gblur,5)


#erosion
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(median,kernel,iterations = 1)
#erosion followed dilation
dilation = cv2.dilate(erosion,kernel,iterations = 2)
#canny edge detection
edges = cv2.Canny(dilation,9,220)

#Segmentation
ret, thresh = cv2.threshold(edges,0,255,cv2.THRESH_OTSU)





#contourdetection
im1 = cv2.imread('ellipse.jpg')
gray1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
gblur1 = cv2.GaussianBlur(gray1,(5,5),0)
median1 = cv2.medianBlur(gblur1,5)
kernel = np.ones((5,5),np.uint8)
erosion1 = cv2.erode(median1,kernel,iterations = 1)
dilation1 = cv2.dilate(erosion1,kernel,iterations = 2)
edges1 = cv2.Canny(dilation1,9,220)
ret, thresh1 = cv2.threshold(edges1,0,255,cv2.THRESH_OTSU)

_, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt=contours[0]
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contour_list = []
avg=0
avg1=0
i=0
for c in contours:
    avg += len(c)
    avg1+= cv2.contourArea(c)
    i += 1
avg /= len(contours)
avg1 /= i
print avg1
for contour in contours:
    approx = cv2.approxPolyDP(contour,0.015*cv2.arcLength(contour,True),True)
    area = cv2.contourArea(contour)
    print cv2.isContourConvex(contour)
    print ret
    
    
    if ((len(contour) > avg/2) and (area > avg1/20)):
        print 'working'
        contour_list.append(contour)
        ellipse = cv2.fitEllipse(contour)
        imgg=cv2.ellipse(im,ellipse,(0,255,0),2)
        

cv2.drawContours(im, contour_list,  -1, (255,0,0), 2)
for c in contour_list:
    ret=cv2.matchShapes(c,cnt,1,0.0)
    print ret
cv2.imwrite('contourpothholeresult.jpg', im)
result1 = cv2.imread('contourpothholeresult.jpg')
gray2 = cv2.cvtColor(result1,cv2.COLOR_BGR2GRAY)
if contour_list is not None:
    print 'No contours'




#houghtransform
circles = cv2.HoughCircles(gray2, cv2.HOUGH_GRADIENT, 1,minDist=10,
param1=100, param2=40, minRadius=0, maxRadius=150)
img = cv2.imread('contourpothholeresult.jpg')
if circles is not None:
    for i in circles[0,:]:
# draw the circles
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        x=i[0]
        y=i[1]
        Red=img[i[1],i[0],[2]]
        Green=img[i[1],i[0],[1]]
        Blue=img[i[1],i[0],[0]]
        cv2.putText(img,str(int(Red))+","+str(int(Green))+","+  str(int(Blue)), (int(x-40),y), cv2.FONT_HERSHEY_SIMPLEX, 5, 0)
        cv2.putText(img,"Found Circle", (int(x-50),int(y+30)),cv2.FONT_HERSHEY_SIMPLEX, 5, 0)
        cv2.imwrite('detectedcircles.jpg', img)
        
else:
   print 'There is no circles to be detected!'
    
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
plt.subplot(336),plt.imshow(imgg,cmap = 'gray'),plt.title('hough')
plt.xticks([]), plt.yticks([])
plt.subplot(339),plt.imshow(result1),plt.title('Contours')
plt.xticks([]), plt.yticks([])

plt.show()




