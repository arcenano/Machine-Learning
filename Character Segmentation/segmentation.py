import cv2
import numpy as np

def projection(ogImg,img,xc,yc,wc,hc):
    img = img[yc:yc+hc, xc:xc+wc]

    height, width = img.shape[:2]
    
    #resized = cv2.resize(img, (2*width,2*height), interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, thresh) = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    
    # Make words grow into blocks
    # thresh = cv2.erode(thresh, None, iterations = 7)
    
    height, width = thresh.shape[:2]
    z = [0]*height
    v = [0]*width
    hfg = [[0 for col in range(2)] for row in range(height)]
    lfg = [[0 for col in range(2)] for row in range(width)]
    box = [0,0,0,0]
     
    #Horizontal projection
    a = 0
    emptyImage1 = np.zeros((height, width, 3), np.uint8) 
    for y in range(0, height):
      for x in range(0, width):
        cp = thresh[y,x]
        if cp == 0:
          a = a + 1
        else :
          continue
      z[y] = a
      
      a = 0
    #Select the line split point according to the horizontal projection value
    inline = 1
    start = 0
    j = 0
    threshold = 1 # BLACK PIXELS THRESHOLD
    
    for i in range(0,height):
      if inline == 1 and z [i] >= threshold: # BLACK PIXELS THRESHOLD
        start = i # record the starting line split point
        inline = 0
      elif (i - start > 3) and z [i] < threshold and inline == 0: # BLACK PIXELS THRESHOLD
        inline = 1
        hfg [j] [0] = start - 2 # save row split position
        hfg[j][1] = i + 2
        j = j + 1
     
    # Each line is projected and segmented vertically
    a = 0
    c=0

    for p in range(0, j):
      for x in range(0, width):
        for y in range(hfg[p][0], hfg[p][1]-1):
          cp1 = thresh[y,x]
          if cp1 == 0:
            a = a + 1
          else :
            continue
        v[x] = a # saves the pixel values of each column
        a = 0
      #Vertical split point
      incol = 1
      start1 = 0
      j1 = 0
      z1 = hfg[p][0]
      z2 = hfg[p][1]
      threshold = 1 # BLACK PIXELS THRESHOLD
      for i1 in range(0,width):
        if incol == 1 and v [i1] >= threshold: 
          start1 = i1 # record the starting column split point
          incol = 0
        elif (i1 - start1 > 3) and v [i1] < threshold and incol == 0: 
          incol = 1
          lfg [j1] [0] = start1 - 2 # save column split position
          lfg[j1][1] = i1 + 2
          l1 = start1 - 2
          l2 = i1 + 2
          j1 = j1 + 1
          #print("sp",xc,yc,wc,hc)
          c+=1


    """
        
    # === SKELETON ===
    
    # Original Skel doesn't use the gray scale
    # Step 1: Create an empty skeleton
    ret,img = cv2.threshold(thresh, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))

    # Repeat steps 2-4
    while True:
        #Step 2: Open the image
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv2.subtract(img, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(img)==0:
            break

    img = skel

    # === SKELETON ENDS ===
    """
    
    img = thresh

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    #img = cv2.dilate(img, rect_kernel, iterations = 1)

    #(_, thresh2) = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    

    """
    # CONTOUR AREA CONDITIONAL
    flag = 0
    
    if(len(contours)>c):
        for cnt in contours:
            if cv2.contourArea(cnt)<1:
                flag = 1
    """

    if(len(contours)>c):
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            # Draw Rectangle
            cv2.rectangle(ogImg,(x+xc,y+yc),(x+w+xc,y+h+yc),(0,255,0),2)

    else:
        for p in range(0, j):
            incol = 1
            start1 = 0
            j1 = 0
            z1 = hfg[p][0]
            z2 = hfg[p][1]
            threshold = 1 # BLACK PIXELS THRESHOLD
            c=0
            for i1 in range(0,width):
                if incol == 1 and v [i1] >= threshold: 
                  start1 = i1 # record the starting column split point
                  incol = 0
                elif (i1 - start1 > 3) and v [i1] < threshold and incol == 0: 
                  incol = 1
                  lfg [j1] [0] = start1 - 2 # save column split position
                  lfg[j1][1] = i1 + 2
                  l1 = start1 - 2
                  l2 = i1 + 2
                  j1 = j1 + 1
                  cv2.rectangle(ogImg, (l1+xc, z1+yc), (l2+xc, z2+yc), (255,0,0), 2)


# =====  MAIN =====

im = cv2.imread('./files/emb.jpg')

# Convert to grey scale
img = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

# Specify structure shape and kernel size.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Outline with empty inside (Dilate then Erode)
#img = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,rect_kernel)

# Binarize with OTSU's method
ret,img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# Denoising
#img = cv2.fastNlMeansDenoising(img,img , 10.0, 7, 18)

# Blur the Image
# img = cv2.GaussianBlur(img,(2,2),0)
# img = cv2.blur(img,(2,2));

# Simple Skeleton Method
#erode_kernel= cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))
#img = cv2.erode(img,erode_kernel,iterations = 1)

# Appplying dilation on the threshold image
img = cv2.dilate(img, rect_kernel, iterations = 1)

# Run the Canny Algorithm
#canny = cv2.Canny(thresh, 100, 200)

# Find contours
contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

im2 = im.copy()
im3 = im.copy()

# Loop through contours adding their boundingRects to the image
for cnt in contours:
        #if cv2.contourArea(cnt)>5:
              # Find rectangle
              
              x,y,w,h = cv2.boundingRect(cnt)

              # Draw Rectangle
              cv2.rectangle(im2,(x,y),(x+w,y+h),(0,255,0),2)

              
              projection(im3,im2, x,y,w,h)

              """
              # === DOUBLE CONTOUR DETECTION ===

              # Create sub-image from rectagle
              #ROI = im2[y:y+h, x:x+w]
            
              # Find Contours in Sub Image
              subcontours, subhierarchy = cv2.findContours(ROI,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

              # Loop through subcontours drawing their rectangles
              for subcnt in subcontours:
                  #if cv2.contourArea(cnt)>5:
                  sx,sy,sw,sh = cv2.boundingRect(subcnt)
                  cv2.rectangle(im2,(x+sx,y+sy),(x+sx+sw,y+sy+sh),(0,255,0),2)
                  """

cv2.imshow('thresh', img)
cv2.imshow('words', im2)
cv2.imshow('final', im3)


cv2.waitKey()
