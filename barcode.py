'''
    this file containes functions for mac address recognition in images using seperatly OCR and zbar library. 
'''

'''
    dependency : 
        1-pytesseract
        2-pyzbar
        3-openCV
        4-numpy
'''
import pytesseract
from pyzbar import pyzbar
import cv2 
import numpy as np 
import os 
from os.path import join 

def crop_minAreaRect(image, rect):
    '''
    helper function to crop and rotate rectangles that may contain barcodes 

    Parameters
    ----------
    image : numpy array
        original image.
    rect : rectangle
        openCV countour rectangle.

    Returns
    -------
    image : numpy array
        cropped image.

    '''

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    #crop image inside bounding box
    scale = 1.2  # cropping margin, 1 == no margin
    W = rect[1][0]
    H = rect[1][1]
    
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    
    angle = rect[2]
    rotated = False
    if angle < -45:
        angle += 90
        rotated = True
    
    center = (int((x1+x2)/2), int((y1+y2)/2))
    size = (int(scale*(x2-x1)), int(scale*(y2-y1)))
    
    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
    
    cropped = cv2.getRectSubPix(image, size, center)
    cropped = cv2.warpAffine(cropped, M, size)
    
    croppedW = W if not rotated else H
    croppedH = H if not rotated else W
    
    image = cv2.getRectSubPix( cropped, (int(croppedW*scale), int(croppedH*scale)), (size[0]/2, size[1]/2))
   # image =cv2.rotate( image,rotateCode = 0)
    return image

def preprocessing(image,show=0) :
    '''
    wrapper function for preprocessing images and produce canddidate partial images from the original 

    Parameters
    ----------
    image : numpy array
        image to process.
    show : bool, optional
        1 if you want to show the partial output of preprocessing stage. The default is 0.

    Returns
    -------
    candidate_images : list 
        list of partial images that may containes barcodes.
    image : TYPEnumpy array 
        modified image where we add detected boxes that may contain barcode.

    '''
    
    # # to gray scale 
    # image =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    # #thresholding 
    # image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY )[1]
    #resize image
    image = cv2.resize(image,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)
    
    #convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #calculate x & y gradient
    gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
    
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    if show == 1:
     	cv2.imshow("gradient-sub",cv2.resize(gradient,None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC))
    
    # blur the image
    blurred = cv2.blur(gradient, (3, 3))
    
    # threshold the image
    (_, thresh) = cv2.threshold(blurred, 250, 255, cv2.THRESH_BINARY)
    
    if show == 1:
     	cv2.imshow("threshed",cv2.resize(thresh,None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC))
    
    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    if show == 1:
     	cv2.imshow("morphology",cv2.resize(closed,None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC))
    
    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations = 10)
    closed = cv2.dilate(closed, None, iterations = 10)
    
    if show == 1:
     	cv2.imshow("erode/dilate",cv2.resize(closed,None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC))
    
    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    cnts,hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    candidats  = sorted(cnts, key = cv2.contourArea, reverse = True)[0:2]
    candidate_images = [] 
   # c1 = sorted(cnts, key = cv2.contourArea, reverse = True)[1]
    for c in candidats : 
        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
     
        # draw a bounding box arounded the detected barcode and display the
        # image

       # tmp_image = image[y_axis[min_y]:y_axis[max_y],x_axis[min_x] : x_axis[max_x],:]
        
        tmp_image = crop_minAreaRect(image,rect)
        tmp_image= cv2.cvtColor(tmp_image, cv2.COLOR_BGR2GRAY)
        #_,tmp_image=cv2.threshold(tmp_image, 200, 255, cv2.THRESH_BINARY)
     
       # tmp_image =cv2.bitwise_not(tmp_image)
        candidate_images.append(tmp_image)
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
       # cv2.imshow(str(box[1]),tmp_image)
        #image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

    return candidate_images ,image

def process_image_zbar (candidate_images) : 
    '''
    extract barcode using zbar library 

    Parameters
    ----------
    candidate_images : list
         list of candidate images that contains the mac address.

    Returns
    -------
    ret_text : string
        the returned mac address or NA if no mac address was detected or if two macs are detected.

    '''
 
    ret_text = "NA"
    for image in candidate_images : 
        barcodes = pyzbar.decode(image)
        text="NA"
        for barcode in barcodes:
    		# extract the bounding box location of the barcode and draw
    		# the bounding box surrounding the barcode on the image
            (x, y, w, h) = barcode.rect
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    		# the barcode data is a bytes object so if we want to draw it
    		# on our output image we need to convert it to a string first
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type
    		# draw the barcode data and barcode type on the image
            text = barcodeData
            cv2.putText(image, text, (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if not text == "NA" : 
                if not ret_text == "NA" : 
                    print('two mac addresses detected  %s %s'%(text,ret_text))
                    return "two macs" 
                ret_text =text 
            

    return ret_text

def process_image_OCR (candidate_images) : 
    '''
    extracting mac address using OCR     

    Parameters
    ----------
    candidate_images : list
        list of candidate images that contains the mac address.

    Returns
    -------
    ret_text : string
        the returned mac address or NA if no mac address was detected or if two macs are detected.

    '''
    
    
    ret_text="NA"
    for image in candidate_images : 
        text = pytesseract.image_to_string(image)
        subst_idx = text.find('MAC')
        if not subst_idx == -1 : 
            if not ret_text == "NA" : 
                    print('two mac addresses detected  %s %s'%(text,ret_text))
                    return "two macs" 
            ret_text =  text[subst_idx+4 :subst_idx+16 ]
      
    return ret_text

def get_mac_address (image_path ,show_result = True ) :
    '''
    wrapper function for extracting mac address.    

    Parameters
    ----------
    image_path : string 
        image path.
    show_result : bool, optional
        True if you want to show resutls. The default is True.

    Returns
    -------
    result_OCR : string
        OCR result.
    result_barcode : string
        zbar barcode result.

    '''

    image = cv2.imread(image_path)
    candidate_images ,orig_image= preprocessing(image)
    result_OCR = process_image_OCR(candidate_images)
    result_barcode = process_image_zbar ( candidate_images)
    print('result OCR ' , result_OCR)
    print('result barcode scan ', result_barcode) 
    if result_OCR =="two macs" or result_barcode == "two macs" : 
        raise  ValueError('Больше одного штрих-кода в кадре')
    if show_result : 
      
        cv2.imshow('image',orig_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return result_OCR, result_barcode

def process_stream (streamURL,show_result = True) :
    '''
    detect adn decode barcode in a stream, press q to exit the function. 

    Parameters
    ----------
    streamURL : string
        stream url.
    show_result : bool, optional
        True if you want to show the resutls. The default is True.

    Returns
    -------
    None.

    '''
    print('processing URL : %s'%(streamURL))
    vidcap = cv2.VideoCapture(streamURL)
    success,frame  = vidcap.read() 
    while success : 
        if cv2.waitKey(1) & 0xFF == ord("q") : 
            print('breaking ...')
            break 
        candidate_images ,orig_image= preprocessing(frame)
        result_OCR = process_image_OCR(candidate_images)
        result_barcode = process_image_zbar ( candidate_images)
        print('result OCR ' , result_OCR)
        print('result barcode scan ', result_barcode) 
        if show_result : 
             cv2.imshow('image',orig_image)
        success,frame = vidcap.read()
    vidcap.release()
    cv2.destroyAllWindows()
        
def test_folder ( data_dir = "/home/ki/projects/barcode/Приставки/images",show_result=False) : 
    '''
    perform test on a set of images in a folder

    Parameters
    ----------
    data_dir : string, optional
        path to the folder containing images. The default is "/home/ki/projects/barcode/Приставки/images".

    Returns
    -------
    None.

    '''
    images_paths = [join(data_dir,f) for f in os.listdir(data_dir) ]
    for image_path in images_paths[:20] : 
        print('processing image %s ... '%(image_path))
        get_mac_address(image_path,show_result=show_result)
        print()
    
if __name__=='__main__' : 
    #image_path = "/home/ki/projects/barcode/Приставки/images/WhatsApp Image 2020-08-03 at 11.43.41.jpeg"
    #get_mac_address(image_path,show_result=True)
    #test_folder()   
    process_stream(0)
    '''
        to do : 
            -check for multipule macs(yes)
            -check on flliped and noisy images (yes)
            -add video capability  (yes)
            -raise exceptions (yes)          
    '''