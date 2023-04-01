import numpy as np
import imutils
import cv2

def to_gray_and_edge_enh(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # equalize lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # edge enhancement
    edge_enh = cv2.Laplacian(gray, ddepth = cv2.CV_8U, ksize = 3, scale = 1, delta = 0)
    return edge_enh
    

# bilateral blur, which keeps edges
def blur_and_thresholding(edge_enh):
    blurred = cv2.bilateralFilter(edge_enh, 13, 50, 50)
    # use simple thresholding. adaptive thresholding might be more robust
    # thresh = cv2.adaptiveThreshold(blurred,55,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    #  or
    (_, thresh) = cv2.threshold(blurred, 75, 255, cv2.THRESH_BINARY)
    return thresh



def find_countur_and_draw_box(thresh):
    # do some morphology to isolate just the barcode blob
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)

    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    return box


image = cv2.imread("./barcode_01.jpg")
edge_enh = to_gray_and_edge_enh(image=image)
thresh = blur_and_thresholding(edge_enh=edge_enh)
box = find_countur_and_draw_box(thresh=thresh)


cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
print(box)
cv2.imshow("found barcode", image)
cv2.waitKey(0)
retval = cv2.imwrite("found.jpg", image)
