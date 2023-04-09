import pyzbar.pyzbar as pyzbar
import imutils
import time
import cv2


vs = cv2.VideoCapture(0)
time.sleep(2.0)
found = set()

while True:
	_, frame = vs.read()
	frame = imutils.resize(frame, width=400)
	barcodes = pyzbar.decode(frame)
	for barcode in barcodes:
		(x, y, w, h) = barcode.rect
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
		barcodeData = barcode.data.decode("utf-8")
		barcodeType = barcode.type
		cv2.putText(frame, f"{barcodeData} ({barcodeType})", (x, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		if barcodeData not in found:
			found.add(barcodeData)
	cv2.imshow("Barcode Scanner", frame)
	key = cv2.waitKey(1) & 0xFF
 
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.release()
