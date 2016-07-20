import numpy as np

import cv2

key_wait_time = 10

cap = cv2.VideoCapture(0)
n = 0

while True:
	success, frame = cap.read()

	if success:
		cv2.imshow('video', frame)

		ch = cv2.waitKey(key_wait_time) & 0xFF
		if ch == 27:
			break
		if ch == ord(' '):
			name = 'image%04d.png' % n
			cv2.imwrite(name, frame)
			print 'wrote: %s' % name
			n += 1

capture.release()
cv2.destroyAllWindows()
