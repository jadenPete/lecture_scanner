#!/usr/bin/env python

import cv2
import sys

def bounding_boxes(image):
	image_blurred = cv2.GaussianBlur(image, (3, 3), 0)
	image_canny = cv2.Canny(image_blurred, 50, 100)
	image_contours = cv2.findContours(
		image_canny,
		mode=cv2.RETR_EXTERNAL,
		method=cv2.CHAIN_APPROX_SIMPLE
	)[0]

	result = []

	for contour in image_contours:
		polygon = cv2.approxPolyDP(contour, 3, True)
		rectangle = cv2.boundingRect(polygon)

		result.append((
			rectangle[0],
			rectangle[1],
			rectangle[0] + rectangle[2],
			rectangle[1] + rectangle[3]
		))

	return result

def box_area(bb):
	return (bb[2] - bb[0]) * (bb[3] - bb[1])

def display_mutably(image, bbs):
	for bb in bbs:
		cv2.rectangle(image, bb[:2], bb[2:], (0, 0, 255), 2)

	cv2.imshow("Bounding boxes", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	cv2.waitKey()
	cv2.destroyAllWindows()

def largest_box_confident(image, largest, second_largest):
	if box_area(largest) / box_area(second_largest) < 1.5:
		return False

	if box_area(largest) / (image.shape[0] * image.shape[1]) < 0.25:
		return False

	if abs((largest[2] - largest[0]) / (largest[3] - largest[1]) - 1) > 0.8:
		return False

	return True

if len(sys.argv) < 2:
	print("Please provide an image path.", file=sys.stderr)
	exit(1)

image = cv2.imread(sys.argv[1])

bbs = bounding_boxes(image)
bbs.sort(key=box_area, reverse=True)

if len(bbs) > 1:
	print(largest_box_confident(image, bbs[0], bbs[1]))

display_mutably(image, bbs[:3])
