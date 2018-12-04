#!/usr/bin/env python
""" This script turns your Neato into a NeatoCop -- for the masses """
import rospy
import numpy as np
import tensorflow as tf
import cv2
import time
import math
from array import array
from geometry_msgs.msg import Twist, Vector3, Pose
from sensor_msgs.msg import LaserScan, Image
import itertools


class ObjectDetector:
	""" The object detecting model code was adapted from the Tensorflow Object Detection Framework"""
	def __init__(self):
		""" initializes the object_detector object """
		rospy.init_node('copper')
		self.rate = rospy.Rate(100)
		
		# initialize objects
		self.publisher = rospy.Publisher('cmd_vel', Twist, queue_size = 10)
		self.vel_msg = Twist()
		self.vel_msg.linear.x = 0
		self.vel_msg.angular.z = 0

		# initialize parameters
		self.elapsed_time_for_speed_check = 1 # we do a "speed check" every second for an object
		self.threshold_for_running = 150 # this is how much the lower left point needs to move per second for the motion to be considered running
		self.base_angular_speed = .0005
		self.base_linear_speed = .1
		self.overlap_threshold = 50 # used to determine if meanshifted box overlaps with a detected object
		# set up the termination criteria, either 10 iteration or move by atleast 1 pt
		self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

		# initialilze global variables
		self.object_boxes = []
		self.most_recent_image = None
		self.should_follow = False
		self.trackers_list = []
		self.hot_tracker_index = None

		# (part of the tutorial code)
		self.detection_graph = tf.Graph()
		with self.detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile('ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb', 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

		self.default_graph = self.detection_graph.as_default()
		self.sess = tf.Session(graph=self.detection_graph)

		# Definite input and output Tensors for detection_graph
		self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
		
		# Each box represents a part of the image where a particular object was detected.
		self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
		
		# Each score represent how level of confidence for each of the objects.
		# Score is shown on the result image, together with the class label.
		self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
		self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
		self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
		self.threshold_for_detection = 0.7


	def detect_objects(self, image):
		"""" (part of the tutorial code) This applies object_detection model to the most recent frame of the video and returns a list of boxes"""

		# expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
		image_np_expanded = np.expand_dims(image, axis=0)

		# actual detection.
		(boxes, scores, classes, num) = self.sess.run(
			[self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
			feed_dict={self.image_tensor: image_np_expanded})

		im_height, im_width,_ = image.shape
		boxes_list = [None for i in range(boxes.shape[1])]
		for i in range(boxes.shape[1]):
			boxes_list[i] = (int(boxes[0,i,0] * im_height),
						int(boxes[0,i,1]*im_width),
						int(boxes[0,i,2] * im_height),
						int(boxes[0,i,3]*im_width))

		return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])


	def update_trackers(self, boxes, classes, scores):
		""" """
		objects_in_frame = []
		num_boxes = len(boxes)
		for i in range(num_boxes):
			# class 1 represents human
			if classes is None or classes[i] == 1 and scores[i] > self.threshold_for_detection:
				box = boxes[i]
				objects_in_frame.append(box)

		self.object_boxes.append(objects_in_frame)

		# try to find object for each tracker -- if there are any leftover, init new tracker
		hsv = cv2.cvtColor(self.most_recent_image, cv2.COLOR_BGR2HSV)
		for tracker in self.trackers_list:
			dst = cv2.calcBackProject([hsv], [0], tracker.roi_hist, [0,180], 1)
			# apply meanshift to get the new location
			ret, tracker.most_recent_box = cv2.meanShift(dst, tracker.most_recent_box, self.term_crit)
			objects_in_frame = self.remove_box_overlap(objects_in_frame, tracker.most_recent_box)

			# Draw it on image
			# x,y,w,h = tracker.most_recent_box
			# cv2.rectangle(self.most_recent_image, (x,y), (x+w,y+h), tracker.color ,2)
			# cv2.imshow('img2',self.most_recent_image)
			# k = cv2.waitKey(60) & 0xff

		# initialize new trackers
		for box in objects_in_frame:
			box_roi_hist = self.find_roi_histogram(box, self.most_recent_image)
			new_tracker = DetectedHumanTracker(box, box_roi_hist)
			self.trackers_list.append(new_tracker)


	def remove_box_overlap(self, boxes_list, tracker_box):
		""" uses COM to throw out any boxes the tracker covers """
		boxes_to_return = []
		tracker_COM = self.find_COM(tracker_box)

		for box in boxes_list:
			box_COM = self.find_COM(box)
			if self.measure_distance(tracker_COM, box_COM) < self.overlap_threshold:
				boxes_to_return.append(box)

		return boxes_to_return


	def find_COM(self, box):
		""" """
		lower_left_y, lower_left_x, upper_right_y, upper_right_x = box
		return (upper_right_x - lower_left_x, upper_right_y - lower_left_y)


	def find_roi_histogram(self, box, frame):
		""" """
		# set up the ROI for tracking
		r,h,c,w = box
		roi = frame[r:r+h, c:c+w]
		hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
		roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
		cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
		return roi_hist


	def add_tracker_frames(self):
		""" """
		for tracker in self.trackers_list:
			x,y,w,h = tracker.most_recent_box
			cv2.rectangle(self.most_recent_image, (x,y), (x + w, y + h), tracker.color, 2)

		cv2.imshow("preview", self.most_recent_image)
		key = cv2.waitKey(1)


	def take_traffic_camera_snapshot(self, start_box, end_box):
		""" """
		x,y,w,h = start_box
		cv2.rectangle(self.most_recent_image, (x,y), (x + w, y + h), (255,0,0), 2)
		x,y,w,h = end_box
		cv2.rectangle(self.most_recent_image, (x,y), (x + w, y + h), (0,255,0), 2)
		cv2.imwrite("traffic_camera_shot.png", self.most_recent_image)


	def calculate_speed(self):
		""" """
		if len(self.object_boxes) >= 2: # we don't proceed unless there's enough data to estimate speed (aka a start and an end position)

			# at the end of the detection period, determine if there were any runners (compare first and last positions of each tracker's boxes)
			for n in range(len(self.trackers_list)):
				tracker = self.trackers_list[n]
				total_movement = self.measure_distance(tracker.initial_box, tracker.most_recent_box)

				if total_movement > self.threshold_for_running:
					print '\a'
					self.take_traffic_camera_snapshot(tracker.initial_box, tracker.most_recent_box)
					self.should_follow = True
					print ("the chase is on!")
					self.hot_tracker_index = n
					return

		self.object_boxes = []
		self.trackers_list = []


	def measure_distance(self, box_1, box_2):
		""" """
		if len(box_1) > 2:
			s_lower_left_y, s_lower_left_x, s_upper_right_y, s_upper_right_x = box_1
			e_lower_left_y, e_lower_left_x, e_upper_right_y, e_upper_right_x = box_2
		else: 
			s_lower_left_y, s_lower_left_x = box_1
			e_lower_left_y, e_lower_left_x = box_2

		# we just track the lower left corner since the COM might confuse two people on top of each other (one is closer than the other)
		difference_in_position = math.sqrt((e_lower_left_x - s_lower_left_x)**2 + (e_lower_left_y - s_lower_left_y)**2)
		return (difference_in_position)


	def update_current_image(self, data):
		""" camera callback -- just saves image as most recent image """
		image = np.fromstring(data.data, np.uint8)
		self.most_recent_image = image.reshape(480,640,3)


	def follow_perp(self):
		""" """
		# calculate center of mass of box and follow that
		curr_lower_left_y, curr_lower_left_x, curr_upper_right_y, curr_upper_right_x = self.trackers_list[self.hot_tracker_index].most_recent_box
		center_of_mass_x = (curr_lower_left_x + curr_upper_right_x) / 2
		distance_from_center_x = 320 - center_of_mass_x # if the center of mass is greater than 640, it's to the right and the angular speed should be negative
		self.vel_msg.linear.x = self.base_linear_speed
		self.vel_msg.angular.z = self.base_angular_speed * distance_from_center_x
		print(distance_from_center_x)


	def run(self):
		""" """
		# rospy.Subscriber('camera/image_raw', Image, self.update_current_image)
		
		# # wait for first image data before starting the general run loop
		# while self.most_recent_image is None and not rospy.is_shutdown():
		# 	self.rate.sleep()

		video_capture = cv2.VideoCapture(0)
		start_time = time.time()

		while not rospy.is_shutdown():

			# check the speed of the box every second unless following
			elapsed_time = time.time() - start_time
			if not self.should_follow and elapsed_time >= self.elapsed_time_for_speed_check:
				self.calculate_speed()
				start_time = time.time()

			# continue with object detection
			ret, frame = video_capture.read()
			self.most_recent_image = cv2.resize(frame, (1280, 720))
			boxes, scores, classes, num = self.detect_objects(self.most_recent_image)
			self.update_trackers(boxes, classes, scores)

			# visualization of the results of a detection
			self.add_tracker_frames()


class DetectedHumanTracker:
	def __init__(self, box, roi_hist):
		self.initial_box = box
		self.most_recent_box = box
		self.color = list(np.random.choice(range(256), size = 3))
		self.roi_hist = roi_hist


if __name__ == "__main__":
	object_detector = ObjectDetector()
	object_detector.run()
	