#!/usr/bin/env python
""" This script turns your Neato into a NeatoCop -- for the masses -- with CAMshifting """
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
		self.threshold_for_running = 200 # f/s
		self.base_angular_speed = .002
		self.base_linear_speed = .2
		self.overlap_threshold = 100 # used to determine if meanshifted box overlaps with a detected object
		self.width_of_person_in_feet = 1 # in feet
		# set up the termination criteria for CAMshifting, either 10 iteration or move by at least 1 pt
		self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

		# initialilze global variables
		self.object_boxes = []
		self.most_recent_image = None
		self.frozen_most_recent_image = None
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
		for n in range(len(self.trackers_list)):
			dst = cv2.calcBackProject([hsv], [0], self.trackers_list[n].roi_hist, [0,180], 1)
			# apply CAMshift to get the new location
			ret, self.trackers_list[n].most_recent_box = cv2.CamShift(dst, self.trackers_list[n].most_recent_box, self.term_crit)

		# determine if new trackers need to be initialized and add them if they do
		if len(objects_in_frame) > len(self.trackers_list): 
			num_of_new_trackers = len(objects_in_frame) - len(self.trackers_list)
			for i in range(num_of_new_trackers):
				new_tracker = DetectedHumanTracker()
				self.trackers_list.append(new_tracker)

		self.trackers_list = self.minimize_difference(objects_in_frame) # updates trackers list with the boxes from the newest frame


	def minimize_difference(self, boxes):
		""" """
		if len(boxes) < len(self.trackers_list):
			num_of_placeholder_boxes = len(self.trackers_list) - len(boxes)
			for i in range(num_of_placeholder_boxes):
				boxes.append(None)

		num_of_boxes = len(boxes)
		possible_orderings_of_boxes = list(itertools.permutations(boxes))
		assignment_sum_offset_dict = dict()

		for order in possible_orderings_of_boxes:
			offset = 0
			for i in range(num_of_boxes):
				if self.trackers_list[i].initial_box == None or order[i] == None:
					offset += 10000
				else:
					offset += self.measure_distance(order[i], self.trackers_list[i].most_recent_box)

			assignment_sum_offset_dict[order] = offset

		smallest_offset_order = min(assignment_sum_offset_dict, key = assignment_sum_offset_dict.get)

		for n in range(num_of_boxes):
			if smallest_offset_order[n] != None:
				if self.trackers_list[n].initial_box == None: # this accounts for "new" trackers
					self.trackers_list[n].initial_box = smallest_offset_order[n]
					self.trackers_list[n].roi_hist = self.find_roi_histogram(smallest_offset_order[n], self.most_recent_image)
				self.trackers_list[n].most_recent_box = smallest_offset_order[n]
				self.trackers_list[n].recently_updated = True
			else:
				self.trackers_list[n].recently_updated = False

		return self.trackers_list


	def find_COM(self, box):
		""" """
		lower_left_y, lower_left_x, upper_right_y, upper_right_x = box
		return ((upper_right_x + lower_left_x) / 2, (upper_right_y + lower_left_y) / 2)


	def find_roi_histogram(self, box, frame):
		""" """
		# set up the ROI for tracking
		lx,ly,tx,ty = box
		roi = frame[lx:tx,ly:ty]
		hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
		roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
		cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
		return roi_hist


	def add_tracker_frames(self):
		""" """
		for tracker in self.trackers_list:
			cv2.rectangle(self.most_recent_image, (tracker.most_recent_box[1], tracker.most_recent_box[0]), 
				(tracker.most_recent_box[3], tracker.most_recent_box[2]), tracker.color, 2)

		cv2.imshow("preview", self.most_recent_image)
		key = cv2.waitKey(1)


	def take_traffic_camera_snapshot(self, start_box, end_box):
		""" """
		cv2.rectangle(self.most_recent_image, (start_box[1], start_box[0]), (start_box[3], start_box[2]), (255,0,0), 2)
		cv2.rectangle(self.most_recent_image, (end_box[1], end_box[0]), (end_box[3], end_box[2]), (0,255,0), 2)
		cv2.imwrite("traffic_camera_shot.png", self.most_recent_image)


	def calculate_speed(self):
		""" """
		if len(self.object_boxes) >= 2: # we don't proceed unless there's enough data to estimate speed (aka a start and an end position)

			# at the end of the detection period, determine if there were any runners (compare first and last positions of each tracker's boxes)
			for n in range(len(self.trackers_list)):
				tracker = self.trackers_list[n]
				total_movement = self.measure_distance(tracker.initial_box, tracker.most_recent_box)
				print (total_movement)

				if total_movement > self.threshold_for_running:
					print ('\a')
					self.take_traffic_camera_snapshot(tracker.initial_box, tracker.most_recent_box)
					self.should_follow = True
					print ("the chase is on!")
					self.hot_tracker_index = n
					return

		self.object_boxes = []
		self.trackers_list = []


	def measure_distance(self, box_1, box_2):
		""" """
		s_lower_left_y, s_lower_left_x, s_upper_right_y, s_upper_right_x = box_1
		e_lower_left_y, e_lower_left_x, e_upper_right_y, e_upper_right_x = box_2
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
		rospy.Subscriber('camera/image_raw', Image, self.update_current_image)
		
		# wait for first image data before starting the general run loop
		while self.most_recent_image is None and not rospy.is_shutdown():
			self.rate.sleep()

		# video_capture = cv2.VideoCapture(0)
		start_time = time.time()

		while not rospy.is_shutdown():

			# check the speed of the box every second unless following
			elapsed_time = time.time() - start_time
			if not self.should_follow and elapsed_time >= self.elapsed_time_for_speed_check:
				self.calculate_speed()
				start_time = time.time()

			# continue with object detection
			# ret, frame = video_capture.read()
			# self.most_recent_image = cv2.resize(frame, (1280, 720))
			boxes, scores, classes, num = self.detect_objects(self.most_recent_image)
			self.update_trackers(boxes, classes, scores)

			# visualization of the results of a detection
			self.add_tracker_frames()

			# follow the perp
			if self.should_follow:
				if self.trackers_list[self.hot_tracker_index].recently_updated: # checks if there's a frame update
					self.follow_perp()
					start_time = time.time()
				elif time.time() - start_time > 3: # the robot will stop moving if it hasn't seen a new frame recently
					self.vel_msg.linear.x = 0
					self.vel_msg.angular.z = 0

				self.publisher.publish(self.vel_msg)


class DetectedHumanTracker:
	def __init__(self, box = None, roi_hist = None):
		self.initial_box = box
		self.most_recent_box = box
		self.color = list(np.random.choice(range(256), size = 3))
		self.roi_hist = roi_hist
		self.recently_updated = False


if __name__ == "__main__":
	object_detector = ObjectDetector()
	object_detector.run()
	