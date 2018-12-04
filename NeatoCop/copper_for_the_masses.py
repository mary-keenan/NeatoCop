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

		# initialilze global variables
		self.object_boxes = []
		self.most_recent_image = None
		self.should_follow = False

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


	def process_frame(self, image):
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


	def add_frame(self, boxes, classes = None, scores = None, runner_present = False):
		""" """

		objects_in_frame = []
		num_boxes = len(boxes)
		for i in range(num_boxes):
			# class 1 represents human
			if classes is None or classes[i] == 1 and scores[i] > self.threshold_for_detection:
				box = boxes[i]
				objects_in_frame.append(box)
				color = self.pick_box_color(num_boxes, i, runner_present = True)
				cv2.rectangle(self.most_recent_image,(box[1],box[0]),(box[3],box[2]), color, 2)

		self.object_boxes.append(objects_in_frame)
		cv2.imshow("preview", self.most_recent_image)
		key = cv2.waitKey(1)

		if runner_present:
			cv2.imwrite("traffic_camera_shot.png", self.most_recent_image)


	def pick_box_color(self, num_boxes, i, runner_present):
		""" """
		colors = {"blue": (255,0,0), "green": (0,255,0), "red": (0,0,255)}

		if runner_present:
			if i == 0:
				return colors["red"]
			elif i == num_boxes - 1:
				return colors["blue"]
			else:
				return colors["green"]
		else:
			return colors["blue"]


	def calculate_speed(self):
		""" """
		if len(self.object_boxes) >= 2: # we don't proceed unless there's enough data to estimate speed (aka a start and an end position)

			trackers_list = []

			# follow detected people throughout detection period -- keep updating their current position
			for frame_boxes in self.object_boxes:

				if len(frame_boxes) > len(trackers_list): # determines if new trackers need to be initialized
					num_of_new_trackers = len(frame_boxes) - len(trackers_list)
					for i in range(num_of_new_trackers):
						new_tracker = DetectedHumanTracker()
						trackers_list.append(new_tracker)

				trackers_list = self.minimize_difference(frame_boxes, trackers_list) # updates trackers list with the boxes from the newest frame

			# at the end of the detection period, determine if there were any runners (compare first and last positions of each tracker's boxes)
			for tracker in trackers_list:
				total_movement = self.measure_distance(tracker.initial_box, tracker.most_recent_box)

				if total_movement > self.threshold_for_running:
					print '\a'
					self.add_frame([tracker.initial_box, tracker.most_recent_box], runner_present = True)
					# self.should_follow = True
					# print ("the chase is on!")
					# somehow save which box to follow
					return


		self.object_boxes = []


	def minimize_difference(self, boxes, trackers):
		""" """
		num_of_boxes = len(boxes)
		possible_orderings_of_boxes = list(itertools.permutations(boxes))
		assignment_sum_offset_dict = dict()

		for order in possible_orderings_of_boxes:
			offset = 0
			for i in range(num_of_boxes):
				if trackers[i].initial_box == None:
					offset += 10000
				else:
					offset += self.measure_distance(order[i], trackers[i].most_recent_box)

			assignment_sum_offset_dict[order] = offset

		smallest_offset_order = min(assignment_sum_offset_dict, key = assignment_sum_offset_dict.get)

		for n in range(num_of_boxes):
			if trackers[n].initial_box == None: # this accounts for "new" trackers
				trackers[n].initial_box = smallest_offset_order[n]
			trackers[n].most_recent_box = smallest_offset_order[n]

		return trackers


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
		curr_lower_left_y, curr_lower_left_x, curr_upper_right_y, curr_upper_right_x = self.object_boxes[-1]
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
			boxes, scores, classes, num = self.process_frame(self.most_recent_image)
			# box_list_len = len(self.object_boxes)

			# visualization of the results of a detection
			self.add_frame(boxes, classes, scores)


class DetectedHumanTracker:
	def __init__(self, box = None):
		self.initial_box = box
		self.most_recent_box = box
		self.direction = None
		self.color = list(np.random.choice(range(256), size = 3))


if __name__ == "__main__":
	object_detector = ObjectDetector()
	object_detector.run()
	