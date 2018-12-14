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


class NeatoCop:
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
		self.threshold_for_running = 200 # this is how much the lower left point needs to move per second for the motion to be considered running
		self.base_angular_speed = .002
		self.base_linear_speed = .2
		self.max_time_since_last_update = 3 # if the Neato has not seen the person it's following in 3 seconds, it stops

		# initialilze global variables
		self.object_boxes = [] # we use this to see if we have detected an object two or more times in a second 
		self.most_recent_image = None # the image last updated by the callback function
		self.should_follow = False # determines if the NeatoCop+ enters following "perp" mode
		self.trackers_list = [] # contains a DetectedHumanTracker for each detected object
		self.hot_tracker_index = None # identifies the "guilty" tracker -- the tracker associated with the running object

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

		# apply the actual detection
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
		""" updates the trackers list with the new object detections """

		# filter out non-human object detections or detections with low confidence
		objects_in_frame = []
		num_boxes = len(boxes)
		for i in range(num_boxes):
			# class 1 represents humans
			if classes is None or classes[i] == 1 and scores[i] > self.threshold_for_detection:
				box = boxes[i]
				objects_in_frame.append(box)

		# determine if new trackers need to be initialized and add them if they do
		if len(objects_in_frame) > len(self.trackers_list): 
			num_of_new_trackers = len(objects_in_frame) - len(self.trackers_list)
			for i in range(num_of_new_trackers):
				new_tracker = DetectedHumanTracker()
				self.trackers_list.append(new_tracker)

		# updates trackers list with the boxes from the newest frame
		self.trackers_list = self.minimize_difference(objects_in_frame) 
		self.object_boxes.append(objects_in_frame)


	def add_tracker_frames(self):
		""" overlays the most recently detected boxes on the camera feed for visualization"""

		# draws a box on the screen for each detected object in the color of that object's tracker
		for tracker in self.trackers_list:
			if tracker.recently_updated:
				cv2.rectangle(self.most_recent_image, (tracker.most_recent_box[1], tracker.most_recent_box[0]), 
					(tracker.most_recent_box[3], tracker.most_recent_box[2]), tracker.color, 2)

		cv2.imshow("preview", self.most_recent_image)
		key = cv2.waitKey(1)


	def take_traffic_camera_snapshot(self, start_box, end_box):
		""" overlays the starting and end position of the guilty object when they were caught running and saves the image """
		cv2.rectangle(self.most_recent_image, (start_box[1], start_box[0]), (start_box[3], start_box[2]), (255,0,0), 2)
		cv2.rectangle(self.most_recent_image, (end_box[1], end_box[0]), (end_box[3], end_box[2]), (0,255,0), 2)
		cv2.imwrite("traffic_camera_shot.png", self.most_recent_image)


	def calculate_speed(self):
		""" determines if the speed of any of the objects that were being tracked for the last second is higher than the self.threshold_for_running """

		# if we have enough data to estimate speed (aka a start and end position), calculate the speed of each of the trackers
		if len(self.object_boxes) >= 2:

			# at the end of the detection period, determine if there were any runners (compare first and last positions of each tracker's boxes)
			for n in range(len(self.trackers_list)):
				tracker = self.trackers_list[n]
				total_movement = self.measure_distance(tracker.initial_box, tracker.most_recent_box)

				# if the movement per second was too high, the person-object is guilty of speeding
				if total_movement > self.threshold_for_running:
					print ('\a') # this is supposed to make a bell sound but my laptop doesn't play sound anymore so I'm not sure it really does...
					self.take_traffic_camera_snapshot(tracker.initial_box, tracker.most_recent_box) # just like real traffic cameras, we need evidence
					self.should_follow = True 
					print ("The chase is on!") # the NeatoCop catchphrase
					self.hot_tracker_index = n # save the index of the guilty tracker globally so we follow the right object
					return

		self.object_boxes = []
		self.trackers_list = [] # we want to start again with fresh trackers to compensate for people walking out of the frame


	def minimize_difference(self, boxes):
		""" assign each of the most recently detected boxes to a DetectedHumanTracker in a way that minimizes the total difference 
		between the boxes and the trackers' last boxes -- this is a hacky solution """

		# if there are more trackers than detected objects, add placeholder objects so we have equal list sizes
		if len(boxes) < len(self.trackers_list):
			num_of_placeholder_boxes = len(self.trackers_list) - len(boxes)
			for i in range(num_of_placeholder_boxes):
				boxes.append(None)

		# create a list of all the possible ways the boxes could be assigned to trackers
		possible_orderings_of_boxes = list(itertools.permutations(boxes)) 

		# run through all of the possible ways the boxes could be assigned to trackers
		num_of_boxes = len(boxes)
		assignment_sum_offset_dict = dict()
		for order in possible_orderings_of_boxes:
			offset = 0
			# for each box-tracker pairing, sum the distance between the box and the tracker's last known position (box)
			for i in range(num_of_boxes):
				if self.trackers_list[i].initial_box == None or order[i] == None: # if it's an uninitialized tracker or placeholder box, every other possible assignation should be preferable
					offset += 10000
				else:
					offset += self.measure_distance(order[i], self.trackers_list[i].most_recent_box)

			assignment_sum_offset_dict[order] = offset

		# determine which way of assigning the boxes to the trackers had the lowest overall difference between the boxes and their trackers 
		smallest_offset_order = min(assignment_sum_offset_dict, key = assignment_sum_offset_dict.get)

		# assign the boxes to their trackers based on the pairings in smallest_offset_order
		for n in range(num_of_boxes):
			if smallest_offset_order[n] != None:
				if self.trackers_list[n].initial_box == None: # this accounts for "new" trackers that have not yet been initialized with a box
					self.trackers_list[n].initial_box = smallest_offset_order[n]
				self.trackers_list[n].most_recent_box = smallest_offset_order[n]
				self.trackers_list[n].recently_updated = True
			else: # if a tracker was assigned a placeholder box, it did not actually get updated
				self.trackers_list[n].recently_updated = False

		return self.trackers_list


	def measure_distance(self, box_1, box_2):
		""" calculates the distance between two boxes' lower-left corners"""
		s_lower_left_y, s_lower_left_x, s_upper_right_y, s_upper_right_x = box_1
		e_lower_left_y, e_lower_left_x, e_upper_right_y, e_upper_right_x = box_2
		# we just track the lower left corner since the COM might confuse two people on top of each other (one is closer than the other)
		difference_in_position = math.sqrt((e_lower_left_x - s_lower_left_x)**2 + (e_lower_left_y - s_lower_left_y)**2)
		return (difference_in_position)


	def update_current_image(self, data):
		""" saves the camera callback function's image as the most recent image """
		image = np.fromstring(data.data, np.uint8)
		self.most_recent_image = image.reshape(480,640,3)


	def follow_perp(self):
		""" calculates the center of mass of the guilty object's box and adjusts the angular velocity based on its location on the x axis"""
		curr_lower_left_y, curr_lower_left_x, curr_upper_right_y, curr_upper_right_x = self.trackers_list[self.hot_tracker_index].most_recent_box
		center_of_mass_x = (curr_lower_left_x + curr_upper_right_x) / 2
		distance_from_center_x = 320 - center_of_mass_x # if the center of mass is greater than 640, it's to the right and the angular speed should be negative
		self.vel_msg.linear.x = self.base_linear_speed
		self.vel_msg.angular.z = self.base_angular_speed * distance_from_center_x
		print(distance_from_center_x)


	def run(self):
		""" runs the NeatoCop+ program indefinitely """

		rospy.Subscriber('camera/image_raw', Image, self.update_current_image)
		
		# wait for first image data before starting the general run loop
		while self.most_recent_image is None and not rospy.is_shutdown():
			self.rate.sleep()

		start_time = time.time()

		while not rospy.is_shutdown():

			# check the speed of the box every second unless following
			elapsed_time = time.time() - start_time
			if not self.should_follow and elapsed_time >= self.elapsed_time_for_speed_check:
				self.calculate_speed()
				start_time = time.time()

			# apply object detection
			boxes, scores, classes, num = self.process_frame(self.most_recent_image)

			# apply object tracking
			self.update_trackers(boxes, classes, scores)

			# visualize the detection results
			self.add_tracker_frames()

			# follow the perp
			if self.should_follow:
				if self.trackers_list[self.hot_tracker_index].recently_updated: # checks if there's a frame update
					self.follow_perp()
					start_time = time.time()
				elif time.time() - start_time > self.max_time_since_last_update: # the robot will stop moving if it hasn't seen a new frame recently
					self.vel_msg.linear.x = 0
					self.vel_msg.angular.z = 0

				self.publisher.publish(self.vel_msg)


class DetectedHumanTracker:
	""" the tracker assigned to each detected object """
	def __init__(self, box = None):
		self.initial_box = box
		self.most_recent_box = box
		self.recently_updated = False
		self.color = list(np.random.choice(range(256), size = 3)) # this reandomly generated color is used for the visualizations


if __name__ == "__main__":
	object_detector = NeatoCop()
	object_detector.run()
	