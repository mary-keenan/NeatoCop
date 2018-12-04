#!/usr/bin/env python
""" This script turns your Neato into a NeatoCop -- for one person """
import rospy
import numpy as np
import tensorflow as tf
import cv2
import time
import math
from array import array
from geometry_msgs.msg import Twist, Vector3, Pose
from sensor_msgs.msg import LaserScan, Image


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
		self.threshold_for_running = 50 # this is how much the lower left point needs to move per second for the motion to be considered running
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

		# Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
		image_np_expanded = np.expand_dims(image, axis=0)

		# Actual detection.
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

		num_boxes = len(boxes)
		for i in range(num_boxes):
			# class 1 represents human
			if classes is None or classes[i] == 1 and scores[i] > self.threshold_for_detection:
				box = boxes[i]
				self.object_boxes.append(box)
				color = self.pick_box_color(num_boxes, i, runner_present)
				cv2.rectangle(self.most_recent_image,(box[1],box[0]),(box[3],box[2]), color, 2)

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
		if len(self.object_boxes) >= 2:
			s_lower_left_y, s_lower_left_x, s_upper_right_y, s_upper_right_x = self.object_boxes[0]
			e_lower_left_y, e_lower_left_x, e_upper_right_y, e_upper_right_x = self.object_boxes[-1]
			
			# if we assume the person is crossing perfectly perpendicular (and not getting closer or farther away),
			# we only to look at how one of the coordinates changed
			difference_in_position = math.sqrt((e_lower_left_x - s_lower_left_x)**2 + (e_lower_left_y - s_lower_left_y)**2)
			
			# if the person moved enough to be considered "running", the computer beeps
			if difference_in_position > self.threshold_for_running:
				print '\a'
				self.add_frame(self.object_boxes, runner_present = True)
				self.should_follow = True
				print ("the chase is on!")
			else:
				self.object_boxes = []

			print (difference_in_position)


	def update_current_image(self, data):
		""" camera callback -- just saves image as most recent image """
		image = np.fromstring(data.data, np.uint8)
		# self.most_recent_image = cv2.resize(image, (1280, 720))
		self.most_recent_image = image.reshape(480,640,3)


	def follow_perp(self):
		""" """
		# maybe add something so it stops when the lower y value is 0

		# calculate center of mass of box and follow that
		curr_lower_left_y, curr_lower_left_x, curr_upper_right_y, curr_upper_right_x = self.object_boxes[-1]
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
			boxes, scores, classes, num = self.process_frame(self.most_recent_image)
			box_list_len = len(self.object_boxes)

			# visualization of the results of a detection
			self.add_frame(boxes, classes, scores)

			if self.should_follow:
				if len(self.object_boxes) > box_list_len: # checks if there's a frame update
					self.follow_perp()
					start_time = time.time()
				elif time.time() - start_time > 3: # the robot will stop moving if it hasn't seen a new frame recently
					self.vel_msg.linear.x = 0
					self.vel_msg.angular.z = 0

				self.publisher.publish(self.vel_msg)


if __name__ == "__main__":
	object_detector = ObjectDetector()
	object_detector.run()
	