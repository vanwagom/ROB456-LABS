#!/usr/bin/env python3


import rospy
import sys

from math import atan2, tanh, sqrt, pi
import numpy as np

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

import actionlib
import tf

from lab2.msg import NavTargetAction, NavTargetResult, NavTargetFeedback


class Driver:
	def __init__(self, position_source, threshold=0.1):
		# Goal will be set later. The action server will set the goal; you don't set it directly
		self.goal = None
		self.threshold = threshold

		self.transform_listener = tf.TransformListener()

		# Publisher before subscriber
		self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
		self.target_pub = rospy.Publisher('current_target', Marker, queue_size=1)

		# Subscriber after publisher
		self.sub = rospy.Subscriber('base_scan', LaserScan, self._callback, queue_size=1)

		# Action client
		self.action_server = actionlib.SimpleActionServer('nav_target', NavTargetAction, execute_cb=self._action_callback, auto_start=False)
		self.action_server.start()

	@classmethod
	def zero_twist(cls):
		"""This is a helper class method to create and zero-out a twist"""
		command = Twist()
		command.linear.x = 0.0
		command.linear.y = 0.0
		command.linear.z = 0.0
		command.angular.x = 0.0
		command.angular.y = 0.0
		command.angular.z = 0.0

		return command

	# Respond to the action request.
	def _action_callback(self, goal):
		""" This gets called when an action is received by the action server
		@goal - this is the new goal """
		rospy.loginfo(f'Got an action request for ({goal.goal.point.x:.2f}, {goal.goal.point.y:.2f})')

		# Set the goal.
		self.goal = goal.goal

		# Build a marker for the goal point
		#   - this prints out the green dot in RViz (the current goal)
		marker = Marker()
		marker.header.frame_id = goal.goal.header.frame_id
		marker.header.stamp = rospy.Time.now()
		marker.id = 0
		marker.type = Marker.SPHERE
		marker.action = Marker.ADD
		marker.pose.position = goal.goal.point
		marker.pose.orientation.x = 0.0
		marker.pose.orientation.y = 0.0
		marker.pose.orientation.z = 0.0		
		marker.pose.orientation.w = 1.0
		marker.scale.x = 0.3
		marker.scale.y = 0.3
		marker.scale.z = 0.3
		marker.color.r = 0.0
		marker.color.g = 1.0
		marker.color.b = 0.0
		marker.color.a = 1.0

		# Wait until we're at the goal.  Once we get there, the callback that drives the robot will set self.goal
		# to None.
		while self.goal:
			self.target_pub.publish(marker)
			rospy.sleep(0.1)

		rospy.loginfo('Action completed')

		# Build a result to send back
		result = NavTargetResult()
		result.success.data = True

		self.action_server.set_succeeded(result)

		# Get rid of the marker
		marker.action = Marker.DELETE
		self.target_pub.publish(marker)

	def _callback(self, lidar):
		# If we have a goal, then act on it, otherwise stay still
		if self.goal:
			# Update the timestamp on the goal and figure out where it it now in the base_link frame.
			self.goal.header.stamp = rospy.Time.now()
			target = self.transform_listener.transformPoint('base_link', self.goal)

			rospy.loginfo(f'Target: ({target.point.x:.2f}, {target.point.y:.2f})')

			# Are we close enough?  If so, then remove the goal and stop
			distance = sqrt(target.point.x ** 2 + target.point.y ** 2)

			feedback = NavTargetFeedback()
			feedback.distance.data = distance
			self.action_server.publish_feedback(feedback)

			if distance < self.threshold:
				self.goal = None
				command = Driver.zero_twist()
			else:
				command = self.get_twist((target.point.x, target.point.y), lidar)
		else:
			command = Driver.zero_twist()

		self.cmd_pub.publish(command)

	# This is the function that controls the robot.
	#
	# Inputs:
	# 	target:	a tuple with the (x, y) coordinates of the target point, in the robot's coordinate frame (base_link).
	# 			x-axis is forward, y-axis is to the left.
	# 	lidar:	a LaserScan message with the current data from the LiDAR.  Use this for obstacle avoidance.
	#           This is the same as your go and stop code
	def get_twist(self, target, lidar):
		command = Driver.zero_twist()

		# TODO:
		#  Step 1) Calculate the angle the robot has to turn to in order to point at the target
		#  Step 2) Set your speed based on how far away you are from the target, as before
		#  Step 3) Add code that veers left (or right) to avoid an obstacle in front of it

  # YOUR CODE HERE

		target_x = target[0]
		target_y = target[1]
		target_distance = sqrt(target_x ** 2 + target_y ** 2)
		target_theta = atan2(target_y, target_x)
		shoulder_width = 0.38

		# Init the angles and distances for storing points infront of the robot
		points = {
			"thetas": [],
			"distances": []
		}

		# Code to determine whether the robot should rotate, or move forward. loop through ranges
		for i, range in enumerate(lidar.ranges):
			if not np.isfinite(range) or range <= 0:
				continue # Skip invalid values

			theta = lidar.angle_min + ( lidar.angle_increment * i )
			points["thetas"].append(theta)
			points["distances"].append(range)
		
		# Find the distance reading that matches the target theta in points, find the closest point in the direction of the target
		closest_theta = np.argmin([abs(theta - target_theta) for theta in points["thetas"]])
		matching_distance = points["distances"][closest_theta]


		if matching_distance < target_distance:
			# There is something in the way between the robot and the target, search through the points to find an clear path
			sorted_points = np.argsort([abs(theta - target_theta) for theta in points["thetas"]])
			
			clear_theta = None
			for i in sorted_points:
				if points["distances"][i] > target_distance:
					clear_theta = points["thetas"][i]
					break
			
			# If there is a clear path, turn to face the clear path and move the distance of the object
			if clear_theta is not None:
				command.angular.z = clear_theta
				command.linear.x = tanh(matching_distance)
				print(f"Turning to face {clear_theta:.2f} and moving {matching_distance:.2f}")
			else:
				# There is no clear path, stop the robot
				command.linear.x = 0.0
				print("No clear path, stopping")
		
		# If no obstacle between robot and target, continue forward
		else:
			command.linear.x = 0.1
			print(f"No obstacle, moving forward")
		
		return command
	

if __name__ == '__main__':
	rospy.init_node('driver', argv=sys.argv)

	driver = Driver('odom')

	rospy.spin()
