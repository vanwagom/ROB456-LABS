#!/usr/bin/env python3


import sys
import rospy

from new_driver import Driver

from math import atan2, sqrt, tanh
import numpy as np


class StudentDriver(Driver):
	'''
	This class implements the logic to move the robot to a specific place in the world.
	All the interesting functionality is hidden in the parent class.
	'''
	def __init__(self, threshold=0.1):
		super().__init__('odom')
		# Set the threshold to a reasonable number
		self._threshold = threshold

	def close_enough_to_waypoint(self, distance, target, lidar):
		'''
		This function is called perioidically if there is a waypoint set.  This is where you should put any code that
		has a smarter stopping criteria then just checking the distance. See get_twist for the parameters; distance
		is the current distance to the target.
		'''
		# Default behavior.
		if distance < self._threshold:
			return True
		return False

	def get_twist(self, target, lidar):
		'''
		This function is called whenever there a current target is set and there is a lidar data
		available.  This is where you should put your code for moving the robot.  The target point
		is in the robot's coordinate frame.  The x-axis is positive-forwards, and the y-axis is
		positive to the left.

		The example sets constant velocities, which is clearly the wrong thing to do.  Replace this
		code with something that moves the robot more intelligently.

		Parameters:
			target:		The current target point, in the coordinate frame of the robot (base_link) as
						an (x, y) tuple.
			lidar:		A LaserScan containing the new lidar data.

		Returns:
			A Twist message, containing the commanded robot velocities.
		'''
		command = Driver.zero_twist()

		# TODO:
		#  Step 1) Calculate the angle the robot has to turn to in order to point at the target
		#  Step 2) Set your speed based on how far away you are from the target, as before
		#  Step 3) Add code that veers left (or right) to avoid an obstacle in front of it

		target_x = target[0]
		target_y = target[1]
		target_distance = sqrt(target_x ** 2 + target_y ** 2)
		target_theta = atan2(target_y, target_x)
		shoulder_width = 0.38

		# Init the angles and distances for storing points in front of the robot
		points = {
			"thetas": [],
			"distances": []
		}

		# Code to determine whether the robot should rotate, or move forward. loop through ranges
		for i, ind_range in enumerate(lidar.ranges):
			if not np.isfinite(ind_range) or ind_range <= 0:
				continue  # Skip invalid values

			theta = lidar.angle_min + (lidar.angle_increment * i)

			points["thetas"].append(theta)
			points["distances"].append(ind_range)

		# Find the distance reading that matches the target theta in points, find the closest point in the direction
		# of the target
		theta_differences = [abs(theta - target_theta) for theta in points["thetas"]]
		if theta_differences:
			closest_theta_idx = np.argmin(theta_differences)
			matching_distance = points["distances"][closest_theta_idx]
		else:
			matching_distance = float('inf')  # No obstacle in the target direction

		if matching_distance < target_distance:
			# There is something in the way between the robot and the target, search through the points to find an
			# clear path
			sorted_points = np.argsort(theta_differences)

			clear_theta = None
			for i, theta in enumerate(points["thetas"]):
				y = points["distances"][i] * np.sin(theta)
				if abs(y) > shoulder_width / 2 and points["distances"][i] > 2.0:
					clear_theta = theta
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
			command.angular.z = target_theta
			command.linear.x = tanh(target_distance)
			print(f"No obstacle, moving toward target")

		return command


if __name__ == '__main__':
	rospy.init_node('student_driver', argv=sys.argv)

	driver = StudentDriver()

	rospy.spin()
