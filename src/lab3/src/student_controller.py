#!/usr/bin/env python3


import sys
import rospy
import signal
import numpy as np

from controller import RobotController
import path_planning
import exploring


def world_to_map(x, y, map_data):
    grid_x = int((x - map_data.origin.position.x) / map_data.resolution)
    grid_y = int((y - map_data.origin.position.y) / map_data.resolution)

    x_out_of_bounds = grid_x < 0 or grid_x >= map_data.width
    y_out_of_bounds = grid_y < 0 or grid_y >= map_data.height
    out_of_bounds = x_out_of_bounds or y_out_of_bounds

    if out_of_bounds:
        return None

    return (grid_y, grid_x)


class StudentController(RobotController):
    """
    This class allows you to set waypoints that the robot will follow.  These robots should be in the map
    coordinate frame, and will be automatically sent to the code that actually moves the robot, contained in
    StudentDriver.
    """
    def __init__(self):
        super().__init__()

        self.goal = None

    def distance_update(self, distance):
        """
        This function is called every time the robot moves towards a goal.  If you want to make sure that
        the robot is making progress towards its goal, then you might want to check that the distance to
        the goal is generally going down.  If you want to change where the robot is heading to, you can
        make a call to set_waypoints here.  This call will override the current set of waypoints, and the
        robot will start to drive towards the first waypoint in the new list.

        Parameters:
            distance:	The distance to the current goal.
        """
        rospy.loginfo(f'Distance: {distance}')

    def map_update(self, point, map, map_data):
        """
        This function is called every time a new map update is available from the SLAM system.  If you want
        to change where the robot is driving, you can do it in this function.  If you generate a path for
        the robot to follow, you can pass it to the driver code using set_waypoints().  Again, this will
        override any current set of waypoints that you might have previously sent.

        Parameters:
            point:		A PointStamped containing the position of the robot, in the map coordinate frame.
            map:		An OccupancyGrid containing the current version of the map.
            map_data:	A MapMetaData containing the current map meta data.
        """
        rospy.loginfo('Got a map update.')

        try:
            # The (x, y) position of the robot can be retrieved like this.
            robot_position = (point.point.x, point.point.y)
            x = int(robot_position[0] / map_data.resolution + map.info.width / 2)
            y = int(robot_position[1] / map_data.resolution + map.info.height / 2)
            robot_in_map = (x, y)
            print("robot_in_map: ", robot_in_map)

            im = np.array(map.data).reshape(map.info.height, map.info.width)
            im_thresh = path_planning.convert_image(im, 0.7, 0.9)
            print("got image threshold")

            fatten_pixels = int(np.ceil(0.19 / map_data.resolution)) + 1
            im_thresh_fattened = path_planning.fatten_image(im_thresh, fatten_pixels)
            print("got fat image :3")

            if self.goal is None:
                all_unseen = exploring.find_all_possible_goals(im_thresh_fattened)
                if all_unseen is None:
                    rospy.loginfo('Done, Stopped!')
                    rospy.signal_shutdown('Done exploring!')

                    return

                print("trying for best points")
                self.goal = exploring.find_best_point(im_thresh_fattened, all_unseen, robot_loc=robot_in_map)
                print("got best points")

            path = path_planning.dijkstra(im_thresh_fattened, robot_in_map, self.goal)
            print("dijkstra done")
            waypoints = exploring.find_waypoints(im_thresh, path)
            print("waypoints done, setting...")

            # waypoints = [((x - 2000) * map_data.resolution, (y - 2000) * map_data.resolution) for x, y in waypoints]
            self.set_waypoints(waypoints)

            # rospy.loginfo(f'Robot is at {robot_position} {point.header.frame_id}')
        except:
            rospy.loginfo('No odometry information')


if __name__ == '__main__':
    # Initialize the node.
    rospy.init_node('student_controller', argv=sys.argv)

    # Start the controller.
    controller = StudentController()

    # This will move the robot to a set of fixed waypoints.  You should not do this, since you don't know
    # if you can get to all of these points without building a map first.  This is just to demonstrate how
    # to call the function, and make the robot move as an example.
    # controller.set_waypoints(((-4, -3), (-4, 0), (5, 0)))

    # Once you call this function, control is given over to the controller, and the robot will start to
    # move.  This function will never return, so any code below it in the file will not be executed.
    controller.send_points()
