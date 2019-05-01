#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight, Waypoint
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import sys
import cv2
import yaml
import math
from timeit import default_timer as timer

STATE_COUNT_THRESHOLD = 3
DISTANCE_THRESHOLD = 280


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.image_processing_time = 0
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        self.state = TrafficLight.UNKNOWN
        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.is_sim = not self.config["is_site"]

        print("Is Sim %d" % self.is_sim)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(self.is_sim)
        self.listener = tf.TransformListener()

        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        # Skip Traffic Light Detection if processing takes longer than current frequency, avoid queueing
        if self.image_processing_time > 0:
            self.image_processing_time -= self.sleep(10.0)
            return
            # rospy.logwarn("Skipping traffic light detection for this frame ...: %f s", self.image_processing_time)

        self.has_image = True
        self.camera_image = msg

        start_detection = timer()
        light_wp, nn_output = self.process_traffic_lights()
        end_detection = timer()
        self.image_processing_time = end_detection - start_detection

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        # print(nn_output)
        state, prob = nn_output

        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if (state == TrafficLight.RED or state == TrafficLight.YELLOW) else -1
            self.last_wp = light_wp

            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        return self.get_closest_index(pose, self.waypoints.waypoints)

    def get_closest_light(self, pose):
        return self.get_closest_index(pose, self.lights)

    def get_closest_stop_line(self, pose):
        return self.get_closest_index(pose, self.get_stop_line_3d_positions())

    def get_car_waypoint_index(self):
        return self.get_closest_waypoint(self.pose.pose.position)

    def get_car_waypoint_position(self):
        car_inx = self.get_car_waypoint_index()
        car_pos = self.waypoints.waypoints[car_inx].pose.pose.position
        return car_pos

    def sleep(self, time):
        return time / 100.0

    def get_stop_line_3d_positions(self):
        stop_line_positions = []
        for light_position in self.config['stop_line_positions']:
            p = Waypoint()
            p.pose.pose.position.x = light_position[0]
            p.pose.pose.position.y = light_position[1]
            p.pose.pose.position.z = 0.0
            stop_line_positions.append(p)
        return stop_line_positions

    def get_closest_index(self, pose, positions):
        minimal_distance = sys.maxsize
        index = -1

        for i in range(len(positions)):
            distance = self.get_distance(pose, positions[i].pose.pose.position)
            if distance < minimal_distance:
                minimal_distance = distance
                index = i

        return index

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if (not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        # Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection

        stop_waypoint = None

        if (self.pose):

            # get car index, car position
            car_waypoiny_inx = self.get_car_waypoint_index()
            car_waypoint_pos = self.get_car_waypoint_position()

            # get light index
            light_inx = self.get_closest_light(car_waypoint_pos)

            if light_inx != -1:

                # get closest waypoint to light index
                light_waypoint_inx = self.get_closest_waypoint(self.lights[light_inx].pose.pose.position)
                light_pos = self.waypoints.waypoints[light_waypoint_inx].pose.pose.position

                # get stop line waypoint
                if light_waypoint_inx > car_waypoiny_inx:
                    distance_to_traffic_light = self.get_distance(car_waypoint_pos, light_pos)
                    if distance_to_traffic_light < DISTANCE_THRESHOLD:
                        light = self.lights[light_inx]
                        stop_inx = self.get_closest_stop_line(light_pos)
                        stop_pos = self.get_stop_line_3d_positions()[stop_inx].pose.pose
                        stop_waypoint = self.get_closest_waypoint(stop_pos.position)

        # if we have line and stop line pos then change state and return waypoint
        if light and stop_waypoint:
            # todo : we should search in light area
            state = self.get_light_state(light)
            # rospy.logerr('Detected traffic light', light)
            return stop_waypoint, state

        return -1, (TrafficLight.UNKNOWN, 1)

    def get_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
