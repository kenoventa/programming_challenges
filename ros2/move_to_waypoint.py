#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration

class WaypointNavigator(Node):
    def __init__(self):
        super().__init__('move_to_waypoint')
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.waypoints = self.create_waypoints()
        self.current_waypoint = 0
        
    def create_waypoints(self):
        """Create list of waypoints"""
        waypoints = []
        
        # Waypoint 1
        wp1 = PoseStamped()
        wp1.header.frame_id = 'map'
        wp1.pose.position.x = 0.4310684877304662
        wp1.pose.position.y = -0.8853890960319246
        wp1.pose.position.z = 0.0
        wp1.pose.orientation.x = 0.0
        wp1.pose.orientation.y = 0.0
        wp1.pose.orientation.z = -0.1560336286855684
        wp1.pose.orientation.w = 0.9877517434655401
        waypoints.append(wp1)
        
        # Waypoint 2
        wp2 = PoseStamped()
        wp2.header.frame_id = 'map'
        wp2.pose.position.x = 0.5841214163246903
        wp2.pose.position.y = 1.9831914977569194
        wp2.pose.position.z = 0.0
        wp2.pose.orientation.x = 0.0
        wp2.pose.orientation.y = 0.0
        wp2.pose.orientation.z = 0.6748816811047064
        wp2.pose.orientation.w = 0.7379259559801955
        waypoints.append(wp2)
        
        # Waypoint 3
        wp3 = PoseStamped()
        wp3.header.frame_id = 'map'
        wp3.pose.position.x = -1.6388831475364167
        wp3.pose.position.y = 1.2260668428735444
        wp3.pose.position.z = 0.0
        wp3.pose.orientation.x = 0.0
        wp3.pose.orientation.y = 0.0
        wp3.pose.orientation.z = -0.41341112322856194
        wp3.pose.orientation.w = 0.9105444762288654
        waypoints.append(wp3)
        
        return waypoints
    
    def go_to_waypoint(self):
        """Send a waypoint to the navigation system"""
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("Waiting for navigation server to come up...")
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self.waypoints[self.current_waypoint]
        goal_msg.behavior_tree = ''
        
        self.get_logger().info(f"Navigating to waypoint {self.current_waypoint + 1}...")
        send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)
        send_goal_future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        """Handle the response from the navigation action server"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return
        
        self.get_logger().info('Goal accepted')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future):
        """Handle the result from the navigation action server"""
        result = future.result().result
        self.get_logger().info(f'Navigation result: {result}')
        
        # Move to next waypoint
        self.current_waypoint += 1
        if self.current_waypoint < len(self.waypoints):
            self.go_to_waypoint()
        else:
            self.get_logger().info('All waypoints completed!')
            rclpy.shutdown()
    
    def feedback_callback(self, feedback_msg):
        """Handle feedback from the navigation action server"""
        feedback = feedback_msg.feedback
        # self.get_logger().info(
        #     f'Distance remaining: {feedback.distance_remaining:.2f} meters')

def main(args=None):
    rclpy.init(args=args)
    
    navigator = WaypointNavigator()
    navigator.go_to_waypoint()
    
    rclpy.spin(navigator)

if __name__ == '__main__':
    main()