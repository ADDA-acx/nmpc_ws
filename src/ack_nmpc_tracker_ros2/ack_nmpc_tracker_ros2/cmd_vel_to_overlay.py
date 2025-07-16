#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32

class CmdVelToOverlay(Node):
    def __init__(self):
        super().__init__('cmdvel_to_overlay')
        self.sub = self.create_subscription(
            Twist, '/cmd_vel', self.cb_cmd, 10)
        self.pub_v = self.create_publisher(Float32, '/hud/linear_vel', 10)
        self.pub_d = self.create_publisher(Float32, '/hud/steer_angle', 10)

    def cb_cmd(self, msg: Twist):
        v = Float32(); v.data = msg.linear.x          # m/s
        d = Float32(); d.data = msg.angular.z         # rad (=δ)
        self.pub_v.publish(v)
        self.pub_d.publish(d)

def main():
    rclpy.init()
    rclpy.spin(CmdVelToOverlay())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
