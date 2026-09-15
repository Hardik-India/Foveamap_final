"""ROS2 optional bridge: PointCloud2 -> elevation PointCloud2 + cell JSON.
Install ROS2 and sensor_msgs_py through your ROS distribution; not pip.
"""
import json
import numpy as np
from .grid import build_grid

def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import String
    class Mapper(Node):
        def __init__(self):
            super().__init__('foveamap')
            self.declare_parameter('input_topic','/points_raw');self.declare_parameter('ground_offset',0.)
            self.create_subscription(PointCloud2,self.get_parameter('input_topic').value,self.scan,qos_profile_sensor_data)
            self.elevation=self.create_publisher(PointCloud2,'/foveamap/elevation',10)
            self.cells=self.create_publisher(String,'/foveamap/cells',10)
        def scan(self,msg):
            try:
                raw=list(point_cloud2.read_points(msg,field_names=('x','y','z'),skip_nans=True))
                p=np.array([[float(r[0]),float(r[1]),float(r[2])] for r in raw]).reshape(-1,3)
                p[:,2]-=self.get_parameter('ground_offset').value
                grid=build_grid(p)
                # Published geometry remains in the incoming sensor frame.
                offset=self.get_parameter('ground_offset').value
                cloud=point_cloud2.create_cloud_xyz32(msg.header,[(c['x'],c['y'],c['mean']+offset) for c in grid['cells']])
                self.elevation.publish(cloud)
                self.cells.publish(String(data=json.dumps({'frame_id':msg.header.frame_id,'stamp':{'sec':msg.header.stamp.sec,'nanosec':msg.header.stamp.nanosec},'ground_offset':offset,**grid})))
            except Exception as e:self.get_logger().error(str(e))
    rclpy.init();node=Mapper()
    try:rclpy.spin(node)
    finally:node.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
