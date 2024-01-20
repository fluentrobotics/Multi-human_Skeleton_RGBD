from rospy.numpy_msg import numpy_msg
import numpy
import rospy
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Header, ColorRGBA, MultiArrayDimension

pub = rospy.Publisher('mytopic', numpy_msg(TopicTInt32MultiArrayype))
rospy.init_node('mynode')
a = numpy.array([1.0, 2.1, 3.2, 4.3, 5.4, 6.5], dtype=numpy.float32)
pub.publish(a)