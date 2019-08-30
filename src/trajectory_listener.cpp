#include "ros/ros.h"
#include "trajectory_publisher/FullStateTrajectory.h"

void pen_trajectory_callback(const trajectory_publisher::FullStateTrajectory::ConstPtr &msg)
{
	ROS_INFO_STREAM(*msg);
}

int main(int argc, char **argv)
{
	ros::init(argc,argv,"listener");
	ros::NodeHandle n;
	ros::Subscriber sub = n.subscribe("/pen_trajectory",10,pen_trajectory_callback);
	ros::spin();
	return 0;
}