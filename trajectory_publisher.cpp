# include "ros/ros.h"
# include "Dimos_package/FullStateTrajectory.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

std::vector<Dimos_package::FullStateTrajectory> my_function_that_calculates_all_the_states() {
	std::vector<Dimos_package::FullStateTrajectory> ret_val;

	for line in file:
		Dimos_package::FullStateTrajectory data_point;
		// open file and get a data point
		data_point.position = ...
		ret_val.push_back(data_posint);
	return ret_val;
}
 
/*
 * It will iterate through all the lines in file and
 * put them in given vector
 */
bool getFileContent(std::string fileName, std::vector<std::string> & vecOfStrs)
{
 
	// Open the File
	std::ifstream in(fileName.c_str());
 
	// Check if object is valid
	if(!in)
	{
		std::cerr << "Cannot open the File : "<<fileName<<std::endl;
		return false;
	}
 
	std::string str;
	// Read the next line from File untill it reaches the end.
	while (std::getline(in, str))
	{
		// Line contains string of length > 0 then save it in vector
		if(str.size() > 0)
			vecOfStrs.push_back(str);
	}
	//Close The File
	in.close();
	return true;
}

std::vector<Dimos_package::FullStateTrajectory> calculates_all_the_states() {
	std::vector<Dimos_package::FullStateTrajectory> ret_val;

	std::vector<std::string> vecOfStr;
 
	// Get the contents of file in a vector
	bool result = getFileContent("trajectory.txt", vecOfStr);
 
	if(result)
	{
		// Print the vector contents
		for(std::string & line : vecOfStr)
			std::cout<<line<<std::endl;
	}
	for line in file:
		Dimos_package::FullStateTrajectory data_point;
		// open file and get a data point
		data_point.position = ...
		ret_val.push_back(data_posint);
	return ret_val;
}


int main(int argc, char** argv) {

	

	ros::init(argc, argv, "trajectory_publisher");

	ros::NodeHandle nh;
	// get parameters if you have any
	// int example_param;
	// if(!nh.getParam("example_param", example_param)) {
	// 	ROS_ERROR("Fail to get param: example_param");
	// }

	// setup ...

	auto pub = nh.advertise<Dimos_package::FullStateTrajectory>("pen_trajectory", 10);

	// do shit
	std::vector<Dimos_package::FullStateStamped> vector_of_states = my_function_that_calculates_all_the_states();
	
	// fill msg
	Dimos_package::FullStateTrajectory trajectory_msg;

	trajectory_msg.header.stamp = ros::Time::now();
	trajectory_msg.header.frame_id = "dimos_frame";
	trajectory_msg.fullStateTrajectory = vector_of_states;

	// publish
	pub.publish(trajectory_msg);
	
	ros::spinOnce();

	return 0;
}