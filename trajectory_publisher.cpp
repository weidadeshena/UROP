#include "ros/ros.h"
#include "Dimos_package/FullStateTrajectory.h"
#include "std_msgs/UInt64"
#include "geometry_msgs/Vector3.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

 
std::vector<float> convertStringVectortoFloatVector(const std::vector<std::string>& stringVector)
{
	std::vector<float> floatVector(stringVector.size());
	// convert vector of string to float
	std::transform(stringVector.begin(), stringVector.end(), floatVector.begin(), [](const std::string& val)
                 {
                     return stof(val);
                 });
	return floatVector;
}


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

std::vector<Dimos_package::FullStateTrajectory> calculatesAllStates() 
{
	std::vector<Dimos_package::FullStateTrajectory> retVal;

	std::vector<std::string> vecOfStr;
 
	// Get the contents of file in a vector
	bool result = getFileContent("trajectory.txt", vecOfStr);
 
	if(result)
	{
		std::string delimiter = ",";
		// get each line
		for(std::string & line : vecOfStr){
			Dimos_package::FullStateTrajectory trajectory_point;
			size_t pos = 0;
			double data_array[10];
			int i = 0;
			while ((pos = line.find(delimiter)) != std::string::npos)
			{
				data = line.substr(0,pos);
				data_array[i] = data;
				++i;
			}
			geometry_msgs::Vector3 position;
			geometry_msgs::Vector3 linearVelocity;
			geometry_msgs::Vector3 linearAcceleration;
			std_msgs::UInt64 timestampNanoseconds;
			position.x = data_array[0];
			position.y = data_array[1];
			position.z = data_array[2];
			linearVelocity.x = data_array[3];
			linearVelocity.y = data_array[4];
			linearVelocity.z = data_array[5];
			linearAcceleration.x = data_array[6];
			linearAcceleration.y = data_array[7];
			linearAcceleration.z = data_array[8];
			timestampNanoseconds.data = data_array[9];
			trajectory_point.position = position;
			trajectory_point.linearVelocity = linearVelocity;
			trajectory_point.linearAcceleration = linearAcceleration;
			trajectory_point.timestampNanoseconds = timestampNanoseconds;
			retVal.push_back(trajectory_point);
		}
	}
	return retVal;
}


int main(int argc, char** argv) {
	
	std::string fileName = "trajectory.txt"
	// init
	ros::init(argc, argv, fileName);
	ros::NodeHandle nh;

	// setup ...
	ros::Publisher pub = nh.advertise<Dimos_package::FullStateTrajectory>("pen_trajectory", 10);

	// do shit
	std::vector<Dimos_package::FullStateStamped> vector_of_states = calculatesAllStates();
	
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