#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "ros/ros.h"
#include "trajectory_publisher/FullStateTrajectory.h"
#include "geometry_msgs/Vector3.h"
#include "std_msgs/UInt64.h"

bool getFileContent(std::string &fileName, std::vector<std::string>& vecOfStrs) {
  // Open the File
  std::ifstream in(fileName.c_str());
  // Check if object is valid
  if (!in) {
    std::cerr << "Cannot open the File : " << fileName << std::endl;
    return false;
  }
  std::string str;
  // Read the next line from File untill it reaches the end.
  while (std::getline(in, str)) {
    // Line contains string of length > 0 then save it in vector
    if (str.size() > 0) vecOfStrs.push_back(str);
  }
  // Close The File
  in.close();
  return true;
}

bool parseLine(const double *data_array, geometry_msgs::Vector3 &position, 
                geometry_msgs::Vector3 &linearVelocity, 
                geometry_msgs::Vector3 &linearAcceleration, 
                std_msgs::UInt64 &timestampNanoseconds) 
{
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
  return true;
}

std::vector<trajectory_publisher::FullStateStamped> calculatesAllStates() {
  std::vector<trajectory_publisher::FullStateStamped> retVal;

  std::vector<std::string> vecOfStr;
  std::string fileName = "/home/cheryl/UROP/trajectory.txt";

  // Get the contents of file in a vector
  bool result = getFileContent(fileName, vecOfStr);

  if (result) {
    std::string delimiter = ",";
    // get each line
    for (std::string& line : vecOfStr) {
      size_t pos = 0;
      double data_array[10];
      int i = 0;
      std::string data_string;
      double data;
      while ( (pos = line.find(delimiter)) != std::string::npos) {
        data_string = line.substr(0, pos);
        data = std::stod(data_string);
        data_array[i] = data;
        line.erase(0, pos + delimiter.length());
        ++i;
      }
      data_string = line;
      data = std::stod(data_string);
      data_array[9] = data;

      trajectory_publisher::FullStateStamped trajectory_point;
      geometry_msgs::Vector3 position;
      geometry_msgs::Vector3 linearVelocity;
      geometry_msgs::Vector3 linearAcceleration;
      std_msgs::UInt64 timestampNanoseconds;
      parseLine(data_array,position,linearVelocity,
            linearAcceleration,timestampNanoseconds);
      trajectory_point.position = position;
      trajectory_point.linearVelocity = linearVelocity;
      trajectory_point.linearAcceleration = linearAcceleration;
      trajectory_point.timestampNanoSeconds = timestampNanoseconds;
      retVal.push_back(trajectory_point);
    }
  return retVal;
  }
}

int main(int argc, char** argv) {
  // init
  ros::init(argc, argv, "trajectory_publisher");
  ros::NodeHandle nh;

  // setup ...
  ros::Publisher pub =
      nh.advertise<trajectory_publisher::FullStateTrajectory>("/pen_trajectory", 0.01);

  // fill msg
  trajectory_publisher::FullStateTrajectory trajectory_msg;
  trajectory_msg.header.stamp = ros::Time::now();
  trajectory_msg.header.frame_id = "dimos_frame";
  trajectory_msg.fullStateTrajectory = calculatesAllStates();
  // for(auto point: trajectory_msg.fullStateTrajectory){
  //   std::cout << point.position << "\n" << point.linearVelocity << "\n";
  //   std::cout << point.linearAcceleration << "\n" <<point.timestampNanoSeconds <<"\n\n";
  // }

  ros::Rate poll_rate(1);
  while(pub.getNumSubscribers() == 0)
  {
    ROS_INFO("Waiting for subscriber...");
    poll_rate.sleep();
  }
  ROS_INFO("Found subscriber, publishing...");

  // publish
  pub.publish(trajectory_msg);
  ros::spinOnce();
  sleep(2);
  

  return 0;
}
