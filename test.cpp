
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


bool getFileContent(std::string fileName, std::vector<std::string>& vecOfStrs) {
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
    if (str.size() > 0) 
      vecOfStrs.push_back(str);
  }
  // Close The File
  in.close();
  return true;
}

int main(){
  std::vector<std::string> vecOfStr;
  bool result = getFileContent("/home/cheryl/catkin_ws/src/trajectory_publisher/src/trajectory.txt", vecOfStr);

  if (result) {
    std::string delimiter = ",";
    // get each line
    for (std::string& line : vecOfStr) {
      size_t pos = 0;
      // std::vector<double> data_array;
      double data_array[10];
      int i = 0;
      std::string data_string;
      double data;
      while ( (pos = line.find(delimiter)) != std::string::npos) {
        data_string = line.substr(0, pos);
        line.erase(0, pos + delimiter.length());
        data = std::stod(data_string);
        data_array[i] = data;
        // data_array.push_back(data)
        ++i;
      }
      data_string = line;
      data = std::stod(data_string);
      data_array[9] = data;
      for(int i = 0; i < 10; i++) 
        std::cout << data_array[i] << "\n";
      // std::cout<<data_array.size()<<std::endl;
    }
  }

}
