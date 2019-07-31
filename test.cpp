#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <string>
#include <algorithm>

std::vector<float> convertStringVectortoFloatVector(const std::vector<std::string>& stringVector){
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

int main()
{
	std::vector<std::string> vecOfStr;
	bool result = getFileContent("trajectory.txt", vecOfStr);
 
	if(result)
	{
		// get each line
		std::cout<<typeid(vecOfStr).name() << "\n" << vecOfStr;
		// for(std::string & line : vecOfStr){
			// std::vector<float> lineF = convertStringVectortoFloatVector(line);
			// std::vector<float> position = std::vector<float>(lineF.begin(),lineF.begin()+3);
			// std::vector<float> linearVelocity = std::vector<float>(lineF.begin()+3,lineF.begin()+6);
			// std::vector<float> linearAcceleration = std::vector<float>(lineF.begin+6,lineF.begin()+9);
			// std::vector<float> timestampNanoSeconds = std::vector<float>(lineF.begin()+9,lineF.end());
			// std::cout << position << "\n" << linearVelocity << "\n";
			// std::cout << linearAcceleration << "\n";
			// std::cout << timestampNanoSeconds << "\n"
			// std::cout << lineF << std::endl;
			// std::cout << typeid(line).name() << '\n';
		// }
	}
}