// EKF for white board estimation
#include "white_board_estimator.h"
// for generating random number
#include <cstdlib>
#include <ctime>
#include <Eigen/Cholesky>

using namespace Eigen;
using namespace std;

void whiteBoardEstimator::getBoardRotationMatrix(const float phi, const float psi,Matrix3d &C_WT)
{
	Matrix3d C_z,C_x;
	C_z << cos(psi), sin(psi), 0,
				-sin(psi), cos(psi), 0,
					0, 0, 1;
	C_x << 1, 0, 0,
				0, cos(phi), sin(phi),
				0, -sin(phi), cos(phi);
	C_WT = C_z * C_x;
}

void whiteBoardEstimator::initialisation()
{
	srand (static_cast <unsigned> (time(0)));
	float LO = -0.1, HI = 0.1;
	phi_true = LO + static_cast <float> (rand()) / static_cast <float> (RAND_MAX/(HI-LO));
	LO = -3.1415926;
	HI = 3.1415926;
	psi_true = LO + static_cast <float> (rand()) / static_cast <float> (RAND_MAX/(HI-LO));
	w_r_true = (Vector3d::Random()+Vector3d(1,1,1))*5.0;
	x << w_r_true(0), w_r_true(1), w_r_true(2), phi_true, psi_true;
	Matrix<double,5,5> L = P.llt().matrixL();
	Matrix<double,5,1> randVec = Matrix<double,5,1>::Random();
	randVec = (randVec + Matrix<double,5,1>::Constant(1.0))/2.0;
	x = x + L * randVec;
}


int main()
{
  whiteBoardEstimator WBE;
  WBE.initialisation();
  // Matrix<double,5,1> PVec;
  // PVec << 0.01,0.01,0.01,0.01,0.01;
  // Matrix<double,5,5> P = PVec.asDiagonal();
  // Matrix<double,5,5> L = P.llt().matrixL();
  // std::cout<< L <<std::endl;
  std::cout << WBE.get_x() << std::endl;
}











