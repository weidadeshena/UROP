// EKF for white board estimation
#include "white_board_estimator.h"
// for generating random number

#include <ctime>
#include <Eigen/Cholesky>
#include <iostream>
#include <stdio.h>



void whiteBoardEstimator::getBoardRotationMatrix(const double phi_, const double psi_, 
												Eigen::Matrix3d &C_WT_)
{
	Eigen::Matrix3d C_z,C_x;
	C_z <<  cos(psi_), sin(psi_), 0,
		    -sin(psi_), cos(psi_), 0,
		    0, 0, 1;
	C_x <<  1, 0, 0,
			0, cos(phi_), sin(phi_),
			0, -sin(phi_), cos(phi_);
	C_WT_ = C_z * C_x;
}

Eigen::Vector3d whiteBoardEstimator::generateMeasurement()
{
	Eigen::Vector3d Measurement;
	double x,z;
	x = static_cast<double>(rand())/static_cast<double>(RAND_MAX/2);
	z = static_cast<double>(rand())/static_cast<double>(RAND_MAX/1);
	Eigen::Vector3d randXZVect(x,0.0,z);
	Eigen::Matrix3d L = R_.llt().matrixL();
	Eigen::Vector3d randVec = (Eigen::Vector3d::Random().array()+1)/2;
	Measurement = w_r_true_ + C_WT_true_ * randXZVect; //+ L * randVec;
	return Measurement;
}

void whiteBoardEstimator::EKF_initialisation()
{
	srand (static_cast <unsigned> (time(0)));
	double LO = -0.1, HI = 0.1;
	phi_true_ = LO + static_cast<double>(rand()) / static_cast <double> (RAND_MAX/(HI-LO));
	LO = -3.1415926; HI = 3.1415926;
	psi_true_ = LO + static_cast<double>(rand()) / static_cast <double> (RAND_MAX/(HI-LO));
	w_r_true_ = (Eigen::Vector3d::Random()+Eigen::Vector3d(1,1,1))*5.0;
	this->getBoardRotationMatrix(phi_true_,psi_true_,C_WT_true_);
	 x_true_ << w_r_true_(0), w_r_true_(1), w_r_true_(2), phi_true_, psi_true_;
	Eigen::Matrix<double,5,5> L = P_.llt().matrixL();
	Eigen::Matrix<double,5,1> randVec = (Eigen::Matrix<double,5,1>::Random().array()+1)/2;
	 x_ = x_true_ + L * randVec;
}

void whiteBoardEstimator::EKF_predict(Eigen::Matrix<double,5,5> &P_)
{
	P_ = P_ + Q_;
	
}

void whiteBoardEstimator::EKF_update(Eigen::Matrix<double,5,1> &x_, Eigen::Matrix<double,5,5> &P_)
{
	double phi_est = x_(3);
	double psi_est = x_(4);
	this->getBoardRotationMatrix(phi_est,psi_est,C_WT_);
	Eigen::Vector3d Measurement;
	Measurement = this->generateMeasurement();
	Eigen::Vector3d delta_r = Measurement - x_.head<3>();
	double H_phi = delta_r(2)*cos(phi_est) - delta_r(1)*cos(psi_est)*sin(phi_est)
									+ delta_r(0)*sin(phi_est)*sin(psi_est);
	double H_psi = -cos(phi_est)*(delta_r(0)*cos(psi_est) - delta_r(1)*sin(psi_est));
	Eigen::Vector3d H_r = w_y_.transpose() * C_WT_.transpose();
	Eigen::Matrix<double,5,1> H;
	H << H_r(0), H_r(1), H_r(2), H_phi, H_psi;
	// std::cout << "H:\n" << H << "\n";
	double y = w_y_.transpose() * (C_WT_.transpose() * delta_r);
	double S = H.transpose() * P_ * H + pow(sigma_r_,2) * 
				(pow(C_WT_(1,0),2) + pow(C_WT_(1,1),2) + pow(C_WT_(1,2),2));
	Eigen::Matrix<double,5,1> K = P_ * H /S;
	Eigen::Matrix<double,5,1> delta_x = K * y;
	// projection along normal direction
	Eigen::Matrix3d s = (C_WT_ * w_y_) * delta_x.head<3>().transpose();
	Eigen::Vector3d delta_x_temp = s * C_WT_ * w_y_;
	delta_x(0) = delta_x_temp(0); delta_x(1) = delta_x_temp(1); delta_x(2) = delta_x_temp(2);
  x_ = x_ + delta_x;
	P_ = P_ - K * H.transpose() * P_;
}

bool whiteBoardEstimator::perform_EKF()
{
	this->EKF_predict(P_);
	this->EKF_update(x_,P_);
	return true;
}

int main()
{
	srand(28);
  whiteBoardEstimator WBE;
  WBE.EKF_initialisation();

  for(int i=0; i<50; i++)
  {
  	WBE.perform_EKF();
  }
  Eigen::Matrix<double,5,1> x, x_true;
  x = WBE.get_x();
  x_true = WBE.get_x_true();

  std::cout << "x:" << x << "\n";
  std::cout << "x_true:" << x_true << std::endl;
}











