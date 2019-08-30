#ifndef WHITE_BOARD_ESTIMATOR_H_
#define WHITE_BOARD_ESTIMATOR_H_


#include <Eigen/Dense>
#include <cmath>




class whiteBoardEstimator
{
	public:
		whiteBoardEstimator() 
		{
			w_x_ << 1,0,0;
			w_y_ << 0,1,0;
			w_z_ << 0,0,1;
			origin_ << 0,0,0;
			sigma_r_ = 0.01;
			sigma_q_r_ = 0.001;
			sigma_angle_ = 0.001;
			dt_ = 0.05;
			N_ = 100;
			Eigen::Matrix<double,5,1> Qvector;
			Qvector << pow(sigma_q_r_,2),pow(sigma_q_r_,2),pow(sigma_q_r_,2),pow(sigma_angle_,2),pow(sigma_angle_,2);
			Q_ = dt_*Qvector.asDiagonal();
			Eigen::Matrix<double,5,1> Pvector;
			Pvector << 0.1,0.1,0.1,0.1,0.1;
			P_ = Pvector.asDiagonal();
			Eigen::Vector3d Rvector(pow(sigma_r_,2),pow(sigma_r_,2),pow(sigma_r_,2));
			R_ = Rvector.asDiagonal();
		};
		//~whiteBoardEstimator();
		Eigen::Matrix<double,5,5> get_P() const {return P_;};
		Eigen::Matrix<double,5,1> get_x() const {return x_;};
		Eigen::Matrix<double,5,1> get_x_true() const {return x_true_;};
		Eigen::Matrix3d get_C_WT() const {return C_WT_;};
		Eigen::Matrix3d get_C_WT_true() const {return C_WT_true_;};

		// calculate the rotation matrix given angle phi and psi
		// param in: phi, psi
		// param out: C_WT
		void getBoardRotationMatrix(const double phi_, const double psi_, Eigen::Matrix3d &C_WT_);

		// generate a measurement
		Eigen::Vector3d generateMeasurement();

		// kalman filter to estimate the plane
		// initialise the ground truth
		void EKF_initialisation();
		// predict step
		void EKF_predict(Eigen::Matrix<double,5,5> &P_);
		// update step
		void EKF_update(Eigen::Matrix<double,5,1> &x_, Eigen::Matrix<double,5,5> &P_);
		// perform EKF
		bool perform_EKF();


	private:
		// Ground truth
		// x, y, z axis in world frame
		Eigen::Vector3d w_x_,w_y_,w_z_,origin_;

		// variables for covariance matrix
		float sigma_r_,sigma_q_r_,sigma_angle_;
		Eigen::Matrix<double,5,5> P_,Q_;
		Eigen::Matrix3d R_;
		float dt_;
		int N_;

		// state variable x
		Eigen::Matrix<double,5,1> x_,x_true_;
		
		// rotation matrix from world to plane frame
		Eigen::Matrix3d C_WT_,C_WT_true_;  
		
		// x, y, z axis on plane frame and position of plane in world frame
		Eigen::Vector3d w_p_x_,w_p_y_,w_p_z_,w_r_,w_r_true_;

		// euler angle of the plane
		double phi_,psi_,phi_true_,psi_true_;
};





#endif  //WHITE_BOARD_ESTIMATOR_H_