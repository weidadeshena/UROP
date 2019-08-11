#include <iostream>
#include <Eigen/Dense>
#include <cmath>


using namespace Eigen;


class whiteBoardEstimator
{
	public:
		whiteBoardEstimator() 
		{
			w_x << 1,0,0;
			w_y << 0,1,0;
			w_z << 0,0,1;
			origin << 0,0,0;
			sigma_r = 0.01;
			sigma_q_r = 0.001;
			sigma_angle = 0.001;
			dt = 0.05;
			N = 100;
			Matrix<double,5,1> Qvector;
			Qvector << pow(sigma_q_r,2),pow(sigma_q_r,2),pow(sigma_q_r,2),pow(sigma_angle,2),pow(sigma_angle,2);
			Q = dt*Qvector.asDiagonal();
			Matrix<double,5,1> Pvector;
			Pvector << pow(sigma_r,2),pow(sigma_r,2),pow(sigma_r,2),pow(sigma_r,2),pow(sigma_r,2);
			P = Pvector.asDiagonal();
		};
		//~whiteBoardEstimator();
		float get_sigma_r() const {return sigma_r;};
		Matrix<double,5,5> get_Q() const {return Q;};
		Matrix<double,5,1> get_x() const {return x;};

		// initialise the ground truth
		void initialisation();

		// calculate the rotation matrix given angle phi and psi
		// param in: phi, psi
		// param out: C_WT
		void getBoardRotationMatrix(const float phi, const float psi,Matrix3d &C_WT);

		// kalman filter to estimate the plane
		void EKF(Matrix<double,5,1> x);
	private:
		// Ground truth
		// x, y, z axis in world frame
		Vector3d w_x,w_y,w_z,origin;

		// variables for covariance matrix
		float sigma_r,sigma_q_r,sigma_angle;
		Matrix<double,5,5> P,Q;
		float dt;
		int N;

		// state variable x
		Matrix<double,5,1> x;
		
		// rotation matrix from world to plane frame
		Matrix3d C_WT;  
		
		// x, y, z axis on plane frame and position of plane in world frame
		Vector3d w_p_x,w_p_y,w_p_z,w_r,w_r_true;

		// euler angle of the plane
		float phi,psi,phi_true,psi_true;
};





