/*
 * lasso.cpp
 *
 *  Created on: Nov 23, 2014
 *      Author: manu
 */

#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;


void softThreshold(VectorXd* beta, double eta){
	for (int i=0;i<beta->size() ;i++){
		if(beta->operator()(i) < -1 *eta){
			beta->operator()(i)+=eta;
		}
		else if (beta->operator()(i) > eta){
			beta->operator()(i)-=eta;
		}
		else{
			beta->operator()(i) = 0;
		}
	}
}


VectorXd lasso(MatrixXd X, VectorXd Y, double lambda){

	int minIter = 10;
	int maxIter = 4000;
	double epsilon = pow(10,-5);

	//step size = t;
	double t = 0.0001;

	int size = X.cols();
	VectorXd beta = VectorXd::Zero(size);
	MatrixXd Grad = MatrixXd::Zero(maxIter, size);
	double obj = 0.0;
	for(int iter=0; iter<maxIter ; iter++){
		MatrixXd grad = X.transpose() * ((X*beta) - Y);
		VectorXd beta_new = beta - t*grad;
		softThreshold(&beta_new, t*lambda);
		double obj_new = ((Y-(X*beta_new)).squaredNorm())/2  - (lambda * (beta_new.cwiseAbs().sum()));
		if(iter > minIter  && (abs(obj_new - obj) < epsilon)){
			break;
		}
		beta = beta_new;
		obj = obj_new;
	}
	return beta;
}



//int main()
//{
////	MatrixXd X(5,5);
////	X << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25;
////	cout << X << endl;
////	VectorXd Y(5);
////	Y<< 1,-1,1,1,-1;
////	cout << Y << endl;
////
////	int size = X.cols();
////	VectorXd beta = VectorXd::Zero(size);
////	MatrixXd Grad = MatrixXd::Zero(100, size);
////	double obj = 0.0;
////
////	cout << beta << endl;
////	MatrixXd grad = X.transpose() * ((X*beta) - Y);
////
////	cout <<"grad" << endl<< grad << endl;
////	double t= 0.001;
////	VectorXd beta_new = beta - t*grad;
////	double lambda = 0.01;
////	cout << "beta_new" << endl<< beta_new << endl;
////	softThreshold(&beta_new, t*lambda);
////
////	cout << "beta_new" << endl<< beta_new << endl;
////
////
////
////	cout << "objective" << endl << obj_new << endl;
//
//
//}





