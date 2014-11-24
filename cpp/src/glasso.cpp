/*
 * glasso.cpp
 *
 *  Created on: Nov 23, 2014
 *      Author: manu
 */

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "lasso.h"
using namespace Eigen;
using namespace std;

MatrixXd glasso(MatrixXd X, double lambda) {
	int n = X.rows();
	int p = X.cols();
	MatrixXd mu = (X.colwise().sum())/n;

	for(int i=0;i<n;i++){
		X.row(i) = X.row(i) - mu;
	}
	// Emperical Covariance matrix
	MatrixXd S = (X.transpose()  * X) * (1/(double)n);

	MatrixXd W = S + (lambda * MatrixXd::Identity(S.rows(), S.cols()));
	MatrixXd  Wnew = MatrixXd::Zero(W.rows(),W.cols());
	MatrixXd Beta  = MatrixXd::Zero(p,p);
	MatrixXd  Theta = MatrixXd::Zero(p,p);
	int maxiter = 4000;
	int minIter = 10;
	double epsilon = pow(10,-5);
	for(int iter=0;iter<maxiter ;iter++){
		for (int a = 0; a < p; a++) {

			MatrixXd W11(p - 1, p - 1);
			MatrixXd W12(p - 1, 1);
			MatrixXd W22(1, 1);

			W11.topLeftCorner(a, a) = W.topLeftCorner(a, a);
			W11.topRightCorner(a, p - a - 1) = W.topRightCorner(a, p - a - 1);
			W11.bottomLeftCorner(p - a - 1, a) = W.bottomLeftCorner(p - a - 1,
					a);
			W11.bottomRightCorner(p - a - 1, p - a - 1) = W.bottomRightCorner(
					p - a - 1, p - a - 1);
			W12.topLeftCorner(a, 1) = W.block(0, a, a, 1);
			W12.bottomRightCorner(p - a - 1, 1) = W.block(a + 1, a, p - a - 1,
					1);
			W22(0, 0) = W(a, a);

			MatrixXd S11(p - 1, p - 1);
			MatrixXd S12(p - 1, 1);
			MatrixXd S22(1, 1);


			S11.topLeftCorner(a, a) = S.topLeftCorner(a, a);
			S11.topRightCorner(a, p - a - 1) = S.topRightCorner(a, p - a - 1);
			S11.bottomLeftCorner(p - a - 1, a) = S.bottomLeftCorner(p - a - 1,
					a);
			S11.bottomRightCorner(p - a - 1, p - a - 1) = S.bottomRightCorner(
					p - a - 1, p - a - 1);
			S12.topLeftCorner(a, 1) = S.block(0, a, a, 1);
			S12.bottomRightCorner(p - a - 1, 1) = S.block(a + 1, a, p - a - 1,
					1);
			S22(0, 0) = S(a, a);
			EigenSolver<MatrixXd> es(W11);
			MatrixXd D = es.pseudoEigenvalueMatrix();
			MatrixXd V = es.pseudoEigenvectors();

			MatrixXd W11_sqrt = V * (D.cwiseSqrt()) * V.inverse();
			MatrixXd W11_sqrt_Inv = W11_sqrt.inverse();

			MatrixXd b = W11_sqrt_Inv * S12;
			VectorXd beta = lasso(W11_sqrt,b,lambda);

			W12 = W11*beta;
			MatrixXd W12New(p,1);
			W12New.topLeftCorner(a,1) = W12.topLeftCorner(a,1);
			W12New(a) = W22(0,0);
			W12New.bottomLeftCorner(p-a-1,1) = W12.bottomLeftCorner(p-a-1,1);
			Wnew.col(a) = W12New;
			Wnew.row(a) = W12New.transpose();
			MatrixXd BetaNew(p,1);
			BetaNew.topLeftCorner(a, 1) = beta.topLeftCorner(a, 1);
			BetaNew(a)=0;
			BetaNew.bottomLeftCorner(p-a-1,1) = beta.bottomLeftCorner(p-a-1,1);
//			Computer rows of Wned and Beta.

		}
		if(iter > minIter && abs((Wnew-W).sum()) < epsilon){

			break;
		}
		W=Wnew;

	}

	return W;
}

MatrixXd loadData(string fileName) {
	cout << "loading data from fileName" << fileName <<  endl;
	ifstream fin(fileName.c_str());
	int n, p;
	fin >> n >> p;
	MatrixXd X(n,p);
	double d;
	for(int i=0;i<n;i++){
		for(int j=0;j<p;j++){
			fin >> X(i,j);
		}
	}
	return X;
}

int main(int argc, char **argv) {

	MatrixXd X = loadData("data.txt");

//	int n = X.rows();
//		int p = X.cols();
//		MatrixXd mu = (X.colwise().sum())/n;
//		cout << mu  << endl;
//		for(int i=0;i<n;i++){
//				X.row(i) = X.row(i) - mu;
//		}
//	cout << X << endl;
//	MatrixXd  p = X.block(2,2,2,2);
//	p = p*2;
//	cout << p << endl;
//
//	MatrixXd Y(4,4);
//	Y.setZero(4,4);
//	Y.topRightCorner(3, 1)  = X.topLeftCorner(3, 1) ;
//	cout << Y << endl;
//	cout << X.rows() << "  " << X.cols() << endl;
	MatrixXd W = glasso(X,0.1);
	cout  << W << endl;




}

