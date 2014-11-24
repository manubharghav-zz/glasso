/*
 * lasso.h
 *
 *  Created on: Nov 23, 2014
 *      Author: manu
 */

#ifndef LASSO_H_
#define LASSO_H_

#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;

void softThreshold(VectorXd* beta, double eta);
VectorXd lasso(MatrixXd X, VectorXd Y, double lambda);

#endif /* LASSO_H_ */
