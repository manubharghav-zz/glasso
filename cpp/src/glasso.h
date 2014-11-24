/*
 * glasso.h
 *
 *  Created on: Nov 23, 2014
 *      Author: manu
 */

#ifndef GLASSO_H_
#define GLASSO_H_

#include <iostream>
#include <Eigen/Dense>
#include "lasso.h"
using namespace Eigen;
using namespace std;


MatrixXd glasso(MatrixXd X, double lambda);

#endif /* GLASSO_H_ */
