#pragma once

#pragma once

#ifndef __SGGX_H_
#define __SGGX_H_

#include "helper_math.cuh"
#include "Eigen/Eigenvalues"
#include "common.h"

#define M_PI	3.14159265358979323846f

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

using namespace nori;

class SGGX {

public:

	SGGX() {
		S = Eigen::Matrix3f::Zero();
		R = Eigen::Matrix3f::Zero();
		T = Eigen::Matrix3f::Zero();
	}

	SGGX(Eigen::Matrix3f S_) {
		S = S_;
	}

	SGGX(float sigma1, float sigma2, float sigma3) {

		Eigen::Matrix<float, 3, 3> MIdent;
		MIdent.setIdentity(3, 3);

		Eigen::Matrix<float, 3, 3> S_eigen;
		S_eigen << sigma1 * sigma1, 0.0f, 0.0f,
			0.0f, sigma2*sigma2, 0.0f,
			0.0f, 0.0f, sigma3*sigma3;

		S = MIdent * S_eigen * MIdent.transpose();
	}

	// based on the section "Parameter Estimation from Arbitrary Distributions" on the paper and the supplemental material.
	void estimate(Normal3f *normals, int nbNormals) {


		if (nbNormals == 0) {
			return;
		}

		Eigen::Matrix<float, 3, 3> E = Eigen::Matrix3f::Zero();
		for (int i = 0; i < nbNormals; i++) {

			Vector3f normal(normals[i].x(), normals[i].y(), normals[i].z());

			float D_ = 1.0f;

			E(0, 0) += normal.x()*normal.x()*D_;
			E(1, 0) += normal.x()*normal.y()*D_;
			E(2, 0) += normal.x()*normal.z()*D_;

			E(0, 1) += normal.x()*normal.y()*D_;
			E(1, 1) += normal.y()*normal.y()*D_;
			E(2, 1) += normal.y()*normal.z()*D_;

			E(0, 2) += normal.x()*normal.z()*D_;
			E(1, 2) += normal.y()*normal.z()*D_;
			E(2, 2) += normal.z()*normal.z()*D_;
		}

		E(0, 0) /= nbNormals;
		E(1, 0) /= nbNormals;
		E(2, 0) /= nbNormals;

		E(0, 1) /= nbNormals;
		E(1, 1) /= nbNormals;
		E(2, 1) /= nbNormals;

		E(0, 2) /= nbNormals;
		E(1, 2) /= nbNormals;
		E(2, 2) /= nbNormals;


		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver;
		eigenSolver.compute(E);
		Eigen::Matrix3f eigenVectorsMatrix = eigenSolver.eigenvectors();
	
		Eigen::Vector3f lambda(0, 0, 0);

		Vector3f vec1 = eigenSolver.eigenvectors().col(0);
		Vector3f vec2 = eigenSolver.eigenvectors().col(1);
		Vector3f vec3 = eigenSolver.eigenvectors().col(2);

		//saving rotation matrix
		R = eigenSolver.eigenvectors();

		for (int i = 0; i < nbNormals; i++) {

			Vector3f normal(normals[i].x(), normals[i].y(), normals[i].z());

			float D_ = 1.0f;

			lambda.x() += fabs(vec1.dot(normal))*D_;
			lambda.y() += fabs(vec2.dot(normal))*D_;
			lambda.z() += fabs(vec3.dot(normal))*D_;


		}

		lambda /= nbNormals;

		lambda.x() = MAX(lambda.x(), 0.1f);
		lambda.y() = MAX(lambda.y(), 0.1f);
		lambda.z() = MAX(lambda.z(), 0.1f);

		lambda.x() *= lambda.x();
		lambda.y() *= lambda.y();
		lambda.z() *= lambda.z();

		Eigen::Matrix3f lambdaMatrix = lambda.asDiagonal();

		Eigen::Vector3f T_diag(0, 0, 0);
		T_diag.x() = pow((lambda.y()*lambda.z()) / lambda.x(), 1.0f / 4.0f);
		T_diag.y() = pow((lambda.x()*lambda.z()) / lambda.y(), 1.0f / 4.0f);
		T_diag.z() = pow((lambda.x()*lambda.y()) / lambda.z(), 1.0f / 4.0f);
		T_diag *= 1.0f / sqrt(M_PI);

		T = T_diag.asDiagonal();

		S = eigenVectorsMatrix * lambdaMatrix* eigenVectorsMatrix.transpose();


	}


	Eigen::Matrix3f S;
	Eigen::Matrix3f R;
	Eigen::Matrix3f T;


};

#endif
