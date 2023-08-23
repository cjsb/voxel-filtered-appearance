#pragma once

#ifndef __CONE_OF_DIRECTIONS_H_
#define __CONE_OF_DIRECTIONS_H_

#include "helper_math.cuh"
#include "Eigen/Eigenvalues"
#include "common.h"
#include "color.h"
#include "ray.h"
#include "bbox.h"


struct ConeOfDirections {

	Vector3f direction;
	float thetaMax;
	bool valid;

	ConeOfDirections() {
		direction = Vector3f(0, 0, 0);
		thetaMax = 0;
		valid = false;
	}

	ConeOfDirections(Vector3f direction_, float thetaMax_, bool valid_) {

		direction = direction_;
		thetaMax = thetaMax_;
		valid = valid_;
	}


	void hughes_moeller(Vector3f& b1, Vector3f& b2, const Vector3f& n)
	{
		// Choose a vector orthogonal to n as the direction of b2.
		if (fabs(n.x()) > fabs(n.z())) b2 = Vector3f(-n.y(), n.x(), 0.0f);
		else b2 = Vector3f(0.0f, -n.z(), n.y());
		b2 *= rsqrtf(b2.dot(b2)); // Normalize b2
		b1 = b2.cross(n); // Construct b1 using a cross product
	}

	// code based on the book Ray Tracing Gems - Chapter 16 - directions in a cone
	Vector3f getSample(float random1, float random2) {

		float cosTheta = (1 - random1) + random1 * cos(thetaMax);
		float sinTheta = sqrt(1 - cosTheta * cosTheta);
		float phi = random2 * 2 * M_PI;
		float x = cos(phi) * sinTheta;
		float y = sin(phi) * sinTheta;
		float z = cosTheta;

		Vector3f tangent(0, 0, 0);
		Vector3f bitangent(0, 0, 0);

		hughes_moeller(tangent, bitangent, direction);

		/* Make our disk orient towards the normal. */
		Vector3f sampledDir = (tangent * x) + (bitangent * y) + (direction * z);

		return sampledDir.normalized();

	}

	float getHeightTriangle() {
		return cos(thetaMax);
	}

	float getRadiusCircle() {
		return sin(thetaMax);
	}


};


#endif
