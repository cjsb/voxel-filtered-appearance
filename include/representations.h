#pragma once

#ifndef __REPRESENTATIONS_H_
#define __REPRESENTATIONS_H_

#include "helper_math.cuh"
#include "Eigen/Eigenvalues"
#include "common.h"
#include "color.h"
#include "ray.h"
#include "bbox.h"
#include "coneDirections.h"

#include <curand.h>
#include <curand_kernel.h>

#define M_PI	3.14159265358979323846f
#define SIZEOCCLUSIONCENTERS 27
#define RESOLUTIONOCCGRID 4


struct VirtualMesh {

	uchar3 vectorsMesh[3]; // vectors that define the virtual mesh
	uchar1 vectorsMeshScales[3]; // scale/10

	// 1 color per triangle
	uchar3 triangleColors[8] = { make_uchar3(0,0,0), make_uchar3(0,0,0), make_uchar3(0,0,0), make_uchar3(0,0,0), make_uchar3(0,0,0), make_uchar3(0,0,0), make_uchar3(0,0,0), make_uchar3(0,0,0) };

	// 3 normals per triangle
	char3 triangleNormals[8][3] = { {make_char3(0,0,0), make_char3(0,0,0), make_char3(0,0,0)},
									 {make_char3(0,0,0), make_char3(0,0,0), make_char3(0,0,0)},
									 {make_char3(0,0,0), make_char3(0,0,0), make_char3(0,0,0)},
									 {make_char3(0,0,0), make_char3(0,0,0), make_char3(0,0,0)},
									 {make_char3(0,0,0), make_char3(0,0,0), make_char3(0,0,0)},
									 {make_char3(0,0,0), make_char3(0,0,0), make_char3(0,0,0)},
									 {make_char3(0,0,0), make_char3(0,0,0), make_char3(0,0,0)},
									 {make_char3(0,0,0), make_char3(0,0,0), make_char3(0,0,0)} };

	// is a distribution of normals or a small set (1 to 3)
	bool triangleIsDistribution[8];

	__host__ __device__
		VirtualMesh() {

	};

	__host__ __device__
	float3 encodeVector(float3 vec) //[-1, 1] -> [0,1]
	{
		return vec * 0.5f + make_float3(0.5f, 0.5f, 0.5f);
	}

	__host__ __device__
	float3 decodeVector(float3 vec) //[0,1] -> [-1,1]
	{
		return vec * 2.0f - make_float3(1.0f, 1.0f, 1.0f);
	}

	// compute the vertices position of the virtual mesh
	__host__ __device__
	void getVMeshVertices(float3 vertices_[6], float3 &center) {


		for (int vec = 0; vec < 3; vec++) {

			float3 vectorF = make_float3((float)vectorsMesh[vec].x / 255, (float)vectorsMesh[vec].y / 255, (float)vectorsMesh[vec].z / 255);
			vectorF = decodeVector(vectorF);

			float scale = ((float)vectorsMeshScales[vec].x / 255) * 10;

			vertices_[2 * vec] = center + (vectorF * scale);
			vertices_[2 * vec + 1] = center - (vectorF * scale);
		}
	}

	// is it a empty mesh?
	__host__ __device__
	bool hasAnyDataInIt() {

		
		for (int f = 0; f < 8; f++) {
			if (triangleNormals[f][0].x != 0 || triangleNormals[f][0].y != 0 || triangleNormals[f][0].z != 0) {
				return true;
			}

		}

		return false;
	}


	__host__ __device__
	float3 computeProjectionIntoPlane(float3 point, float3 normal, float3 origin) {

		float dotValue = dot((point - origin), (normal));
		float3 proj = point - dotValue * normal;

		return proj;
	}

	// compute the area of a triangle given the 3 vertices
	__host__ __device__
	float areaTriangle(float3 a, float3 b, float3 c) {

		float3 ab = b - a;
		float3 ac = c - a;

		float crossLenght = length(cross(ab, ac));

		return crossLenght / 2.0f;

	}

	// uniformly sample a triangle with vertices p0, p1, and p2 - Chp. 16 from the ray tracing gems book
	__host__ __device__
	float3 samplingTriangle(float3 p0, float3 p1, float3 p2, float u0, float u1) {

		float beta = 1 - sqrtf(u0);
		float gamma = (1 - beta)*u1;
		float alpha = 1 - beta - gamma;
		float3 P = alpha * p0 + beta * p1 + gamma * p2;

		return P;

	}

	// compute the ray - triangle intersection
	__host__ __device__
	bool rayTriangleIntersect(float3 V0, float3 V1, float3 V2, float3 origin, float3 dir, float b[3], float& t) {

		/* Find vectors for two edges sharing v[0] */
		float3 edge1 = V1 - V0;
		float3 edge2 = V2 - V0;

		/* Begin calculating determinant - also used to calculate U parameter */
		float3 pvec = cross(dir, edge2);

		/* If determinant is near zero, ray lies in plane of triangle */
		float det = dot(edge1, pvec);

		if (det > -1e-8f && det < 1e-8f)
			return false;
		float inv_det = 1.0f / det;

		/* Calculate distance from v[0] to ray origin */
		float3 tvec = origin - V0;

		/* Calculate U parameter and test bounds */
		float u = dot(tvec, pvec) * inv_det;
		if (u < 0.0 || u > 1.0)
			return false;

		/* Prepare to test V parameter */
		float3 qvec = cross(tvec, edge1);

		/* Calculate V parameter and test bounds */
		float v = dot(dir, qvec) * inv_det;
		if (v < 0.0 || u + v > 1.0)
			return false;

		/* Ray intersects triangle -> compute t */
		t = dot(edge2, qvec) * inv_det;


		b[0] = u;
		b[1] = v;
		b[2] = 1.0f - b[0] - b[1];

		return t >= 0;

	}

	// get samples from a triangle of the virtual mesh. It compute the ray-triangle intersection and used the uv coordinates to 
	__host__ __device__
	bool getSampleVMesh(int faceIndex, float3 vertA, float3 vertB, float3 vertC, float3 projA, float3 projB, float3 projC, curandState *randState, float3 dir, float b[3], float& t, int nbAttempts, float3 &normal, float3 triNormals0, float3 triNormals1, float3 triNormals2) {

		bool sampleFound = false;
		for (int i = 0; i < nbAttempts && !sampleFound; i++) {

			float random1 = clampf(curand_uniform(randState), 0.0f, 0.999f);
			float random2 = clampf(curand_uniform(randState), 0.0f, 0.999f);

			// sample the project triangle
			float3 sampledPoint = samplingTriangle(projA, projB, projC, random1, random2);
			t = 0;

			//compute the ray-triangle intersection
			bool hit = rayTriangleIntersect(vertA, vertB, vertC, sampledPoint, dir, b, t);
			if (hit) {

				//barycentric interpolation of the vertices' extremal normals
				normal = (b[0] * triNormals0) + (b[1] * triNormals1) + (b[2] * triNormals2);
				normal = normalize(normal);
				if (dot(normal, -dir) > 0) { //test for camera visibility
					sampleFound = true;
				}
			}


		}


		return sampleFound;
	}

	// diffuse phong shading
	__host__ __device__
	float3 phongShading(float3 position, float3 normal, float3 color, float3 lightPos, float3 viewDir) {

		float3 lightColor = make_float3(1, 1, 1);

		float3 lightDir = (lightPos - position);
		lightDir = normalize(lightDir);
		float NdotL = dot(normal, lightDir);
		float diff = fmaxf(NdotL, 0.0f);

		float3 diffuse = diff * lightColor;
		
		return (diffuse * color);

	}

	// compute shading using the virtual mesh
	// return the shaded color, shaded = the sampled normals are visible and it was used in the shadding, sampledAvgColor = the average of the sampled colors, in the paper we use to compute the ambient color.   
	__host__ __device__
	float3 shadeWithVirtualMesh(float3 rayOrigin, float3 rayDirection, float3 lightPos, float3 boundingBoxCenter, unsigned int *vMeshTriangles, int nbSamples, curandState *randState, bool &shaded, float3& sampledAvgColor) {

		float3 finalColor = make_float3(0, 0, 0);
		
		int nbSampledPoints = 0;
		
		float projArea[8] = { 0,0,0,0,0,0,0,0 };
		float3 shadingFaces[8] = { make_float3(0,0,0),make_float3(0,0,0),make_float3(0,0,0),make_float3(0,0,0),make_float3(0,0,0),make_float3(0,0,0),make_float3(0,0,0),make_float3(0,0,0) };
		float3 colorFaces[8] = { make_float3(0,0,0),make_float3(0,0,0),make_float3(0,0,0),make_float3(0,0,0),make_float3(0,0,0),make_float3(0,0,0),make_float3(0,0,0),make_float3(0,0,0) };
		bool isShaded[8] = { false,false,false,false,false,false,false,false };
		float projAreaSum = 0;
		int visibleFaceIndex[8] = { 0,0,0,0,0,0,0,0 }; 
		int nbVisibleFace = 0;

		float3 triangleNormalsF[8][3] = { {make_float3(0,0,0), make_float3(0,0,0), make_float3(0,0,0)},
									 {make_float3(0,0,0), make_float3(0,0,0), make_float3(0,0,0)},
									 {make_float3(0,0,0), make_float3(0,0,0), make_float3(0,0,0)},
									 {make_float3(0,0,0), make_float3(0,0,0), make_float3(0,0,0)},
									 {make_float3(0,0,0), make_float3(0,0,0), make_float3(0,0,0)},
									 {make_float3(0,0,0), make_float3(0,0,0), make_float3(0,0,0)},
									 {make_float3(0,0,0), make_float3(0,0,0), make_float3(0,0,0)},
									 {make_float3(0,0,0), make_float3(0,0,0), make_float3(0,0,0)} };

		float3 projVertices[8][3] = { {make_float3(0,0,0), make_float3(0,0,0), make_float3(0,0,0)},
									{make_float3(0,0,0), make_float3(0,0,0), make_float3(0,0,0)},
									{make_float3(0,0,0), make_float3(0,0,0), make_float3(0,0,0)},
									{make_float3(0,0,0), make_float3(0,0,0), make_float3(0,0,0)},
									{make_float3(0,0,0), make_float3(0,0,0), make_float3(0,0,0)},
									{make_float3(0,0,0), make_float3(0,0,0), make_float3(0,0,0)},
									{make_float3(0,0,0), make_float3(0,0,0), make_float3(0,0,0)},
									{make_float3(0,0,0), make_float3(0,0,0), make_float3(0,0,0)} };



		float3 vertices[6];
		getVMeshVertices(vertices, boundingBoxCenter);

		// test the triangles visibility. Store the project area, normals and project vertices if it is visible.
		for (int f = 0; f < 8; f++) {

			
			float3 triNormals[3];
			triNormals[0] = make_float3((float)triangleNormals[f][0].x / 127, (float)triangleNormals[f][0].y / 127, (float)triangleNormals[f][0].z / 127);
			triNormals[1] = make_float3((float)triangleNormals[f][1].x / 127, (float)triangleNormals[f][1].y / 127, (float)triangleNormals[f][1].z / 127);
			triNormals[2] = make_float3((float)triangleNormals[f][2].x / 127, (float)triangleNormals[f][2].y / 127, (float)triangleNormals[f][2].z / 127);

			bool triVis = false;
			for (int n = 0; n < 3; n++) {
				if (triangleNormals[f][0].x != 0 || triangleNormals[f][0].y != 0 || triangleNormals[f][0].z != 0) {
					triVis = triVis || (dot(triNormals[n], -rayDirection) > 0);
				}

			}

			if (triVis) { //visible

				unsigned int vertA = vMeshTriangles[f * 3 + 0];
				float3 projA = computeProjectionIntoPlane(vertices[vertA], rayDirection, rayOrigin);

				unsigned int vertB = vMeshTriangles[f * 3 + 1];
				float3 projB = computeProjectionIntoPlane(vertices[vertB], rayDirection, rayOrigin);

				unsigned int vertC = vMeshTriangles[f * 3 + 2];
				float3 projC = computeProjectionIntoPlane(vertices[vertC], rayDirection, rayOrigin);

				float areaTri = areaTriangle(projA, projB, projC);

				projArea[nbVisibleFace] = areaTri;
				visibleFaceIndex[nbVisibleFace] = f;

				triangleNormalsF[nbVisibleFace][0] = triNormals[0];
				triangleNormalsF[nbVisibleFace][1] = triNormals[1];
				triangleNormalsF[nbVisibleFace][2] = triNormals[2];

				projVertices[nbVisibleFace][0] = projA;
				projVertices[nbVisibleFace][1] = projB;
				projVertices[nbVisibleFace][2] = projC;


				nbVisibleFace++;

			}

		}

		// for each visible face, sample the normals and get the face color to compute the shading.
		for (int visibleFace = 0; visibleFace < nbVisibleFace; visibleFace++) {

			float3 faceColor = make_float3(0, 0, 0);
			float3 faceAvgColorObj = make_float3(0, 0, 0);
			
			int nbSampledPointsFace = 0;

			int f = visibleFaceIndex[visibleFace];

			int nbSamplesTri = 5;

			unsigned int vertA = vMeshTriangles[f * 3 + 0];
			float3 projA = projVertices[visibleFace][0];

			unsigned int vertB = vMeshTriangles[f * 3 + 1];
			float3 projB = projVertices[visibleFace][1];

			unsigned int vertC = vMeshTriangles[f * 3 + 2];
			float3 projC = projVertices[visibleFace][2];

			if (triangleIsDistribution[f]) {
				for (int i = 0; i < nbSamples; i++) {
					
					float bary[3];
					float t = 0;
					float3 normal = make_float3(0, 0, 0);

					bool hit = getSampleVMesh( f, vertices[vertA], vertices[vertB], vertices[vertC], projA, projB, projC, randState, rayDirection, bary, t, 5, normal, triangleNormalsF[visibleFace][0], triangleNormalsF[visibleFace][1], triangleNormalsF[visibleFace][2]);

					if (hit) {

						float3 color = make_float3((float)triangleColors[f].x / 255, (float)triangleColors[f].y / 255, (float)triangleColors[f].z / 255);
						faceAvgColorObj += color;
						
						faceColor += phongShading(boundingBoxCenter, normal, color, lightPos, rayDirection);
						
						nbSampledPointsFace++;
						nbSampledPoints++;

					}

				}


			}
			else {

				for (int s = 0; s < 3; s++) {

					float3 normal = make_float3((float)triangleNormals[f][s].x / 127, (float)triangleNormals[f][s].y / 127, (float)triangleNormals[f][s].z / 127);

					if (triangleNormals[f][s].x != 0 || triangleNormals[f][s].y != 0 || triangleNormals[f][s].z != 0) { // there is a normal.
						
						if (dot(normal, -rayDirection) > 0.0) { //test for camera visibility.
							
							float3 color = make_float3((float)triangleColors[f].x / 255, (float)triangleColors[f].y / 255, (float)triangleColors[f].z / 255);
							faceAvgColorObj += color;
							
							faceColor += phongShading(boundingBoxCenter, normal, color, lightPos, rayDirection);
						
							nbSampledPointsFace++;
							nbSampledPoints++;

						}

					}



				}


			}



			if (nbSampledPointsFace > 0) {

				faceAvgColorObj /= nbSampledPointsFace;
				faceColor /= nbSampledPointsFace;

				shadingFaces[visibleFace] = faceColor;
				
				colorFaces[visibleFace] = faceAvgColorObj;
				isShaded[visibleFace] = true;
				projAreaSum += projArea[visibleFace];

			}

		}

		// the face contribution to the final color is the shading color multiply by the triangle projected area over the total projected area of the visible faces.
		for (int visibleFace = 0; visibleFace < nbVisibleFace; visibleFace++) {
			
			if (isShaded[visibleFace] == true) {

				finalColor += (projArea[visibleFace] / projAreaSum)*shadingFaces[visibleFace];
				sampledAvgColor += (projArea[visibleFace] / projAreaSum)*colorFaces[visibleFace];
			}

		}

		if (nbSampledPoints > 0) {
			shaded = true;
		}
		
		return finalColor;
		
	}

	int octant(Vector3f p, Eigen::Matrix3f S, Eigen::Matrix3f R)
	{
		float x = p.x();
		float y = p.y();
		float z = p.z();

		int oc = 0;
		if (x >= 0 && y >= 0 && z >= 0) {
			oc = 1;
			//cout << "Point lies in 1st octant\n";
		}
		else if (x < 0 && y >= 0 && z >= 0) {
			oc = 2;
			//cout << "Point lies in 2nd octant\n";
		}
		else if (x < 0 && y < 0 && z >= 0) {
			oc = 3;
			//cout << "Point lies in 3rd octant\n";

		}
		else if (x >= 0 && y < 0 && z >= 0) {
			oc = 4;
			//cout << "Point lies in 4th octant\n";

		}
		else if (x >= 0 && y >= 0 && z < 0) {
			oc = 5;
			//cout << "Point lies in 5th octant\n";

		}
		else if (x < 0 && y >= 0 && z < 0) {
			oc = 6;
			//cout << "Point lies in 6th octant\n";

		}
		else if (x < 0 && y < 0 && z < 0) {
			oc = 7;
			//cout << "Point lies in 7th octant\n";
		}
		else if (x >= 0 && y < 0 && z < 0) {
			oc = 8;
			//cout << "Point lies in 8th octant\n";
		}
		
		return oc;
	}

	void hughes_moeller(Vector3f& b1, Vector3f& b2, const Vector3f& n)
	{
		// Choose a vector orthogonal to n as the direction of b2.
		if (fabs(n.x()) > fabs(n.z())) b2 = Vector3f(-n.y(), n.x(), 0.0f);
		else b2 = Vector3f(0.0f, -n.z(), n.y());
		b2 *= rsqrtf(b2.dot(b2)); // Normalize b2
		b1 = b2.cross(n); // Construct b1 using a cross product
	}

	// compute the extremal normals given a cone of normals
	void estimateVMeshNormals(float radius, float height, Vector3f direction, int faceIndex, Normal3f triangleNormals[8][3]) {

		Vector3f tangent(0, 0, 0);
		Vector3f bitangent(0, 0, 0);

		hughes_moeller(tangent, bitangent, direction);

		for (int i = 0; i < 3; i++) {
			float x = radius * cos(2 * M_PI * i / 3);
			float z = radius * sin(2 * M_PI * i / 3);

			Vector3f pointCircle = (tangent * x) + (bitangent * z);
			Vector3f pointTriangle = pointCircle + (direction*height);

			triangleNormals[faceIndex][i] = pointTriangle.normalized();

		}

	}

	// estimate the cone of normals
	ConeOfDirections estimateCone(Normal3f *normalsInput, Normal3f avgNormal, int nbNormalsInput, bool validCone) {

		float thetaMax = 0;

		for (int i = 0; i < nbNormalsInput; i++) {

			float dot = normalsInput[i].dot(avgNormal);
			float theta = acos(dot);

			if (theta > thetaMax) {
				thetaMax = theta;
			}

		}

		thetaMax = MAX(thetaMax, 0.1);

		ConeOfDirections cone(avgNormal, thetaMax, validCone);

		return cone;

	}

	// estimate if it is a distribution of normals or a small set of normals.
	bool isDistribution(Normal3f *normalsInput, int nbNormalsInput, Normal3f normals[3]) {

		normals[0] = Normal3f(0, 0, 0);
		normals[1] = Normal3f(0, 0, 0);
		normals[2] = Normal3f(0, 0, 0);

		bool isDistribution = false;

		if (nbNormalsInput == 1) {
			normals[0] = normalsInput[0];
			isDistribution = false;
		}
		else if (nbNormalsInput == 2) {
			normals[0] = normalsInput[0];
			normals[1] = normalsInput[1];
			isDistribution = false;
		}
		else if (nbNormalsInput == 3) {
			normals[0] = normalsInput[0];
			normals[1] = normalsInput[1];
			normals[2] = normalsInput[2];
			isDistribution = false;

		} {

			normals[0] = normalsInput[0];
		
			int added = 1;

			for (int i = 1; i < nbNormalsInput && isDistribution == false; i++) {
				bool validNormal = true;
				for (int n = 0; n < added; n++) {

					float dot = normals[n].dot(normalsInput[i]);
					float theta = acos(dot);
					if (theta <= 0.0872665) {
						validNormal = false;

					}
				}

				if (validNormal == true) {


					if (added < 3) {
						normals[added] = normalsInput[i];
						added++;

					}
					else {
						isDistribution = true;
					}
				}

			}

		}

		return isDistribution;

	}

	// compute the color average.
	void estimateVMeshColorsMeans(Color3f *colorOctant_, int nbColors_, int faceIndex, Color3f triangleColors[8]) {

		
		if (nbColors_ > 0) {

			Color3f colorAvg(0, 0, 0);

			for (int i = 0; i < nbColors_; i++) {
				colorAvg += colorOctant_[i];

			}
			colorAvg /= nbColors_;

			triangleColors[faceIndex] = Color3f(colorAvg.x(), colorAvg.y(), colorAvg.z());

		}

	}

	// build the virtual mesh given then sampled normas and colors.
	void buildRepresentation(Normal3f *normals, Color3f *colors, int nbHits, Vector3f bbCenter) {

		SGGX sggx = SGGX();
		sggx.estimate(normals, nbHits);

		// intermediate structure
		Vector3f vertices[6];
		int triangles[8][3] = { {0, 2, 4},
								{0, 4, 3},
								{0, 3, 5},
								{0, 5, 2},
								{1, 2, 5},
								{1, 5, 3},
								{1, 3, 4},
								{1, 4, 2} };

		ConeOfDirections conesOfNormals[8];

		Normal3f triangleNormals_[8][3] = { {Normal3f(0,0,0), Normal3f(0,0,0), Normal3f(0,0,0)},
											{Normal3f(0,0,0), Normal3f(0,0,0), Normal3f(0,0,0)},
											{Normal3f(0,0,0), Normal3f(0,0,0), Normal3f(0,0,0)},
											{Normal3f(0,0,0), Normal3f(0,0,0), Normal3f(0,0,0)},
											{Normal3f(0,0,0), Normal3f(0,0,0), Normal3f(0,0,0)},
											{Normal3f(0,0,0), Normal3f(0,0,0), Normal3f(0,0,0)},
											{Normal3f(0,0,0), Normal3f(0,0,0), Normal3f(0,0,0)},
											{Normal3f(0,0,0), Normal3f(0,0,0), Normal3f(0,0,0)} };

		Color3f triangleColors_[8] = { Color3f(0,0,0), Color3f(0,0,0), Color3f(0,0,0), Color3f(0,0,0), Color3f(0,0,0), Color3f(0,0,0), Color3f(0,0,0), Color3f(0,0,0) };

		bool triangleIsDistribution_[8] = { false, false, false, false, false, false, false, false };


		if (nbHits > 0) {

			Eigen::Matrix3f R = sggx.R;
			Eigen::Matrix3f T = sggx.T;

			//transforming sphere points to ellipsoid - based on supplemental material from sggx paper (The Points of the Surface)
			vertices[0] = R * T * Vector3f(1, 0, 0);
			vertices[1] = R * T * Vector3f(-1, 0, 0);
			vertices[2] = R * T * Vector3f(0, 1, 0);
			vertices[3] = R * T * Vector3f(0, -1, 0);
			vertices[4] = R * T * Vector3f(0, 0, 1);
			vertices[5] = R * T * Vector3f(0, 0, -1);

			for (int face = 0; face < 8; face++) {

				Vector3f v1 = vertices[triangles[face][0]];
				Vector3f v2 = vertices[triangles[face][1]];
				Vector3f v3 = vertices[triangles[face][2]];

				Vector3f middle = (v1 + v2 + v3) / 3;

				Vector3f middleSGGXCoord = sggx.R.inverse()*middle;
				int faceOctant = octant(middleSGGXCoord, sggx.S, sggx.R);

				Vector3f v2v1 = (v2 - v1).normalized();
				Vector3f v3v1 = (v3 - v1).normalized();

				Vector3f normalFace = v2v1.cross(v3v1).normalized();
				Vector3f normalFaceSGGXCoord = sggx.R.inverse()*normalFace;

				int normalFaceOctant = octant(normalFaceSGGXCoord, sggx.S, sggx.R);

				if (normalFaceOctant != faceOctant) { // change the triangle index order to compute the octant correctly
					int vIndex = triangles[face][1];
					triangles[face][1] = triangles[face][2];
					triangles[face][2] = vIndex;

				}

				Normal3f *normalsOctant = new Normal3f[nbHits];
				int nbNormalsOctant = 0;
				Normal3f avgNormalOctant(0, 0, 0);

				Color3f *colorsOctant = new Color3f[nbHits];
				
				for (int i = 0; i < nbHits; i++) {

					Normal3f normal = normals[i];
					Color3f color = colors[i];
					Vector3f normalSGGXCoord = sggx.R.inverse()*normal;
					int normalOctant = octant(normalSGGXCoord, sggx.S, sggx.R); // transform the normal to the sggx coordinates (eigen vectors)

					if (normalOctant == faceOctant) {

						// store the normals and colors within the face octant (based on the normals)
						normalsOctant[nbNormalsOctant] = normal;
						colorsOctant[nbNormalsOctant] = color;
						nbNormalsOctant++;
					}

				}

				bool validCone = false;
				if (nbNormalsOctant > 0) {

					validCone = true;
					for (int i = 0; i < nbNormalsOctant; i++) {

						avgNormalOctant += normalsOctant[i];

					}
					avgNormalOctant /= nbNormalsOctant;
				}

				if (nbNormalsOctant > 0) { // if the number of normals in that octant is bigger than 0, compute the extremal normals 

					conesOfNormals[face] = estimateCone(normalsOctant, avgNormalOctant, nbNormalsOctant, validCone);

					Normal3f possibleNormals[3];
					bool isDistrib = isDistribution(normalsOctant, nbNormalsOctant, possibleNormals);
					triangleIsDistribution_[face] = isDistrib;

					if (isDistrib) {

						estimateVMeshNormals(conesOfNormals[face].getRadiusCircle(), conesOfNormals[face].getHeightTriangle(), avgNormalOctant, face, triangleNormals_);

					}
					else {

						triangleNormals_[face][0] = possibleNormals[0];
						triangleNormals_[face][1] = possibleNormals[1];
						triangleNormals_[face][2] = possibleNormals[2];
					}

					estimateVMeshColorsMeans(colorsOctant, nbNormalsOctant, face, triangleColors_);
				}


				delete[] normalsOctant;
				delete[] colorsOctant;


			}

			vertices[0] += bbCenter;
			vertices[1] += bbCenter;
			vertices[2] += bbCenter;
			vertices[3] += bbCenter;
			vertices[4] += bbCenter;
			vertices[5] += bbCenter;

			for (int i = 0; i < 8; i++) {


				// encode the data from float3 to uchar3
				triangleColors[i] = make_uchar3(triangleColors_[i].x() * 255, triangleColors_[i].y() * 255, triangleColors_[i].z() * 255);

				float3 triNormal0 = make_float3(triangleNormals_[i][0].x(), triangleNormals_[i][0].y(), triangleNormals_[i][0].z());
				float3 triNormal1 = make_float3(triangleNormals_[i][1].x(), triangleNormals_[i][1].y(), triangleNormals_[i][1].z());
				float3 triNormal2 = make_float3(triangleNormals_[i][2].x(), triangleNormals_[i][2].y(), triangleNormals_[i][2].z());

				//encode the data from float3 to char3
				triangleNormals[i][0] = make_char3(triNormal0.x * 127, triNormal0.y * 127, triNormal0.z * 127);
				triangleNormals[i][1] = make_char3(triNormal1.x * 127, triNormal1.y * 127, triNormal1.z * 127);
				triangleNormals[i][2] = make_char3(triNormal2.x * 127, triNormal2.y * 127, triNormal2.z * 127);


				triangleIsDistribution[i] = triangleIsDistribution_[i];

			}

			int vec = 0;
			for (int i = 0; i < 6; i += 2) {

				// compute and store the vectors and scales that defines the virtual mesh
				float scale = sqrtf((vertices[i] - bbCenter).dot(vertices[i] - bbCenter));
				float3 vectorI = normalize(make_float3(vertices[i].x() - bbCenter.x(), vertices[i].y() - bbCenter.y(), vertices[i].z() - bbCenter.z()));
				vectorI = encodeVector(vectorI);

				vectorsMesh[vec] = make_uchar3(vectorI.x * 255, vectorI.y * 255, vectorI.z * 255);
				vectorsMeshScales[vec].x = (scale / 10.0f) * 255;
				vec++;

			}




		}


	}


	

	


	
	
};


struct SubgridOfOpacities {

	float3 centers[SIZEOCCLUSIONCENTERS]; // center coordinates of the subgrid of opacity
	unsigned char occlusions[SIZEOCCLUSIONCENTERS]; // occlusion value of the subgrid of opacity

	__host__ __device__
	SubgridOfOpacities() {

		for (int voxel = 0; voxel < SIZEOCCLUSIONCENTERS; voxel++) {

			centers[voxel] = make_float3(0, 0, 0);
			occlusions[voxel] = 0;

		}

	}

	__host__ __device__
	float3 computeProjectionIntoPlane(float3 point, float3 normal, float3 origin) {

		float dotValue = dot((point - origin), (normal));
		float3 proj = point - dotValue * normal;

		return proj;
	}


	__host__ __device__
	float2 compute2DPointInPlane(float3 point, float3 vecU, float3 vecV, float3 origin) {

		float u = dot(point - origin, vecU);
		float v = dot(point - origin, vecV);


		float2 p = make_float2(u, v);

		return p;
	}

	__host__ __device__
	void getBBCorners(float3 min, float3 max, float3 *corners) {

		corners[0] = make_float3(min.x, min.y, min.z);
		corners[1] = make_float3(min.x, max.y, min.z);
		corners[2] = make_float3(max.x, min.y, min.z);
		corners[3] = make_float3(min.x, min.y, max.z);
		corners[4] = make_float3(max.x, max.y, min.z);
		corners[5] = make_float3(max.x, min.y, max.z);
		corners[6] = make_float3(min.x, max.y, max.z);
		corners[7] = make_float3(max.x, max.y, max.z);

	}

	__host__ __device__
	int computeGridPosition(float point, float gridMin, float step) {

		int cell = int((point - gridMin) / step);
		return cell;
	}


	
	__host__ __device__
	void hughes_moeller(float3& b1, float3& b2, const float3& n)
	{
		// Choose a vector orthogonal to n as the direction of b2.
		if (fabs(n.x) > fabs(n.z)) b2 = make_float3(-n.y, n.x, 0.0f);
		else b2 = make_float3(0.0f, -n.z, n.y);
		b2 *= rsqrtf(dot(b2, b2)); // Normalize b2
		b1 = cross(b2, n); // Construct b1 using a cross product
	}

	// Compute the intersection area between two circles 
	__host__ __device__
	bool circleIntersection(float X1, float Y1, float R1, float X2, float Y2, float R2, float &area) {


		float d, alpha, beta, a1, a2;
		bool intersect = false;

		d = sqrt((X2 - X1) * (X2 - X1) + (Y2 - Y1) * (Y2 - Y1));

		if (d > R1 + R2) {
			area = 0.0f;
		}
		else if (d <= abs(R2 - R1) && R1 >= R2) {
			area = M_PI * R2 * R2;
			intersect = true;
		}
		else if (d <= abs(R2 - R1) && R2 >= R1) {
			area = M_PI * R1 * R1;
			intersect = true;
		}
		else {
			alpha = acos((R1 * R1 + d * d - R2 * R2) / (2 * R1 * d)) * 2;
			beta = acos((R2 * R2 + d * d - R1 * R1) / (2 * R2 * d)) * 2;
			a1 = 0.5 * beta * R2 * R2 - 0.5 * R2 * R2 * sin(beta);
			a2 = 0.5 * alpha * R1 * R1 - 0.5 * R1 * R1 * sin(alpha);
			area = a1 + a2;
			intersect = true;
		}

		return intersect;

	}

	// Compute the contribution of each subvoxel.
	__host__ __device__
	void projectionKernel(float voxelOcclusion, float2 projectionCenter, float projectionRadius, int cellX, int cellY, float occlusionValueGrid[RESOLUTIONOCCGRID][RESOLUTIONOCCGRID], float2 centersGrid[RESOLUTIONOCCGRID][RESOLUTIONOCCGRID], float gridRadius, float &projectionSum) {

		int sizeKernel = 2;
		float areaGridCell = M_PI * gridRadius * gridRadius;
		for (int i = -1; i < sizeKernel; i++) {
			for (int j = -1; j < sizeKernel; j++) {

				int cellNeighX = cellX + i;
				int cellNeighY = cellY + j;


				if (cellNeighX >= 0 && cellNeighX < RESOLUTIONOCCGRID && cellNeighY >= 0 && cellNeighY < RESOLUTIONOCCGRID) { // valid pixel


					float areaIntersection = 0;
					bool intersect = circleIntersection(projectionCenter.x, projectionCenter.y, projectionRadius, centersGrid[cellNeighX][cellNeighY].x, centersGrid[cellNeighX][cellNeighY].y, gridRadius, areaIntersection);

					if (intersect) { // if intersect, compute the coverage and store the max value

						float projectionCoverage = clampf(areaIntersection / areaGridCell, 0.0f, 1.0f);

						float occlusion = voxelOcclusion * projectionCoverage;

						float diffOccl = (occlusion - occlusionValueGrid[cellNeighX][cellNeighY]);

						float m = fmaxf(occlusion, occlusionValueGrid[cellNeighX][cellNeighY]);
						
						occlusionValueGrid[cellNeighX][cellNeighY] = m;

					}

				}

			}
		}


	}

	__host__ __device__
	float computeOcclusion(float3 origin, float3 dir, float3 BBmax, float3 BBmin, float3 *subgridCenters) {
		
		float3 vecU, vecV;
		hughes_moeller(vecU, vecV, dir);

		float occlusionValueGrid[RESOLUTIONOCCGRID][RESOLUTIONOCCGRID];
		float2 centersGrid[RESOLUTIONOCCGRID][RESOLUTIONOCCGRID];

		float3 bbVertex[8];
		getBBCorners(BBmin, BBmax, bbVertex);

		float GridMaxX = -9999999999;
		float GridMaxY = -9999999999;
		float GridMinX = 9999999999;
		float GridMinY = 9999999999;

		//projecting points to compute the bounding box (projection plane)
		for (int i = 0; i < 8; i++) {

			float3 proj = computeProjectionIntoPlane(bbVertex[i], dir, origin);
			float2 pPlane = compute2DPointInPlane(proj, vecU, vecV, origin);

			GridMaxX = fmaxf(pPlane.x, GridMaxX);
			GridMaxY = fmaxf(pPlane.y, GridMaxY);

			GridMinX = fminf(pPlane.x, GridMinX);
			GridMinY = fminf(pPlane.y, GridMinY);
		}

		float stepX = (GridMaxX - GridMinX) / RESOLUTIONOCCGRID;
		float stepY = (GridMaxY - GridMinY) / RESOLUTIONOCCGRID;

		float xmin = GridMinX;
		float ymin = GridMinY;
		float xmax = xmin + stepX;
		float ymax = ymin + stepY;

		// computing grid cells's center
		for (int cellX = 0; cellX < RESOLUTIONOCCGRID; cellX++) {

			float xminNew = xmin + cellX * stepX;
			float xmaxNew = xmax + cellX * stepX;

			for (int cellY = 0; cellY < RESOLUTIONOCCGRID; cellY++) {


				float yminNew = ymin + cellY * stepY;
				float ymaxNew = ymax + cellY * stepY;

				centersGrid[cellX][cellY] = make_float2((xminNew + xmaxNew) / 2.0f, (yminNew + ymaxNew) / 2.0);

				occlusionValueGrid[cellX][cellY] = 0.0f;
			}
		}

		float projectionSum = 0;

		// projecting the cells to compute the occlusion
		for (int voxel = 0; voxel < SIZEOCCLUSIONCENTERS; voxel++) {

			float3 centerVoxel = BBmin + (subgridCenters[voxel] * (BBmax - BBmin));


			float3 proj = computeProjectionIntoPlane(centerVoxel, dir, origin);
			float2 pPlane = compute2DPointInPlane(proj, vecU, vecV, origin);

			int cellX = computeGridPosition(pPlane.x, GridMinX, stepX);
			int cellY = computeGridPosition(pPlane.y, GridMinY, stepY);

			if (occlusions[voxel] > 0) {
				projectionKernel((float)occlusions[voxel] / 255, pPlane, stepX*sqrtf(2.0), cellX, cellY, occlusionValueGrid, centersGrid, stepX / 2.0f, projectionSum);
			}

		}

		// compute the final occlusion by doing the average of the occlusion of the grid cells
		float finalOcclusion = 0;
		int nbFinalOcclusion = 0;
		for (int cellX = 0; cellX < RESOLUTIONOCCGRID; cellX++) {
			for (int cellY = 0; cellY < RESOLUTIONOCCGRID; cellY++) {

				
				finalOcclusion += occlusionValueGrid[cellX][cellY];

				nbFinalOcclusion++;
				

			}
		}

		
		finalOcclusion /= nbFinalOcclusion;
		
		return finalOcclusion;

	}

	__host__
	void buildRepresentation(BoundingBox3f bbox, Vector3f *raysDirections, Vector3f *raysOrigin, float *raysHits, int nbSamples, Point3f *hitsPosition, int nbHits) {
	
	
		int occlusionBBRes = 3;
		BoundingBox3f occlusionBB[SIZEOCCLUSIONCENTERS];
		float occlusionBBValue[SIZEOCCLUSIONCENTERS];
		bool hasHitPoint[SIZEOCCLUSIONCENTERS];

		float xmin = bbox.min.x(), ymin = bbox.min.y(), zmin = bbox.min.z();
		float xmax = xmin + bbox.getExtents().x() / occlusionBBRes, ymax = ymin + bbox.getExtents().y() / occlusionBBRes, zmax = zmin + bbox.getExtents().z() / occlusionBBRes;

		int indexOcBB = 0;
		for (int z = 0; z < occlusionBBRes; z++) {

			float zminNew = zmin + z * (bbox.getExtents().z() / occlusionBBRes);
			float zmaxNew = zmax + z * (bbox.getExtents().z() / occlusionBBRes);

			for (int y = 0; y < occlusionBBRes; y++) {

				float yminNew = ymin + y * (bbox.getExtents().y() / occlusionBBRes);
				float ymaxNew = ymax + y * (bbox.getExtents().y() / occlusionBBRes);

				for (int x = 0; x < occlusionBBRes; x++) {

					float xminNew = xmin + x * (bbox.getExtents().x() / occlusionBBRes);
					float xmaxNew = xmax + x * (bbox.getExtents().x() / occlusionBBRes);

					occlusionBBValue[indexOcBB] = 0;
					occlusionBB[indexOcBB] = BoundingBox3f(Vector3f(xminNew, yminNew, zminNew), Vector3f(xmaxNew, ymaxNew, zmaxNew));
					hasHitPoint[indexOcBB] = false;
					bool foundPointInside = false;
					for (int hit = 0; hit < nbHits && !foundPointInside; hit++) {
					
						if (occlusionBB[indexOcBB].contains(hitsPosition[hit])) { // If it is true for a single case, the value of the subgrid element will be greater than 0.
							hasHitPoint[indexOcBB] = true;
							foundPointInside = true;
						}

					}
					indexOcBB++;
				}

			}

		}

		for (int occlusionVoxelIndex = 0; occlusionVoxelIndex < SIZEOCCLUSIONCENTERS; occlusionVoxelIndex++) {
			int nbHits = 0;
			float occlusionAvgPerVoxel = 0.0f;
			for (int ray = 0; ray < nbSamples; ray++) {

				Ray3f raySampling;
				raySampling.d = raysDirections[ray];
				raySampling.o = raysOrigin[ray];

				raySampling.mint = 0.001f;
				raySampling.maxt = 3.40282e+038;
				raySampling.update();

				float near_, far_;
				bool hitBB = occlusionBB[occlusionVoxelIndex].rayIntersect(raySampling, near_, far_);

				if (hitBB) { // if it hits, the occlusion value of the will be updated. 

					occlusionAvgPerVoxel = (nbHits*occlusionAvgPerVoxel + raysHits[ray]) / (nbHits + 1);
					nbHits++;

				}

			}


			if (hasHitPoint[occlusionVoxelIndex]) {

				occlusionBBValue[occlusionVoxelIndex] = occlusionAvgPerVoxel;


			}


		}

		for (int voxel = 0; voxel < SIZEOCCLUSIONCENTERS; voxel++) {

			centers[voxel] = make_float3(occlusionBB[voxel].getCenter().x(), occlusionBB[voxel].getCenter().y(), occlusionBB[voxel].getCenter().z());
			occlusions[voxel] = occlusionBBValue[voxel] * 255;

		}
	}

};


#endif
