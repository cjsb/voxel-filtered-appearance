#include <iostream>
#include<time.h>
#include "helper_math.cuh"
#include "vector.h"
#include "sggx.h"
#include "representations.h"

// This sample program shows the usage on how to create the representations (Virtual Mesh and Subgrid of Opacities) from a set of samples
// Each sample has a ray origin, ray direction, occlusion value (1 for a hit and 0 for a miss). There is also the normals and colors of the hits.

int main() {

	// A really simple example of samples. You can add your samples. =)
	
	int nbSamples = 5;
	int nbHits = 3;
	Normal3f *normals = new Normal3f[nbHits];
	Color3f *colors = new Color3f[nbHits];
	Point3f *hitsPosition = new Point3f[nbHits];

	Vector3f *raysDirection = new Vector3f[nbSamples];
	Vector3f *raysOrigin = new Vector3f[nbSamples];
	float *raysHit = new float[nbSamples];
	
	Point3f minBoundBox(0.0f, 0.0f, 0.0f);
	Point3f maxBoundBox(1.0f, 1.0f, 1.0f);
	BoundingBox3f voxelBoundingBox(minBoundBox, maxBoundBox); // This is the voxel volume

	
	// Create a simple set of samples
	raysOrigin[0] = Vector3f(-0.799462f, 2.00732f, 0.301566f);
	hitsPosition[0] = Point3f(0.25652f, 0.635509f, 0.0254115f);
	raysDirection[0] = (hitsPosition[0] - raysOrigin[0]).normalized();
	raysHit[0] = 1.0f;
	normals[0] = Normal3f(1.0f, 0.0f, 0.0f);
	colors[0] = Color3f(1.0f, 1.0f, 0.0f);


	
	raysOrigin[1] = Vector3f(1.78157f, -0.296752f, -0.812539f);
	hitsPosition[1] = Point3f(0.97021f, 0.364099f, 0.172783f);
	raysDirection[1] = (hitsPosition[1] - raysOrigin[1]).normalized();
	raysHit[1] = 1.0f;
	normals[1] = Normal3f(1.0f, 0.0f, 0.0f);
	colors[1] = Color3f(1.0f, 1.0f, 0.0f);
	
	raysOrigin[2] = Vector3f(-0.343589f, 0.426322f, -1.31189f);
	hitsPosition[2] = Point3f(0.868913f, 0.242707f, 0.0732709f);
	raysDirection[2] = (hitsPosition[2] - raysOrigin[2]).normalized();
	raysHit[2] = 1.0f;
	normals[2] = Normal3f(1.0f, 0.0f, 0.0f);
	colors[2] = Color3f(1.0f, 1.0f, 0.0f);
	
	raysOrigin[3] = Vector3f(1.62704f, 0.713132f, -1.1384f);
	raysDirection[3] = (Vector3f(0.990497f, 0.66331f, 0.575583f) - raysOrigin[3]).normalized();
	raysHit[3] = 0.0f;
	
	raysOrigin[4] = Vector3f(-0.672379f, 2.01081f, -0.0856516f);
	raysDirection[4] = (Vector3f(0.700251f, 0.779725f, 0.445981f) - raysOrigin[4]).normalized();
	raysHit[3] = 0.0f;

	// Triangles index from the virtual mesh. It will be passed as a parameter in the future.
	unsigned int trianglesLinear[24] = { 0, 2, 4,
										 0, 4, 3,
										 0, 3, 5,
										 0, 5, 2,
										 1, 2, 5,
										 1, 5, 3,
										 1, 3, 4,
										 1, 4, 2 };

	// Precomputed subgrid positions, it will be used later to calculate new subgrid elements centers later.
	nori::BoundingBox3f bbox(nori::Point3f(0, 0, 0), nori::Point3f(1, 1, 1));
	int occlusionBBRes = 3;
	float xmin = bbox.min.x(), ymin = bbox.min.y(), zmin = bbox.min.z();
	float xmax = xmin + bbox.getExtents().x() / occlusionBBRes, ymax = ymin + bbox.getExtents().y() / occlusionBBRes, zmax = zmin + bbox.getExtents().z() / occlusionBBRes;
	float3* centersBB = new float3[SIZEOCCLUSIONCENTERS];
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

				float3 center = make_float3((xminNew + xmaxNew) / 2, (yminNew + ymaxNew) / 2, (zminNew + zmaxNew) / 2);

				centersBB[indexOcBB] = center;
				indexOcBB++;
			}

		}

	}

	// Build virtual mesh and subgrid of opacities

	VirtualMesh vMesh;
	vMesh.buildRepresentation(normals, colors, nbHits, voxelBoundingBox.getCenter());

	SubgridOfOpacities subgridOpac;
	subgridOpac.buildRepresentation(voxelBoundingBox, raysDirection, raysOrigin, raysHit, nbSamples, hitsPosition, nbHits);


	// Simple test
	float3 origin = make_float3(5, 0.5, 0.5);
	float3 direction = make_float3(-1, 0, 0);


	float3 minBB = make_float3(minBoundBox.x(), minBoundBox.y(), minBoundBox.z());
	float3 maxBB = make_float3(maxBoundBox.x(), maxBoundBox.y(), maxBoundBox.z());
	
	float3 lightPosition = make_float3(10, 0, 0);
	
	float3 voxelBoundingBoxCenter = make_float3(voxelBoundingBox.getCenter().x(), voxelBoundingBox.getCenter().y(), voxelBoundingBox.getCenter().z());
	
	int nbSamplesPerFace = 40; // Number of samples per visible face
	
	bool isShaded = false; // Stores if the sampled normals from the virtual mesh and color were used in shading
	
	float3 avgFaceColor = make_float3(0, 0, 0); // Stores the average of the samples normals, in the paper we use as ambient color

	curandState randState; // State of the random number generator
	curand_init(time(NULL), 0, 0, &randState);

	// Shading using the virtual mesh
	float3 shadedColor = vMesh.shadeWithVirtualMesh(origin, direction, lightPosition, voxelBoundingBoxCenter, trianglesLinear, nbSamplesPerFace, &randState, isShaded, avgFaceColor);

	cout << "shaded color: " << shadedColor.x << " " << shadedColor.y << " " << shadedColor.z << endl;

	// Compute occlusion
	float occlusion = subgridOpac.computeOcclusion(origin, direction, maxBB, minBB, centersBB);
	
	cout << "occlusion: " << occlusion << endl;
	
	//deleting new arrays - do not delete centersBB if you want to keep using the representations
	delete[] normals;
	delete[] colors;
	delete[] hitsPosition;

	delete[] raysDirection;
	delete[] raysOrigin;
	delete[] raysHit;
	delete[] centersBB;

	return 0;
}