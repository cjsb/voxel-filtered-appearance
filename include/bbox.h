/*
	This file is part of Nori, a simple educational ray tracer

	Copyright (c) 2015 by Wenzel Jakob

	Nori is free software; you can redistribute it and/or modify
	it under the terms of the GNU General Public License Version 3
	as published by the Free Software Foundation.

	Nori is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "ray.h"
#include <iostream>
#include <iterator>
#include <map>

NORI_NAMESPACE_BEGIN

/**
 * \brief Generic n-dimensional bounding box data structure
 *
 * Maintains a minimum and maximum position along each dimension and provides
 * various convenience functions for querying and modifying them.
 *
 * This class is parameterized by the underlying point data structure,
 * which permits the use of different scalar types and dimensionalities, e.g.
 * \code
 * TBoundingBox<Vector3i> integerBBox(Point3i(0, 1, 3), Point3i(4, 5, 6));
 * TBoundingBox<Vector2d> doubleBBox(Point2d(0.0, 1.0), Point2d(4.0, 5.0));
 * \endcode
 *
 * \tparam T The underlying point data type (e.g. \c Point2d)
 * \ingroup libcore
 */

	enum AXIS { X, Y, Z };

template <typename _PointType> struct TBoundingBox {
	enum {
		Dimension = _PointType::Dimension
	};

	typedef _PointType                             PointType;
	typedef typename PointType::Scalar             Scalar;
	typedef typename PointType::VectorType         VectorType;

	/**
	 * \brief Create a new invalid bounding box
	 *
	 * Initializes the components of the minimum
	 * and maximum position to \f$\infty\f$ and \f$-\infty\f$,
	 * respectively.
	 */
	TBoundingBox() {
		reset();
	}

	/// Create a collapsed bounding box from a single point
	TBoundingBox(const PointType &p)
		: min(p), max(p) { }

	/// Create a bounding box from two positions
	TBoundingBox(const PointType &min, const PointType &max)
		: min(min), max(max) {
	}

	/// Test for equality against another bounding box
	bool operator==(const TBoundingBox &bbox) const {
		return min == bbox.min && max == bbox.max;
	}

	/// Test for inequality against another bounding box
	bool operator!=(const TBoundingBox &bbox) const {
		return min != bbox.min || max != bbox.max;
	}

	/// Calculate the n-dimensional volume of the bounding box
	Scalar getVolume() const {
		return (max - min).prod();
	}

	/// Calculate the n-1 dimensional volume of the boundary
	float getSurfaceArea() const {
		VectorType d = max - min;
		float result = 0.0f;
		for (int i = 0; i < Dimension; ++i) {
			float term = 1.0f;
			for (int j = 0; j < Dimension; ++j) {
				if (i == j)
					continue;
				term *= d[j];
			}
			result += term;
		}
		return 2.0f * result;
	}

	/// Return the center point
	PointType getCenter() const {
		return (max + min) * (Scalar) 0.5f;
	}

	/**
	 * \brief Check whether a point lies \a on or \a inside the bounding box
	 *
	 * \param p The point to be tested
	 *
	 * \param strict Set this parameter to \c true if the bounding
	 *               box boundary should be excluded in the test
	 */
	bool contains(const PointType &p, bool strict = false) const {
		if (strict) {
			return (p.array() > min.array()).all()
				&& (p.array() < max.array()).all();
		}
		else {
			return (p.array() >= min.array()).all()
				&& (p.array() <= max.array()).all();
		}
	}

	/**
	 * \brief Check whether a specified bounding box lies \a on or \a within
	 * the current bounding box
	 *
	 * Note that by definition, an 'invalid' bounding box (where min=\f$\infty\f$
	 * and max=\f$-\infty\f$) does not cover any space. Hence, this method will always
	 * return \a true when given such an argument.
	 *
	 * \param strict Set this parameter to \c true if the bounding
	 *               box boundary should be excluded in the test
	 */
	bool contains(const TBoundingBox &bbox, bool strict = false) const {
		if (strict) {
			return (bbox.min.array() > min.array()).all()
				&& (bbox.max.array() < max.array()).all();
		}
		else {
			return (bbox.min.array() >= min.array()).all()
				&& (bbox.max.array() <= max.array()).all();
		}
	}

	/**
	 * \brief Check two axis-aligned bounding boxes for possible overlap.
	 *
	 * \param strict Set this parameter to \c true if the bounding
	 *               box boundary should be excluded in the test
	 *
	 * \return \c true If overlap was detected.
	 */
	bool overlaps(const TBoundingBox &bbox, bool strict = false) const {
		if (strict) {
			return (bbox.min.array() < max.array()).all()
				&& (bbox.max.array() > min.array()).all();
		}
		else {
			return (bbox.min.array() <= max.array()).all()
				&& (bbox.max.array() >= min.array()).all();
		}
	}

	/**
	 * \brief Calculate the smallest squared distance between
	 * the axis-aligned bounding box and the point \c p.
	 */
	Scalar squaredDistanceTo(const PointType &p) const {
		Scalar result = 0;

		for (int i = 0; i < Dimension; ++i) {
			Scalar value = 0;
			if (p[i] < min[i])
				value = min[i] - p[i];
			else if (p[i] > max[i])
				value = p[i] - max[i];
			result += value * value;
		}

		return result;
	}

	/**
	 * \brief Calculate the smallest distance between
	 * the axis-aligned bounding box and the point \c p.
	 */
	Scalar distanceTo(const PointType &p) const {
		return std::sqrt(squaredDistanceTo(p));
	}

	/**
	 * \brief Calculate the smallest square distance between
	 * the axis-aligned bounding box and \c bbox.
	 */
	Scalar squaredDistanceTo(const TBoundingBox &bbox) const {
		Scalar result = 0;

		for (int i = 0; i < Dimension; ++i) {
			Scalar value = 0;
			if (bbox.max[i] < min[i])
				value = min[i] - bbox.max[i];
			else if (bbox.min[i] > max[i])
				value = bbox.min[i] - max[i];
			result += value * value;
		}

		return result;
	}

	/**
	 * \brief Calculate the smallest distance between
	 * the axis-aligned bounding box and \c bbox.
	 */
	Scalar distanceTo(const TBoundingBox &bbox) const {
		return std::sqrt(squaredDistanceTo(bbox));
	}

	/**
	 * \brief Check whether this is a valid bounding box
	 *
	 * A bounding box \c bbox is valid when
	 * \code
	 * bbox.min[dim] <= bbox.max[dim]
	 * \endcode
	 * holds along each dimension \c dim.
	 */
	bool isValid() const {
		return (max.array() >= min.array()).all();
	}

	/// Check whether this bounding box has collapsed to a single point
	bool isPoint() const {
		return (max.array() == min.array()).all();
	}

	/// Check whether this bounding box has any associated volume
	bool hasVolume() const {
		return (max.array() > min.array()).all();
	}

	/// Return the dimension index with the largest associated side length
	int getMajorAxis() const {
		VectorType d = max - min;
		int largest = 0;
		for (int i = 1; i < Dimension; ++i)
			if (d[i] > d[largest])
				largest = i;
		return largest;
	}

	/// Return the dimension index with the shortest associated side length
	int getMinorAxis() const {
		VectorType d = max - min;
		int shortest = 0;
		for (int i = 1; i < Dimension; ++i)
			if (d[i] < d[shortest])
				shortest = i;
		return shortest;
	}

	/**
	 * \brief Calculate the bounding box extents
	 * \return max-min
	 */
	VectorType getExtents() const {
		return max - min;
	}

	VectorType getInverseExtents() const {
		VectorType extents = max - min;
		extents[0] = 1.0f / extents[0];
		extents[1] = 1.0f / extents[1];
		extents[2] = 1.0f / extents[2];
		return extents;
	}

	/// Clip to another bounding box
	void clip(const TBoundingBox &bbox) {
		min = min.cwiseMax(bbox.min);
		max = max.cwiseMin(bbox.max);
	}

	/**
	 * \brief Mark the bounding box as invalid.
	 *
	 * This operation sets the components of the minimum
	 * and maximum position to \f$\infty\f$ and \f$-\infty\f$,
	 * respectively.
	 */
	void reset() {
		min.setConstant(std::numeric_limits<Scalar>::infinity());
		max.setConstant(-std::numeric_limits<Scalar>::infinity());
	}

	/// Expand the bounding box to contain another point
	void expandBy(const PointType &p) {
		min = min.cwiseMin(p);
		max = max.cwiseMax(p);
	}

	/// Expand the bounding box to contain another bounding box
	void expandBy(const TBoundingBox &bbox) {
		min = min.cwiseMin(bbox.min);
		max = max.cwiseMax(bbox.max);
	}

	/// Merge two bounding boxes
	static TBoundingBox merge(const TBoundingBox &bbox1, const TBoundingBox &bbox2) {
		return TBoundingBox(
			bbox1.min.cwiseMin(bbox2.min),
			bbox1.max.cwiseMax(bbox2.max)
		);
	}

	/// Return the index of the largest axis
	int getLargestAxis() const {
		VectorType extents = max - min;

		if (extents[0] >= extents[1] && extents[0] >= extents[2])
			return 0;
		else if (extents[1] >= extents[0] && extents[1] >= extents[2])
			return 1;
		else
			return 2;
	}

	/// Return the position of a bounding box corner
	PointType getCorner(int index) const {
		PointType result;
		for (int i = 0; i < Dimension; ++i)
			result[i] = (index & (1 << i)) ? max[i] : min[i];
		return result;
	}

	/// Return a string representation of the bounding box
	std::string toString() const {
		if (!isValid())
			return "BoundingBox[invalid]";
		else
			return tfm::format("BoundingBox[min=%s, max=%s]", min.toString(), max.toString());
	}

	/// Check if a ray intersects a bounding box
	bool rayIntersect(const Ray3f &ray) const {
		float nearT = -std::numeric_limits<float>::infinity();
		float farT = std::numeric_limits<float>::infinity();

		for (int i = 0; i < 3; i++) {
			float origin = ray.o[i];
			float minVal = min[i], maxVal = max[i];

			if (ray.d[i] == 0) {
				if (origin < minVal || origin > maxVal)
					return false;
			}
			else {
				float t1 = (minVal - origin) * ray.dRcp[i];
				float t2 = (maxVal - origin) * ray.dRcp[i];

				if (t1 > t2)
					std::swap(t1, t2);

				nearT = std::max(t1, nearT);
				farT = std::min(t2, farT);

				if (!(nearT <= farT))
					return false;
			}
		}

		return ray.mint <= farT && nearT <= ray.maxt;
	}

	/// Return the overlapping region of the bounding box and an unbounded ray
	bool rayIntersect(const Ray3f &ray, float &nearT, float &farT) const {
		nearT = -std::numeric_limits<float>::infinity();
		farT = std::numeric_limits<float>::infinity();

		for (int i = 0; i < 3; i++) {
			float origin = ray.o[i];
			float minVal = min[i], maxVal = max[i];

			if (ray.d[i] == 0) {
				if (origin < minVal || origin > maxVal)
					return false;
			}
			else {
				float t1 = (minVal - origin) * ray.dRcp[i];
				float t2 = (maxVal - origin) * ray.dRcp[i];

				if (t1 > t2)
					std::swap(t1, t2);

				nearT = std::max(t1, nearT);
				farT = std::min(t2, farT);

				if (!(nearT <= farT))
					return false;
			}
		}



		return true;
	}


	Point3f* getCorners() const {
		Point3f* corners = new Point3f[8];
		corners[0] = Point3f(min.x(), min.y(), min.z());
		corners[1] = Point3f(min.x(), max.y(), min.z());
		corners[2] = Point3f(max.x(), min.y(), min.z());
		corners[3] = Point3f(min.x(), min.y(), max.z());
		corners[4] = Point3f(max.x(), max.y(), min.z());
		corners[5] = Point3f(max.x(), min.y(), max.z());
		corners[6] = Point3f(min.x(), max.y(), max.z());
		corners[7] = Point3f(max.x(), max.y(), max.z());
		return corners;
	}

	bool intersectPlane(Vector3f &n, const Point3f &p0, const Point3f &origin, const Vector3f &dir, float &t) const
	{
		// assuming vectors are all normalized
		//float denom = dotProduct(n, dir);
		float denom = n.dot(dir);

		if (denom > 1e-6) {
			Point3f p0l0 = p0 - origin;
			//t = dotProduct(p0l0, n) / denom;
			t = p0l0.dot(n) / (denom);
			if (t >= 0) {

				return true;
			}
		}

		return false;
	}


	bool intersectPlanes(const Ray3f &ray, Vector3f &normal, int &index) const {

		Point3f* corners = getCorners();
		float tmin = std::numeric_limits<float>::max();
		bool intersect = false;

		//ray.o = ray.o + ray.d * ray.mint;
		//	_ray.d = ray.d;
		//	_ray.mint = ray.mint;
		//	_ray.maxt = ray.maxt;
		//	_ray.update();
		Point3f centers[6]; // down, up, right, left, front, back
		Vector3f normals[6];

		centers[0] = (corners[0] + corners[2] + corners[3] + corners[5]) / 4.0;
		normals[0] = Vector3f(0, -1, 0);

		centers[1] = (corners[1] + corners[4] + corners[6] + corners[7]) / 4.0;
		normals[1] = Vector3f(0, 1, 0);

		centers[2] = (corners[4] + corners[7] + corners[5] + corners[2]) / 4.0;
		normals[2] = Vector3f(0, 0, 1);

		centers[3] = (corners[0] + corners[1] + corners[6] + corners[3]) / 4.0;
		normals[3] = Vector3f(0, 0, -1);

		centers[4] = (corners[0] + corners[1] + corners[4] + corners[2]) / 4.0;
		normals[4] = Vector3f(-1, 0, 0);

		centers[5] = (corners[3] + corners[6] + corners[7] + corners[5]) / 4.0;
		normals[5] = Vector3f(1, 0, 0);

		for (int i = 0; i < 6; i++) {
			float t = 0;
			if (intersectPlane(normals[i], centers[i], ray.o, ray.d, t) == true) {
				intersect = true;
				if (t < tmin) {
					tmin = t;
					normal = normals[i];
					index = i;
				}
			}
		}

		return intersect;
	}

	// x = 0; y = 1; z = 2
	bool rayIntersectPlanes(const Ray3f &ray, Vector3f &normal, int &index) const {

		bool intersect = intersectPlanes(ray, normal, index);
		if (index == 0 || index == 1) {
			index = 0;
		}
		else if (index == 2 || index == 3) {
			index = 1;
		}
		else {
			index = 2;
		}

		return intersect;
	}

	bool compare_float(float x, float y, float epsilon = 0.000003f) const {
		if (fabs(x - y) < epsilon)
			return true; //they are same
		return false; //they are not same
	}

	bool rayIntersectFace2(const Ray3f &ray, Vector3f &normal, int &index) const {

		float near, far;
		bool intersect = rayIntersect(ray, near, far);

		Vector3f p = ray.o + ray.d*near;
		p -= min;

		Vector3f extents = getExtents();

		//cout << "point: " << p.x() << " " << p.y() << " "<< p.z() << endl;
		//cout << "extents: " << extents.x() << " " << extents.y() << " " <<  extents.z() << endl;

		if ((compare_float(p.x(), 0.0f) || compare_float(p.x(), extents.x())) && p.y() >= 0 && p.z() >= 0) {
			normal = Vector3f(1, 0, 0);
		}
		else if ((compare_float(p.y(), 0.0f) || compare_float(p.y(), extents.y())) && p.x() >= 0 && p.z() >= 0) {
			normal = Vector3f(0, 1, 0);
		}
		else if ((compare_float(p.z(), 0.0f) || compare_float(p.z(), extents.z())) && p.x() >= 0 && p.y() >= 0) {
			normal = Vector3f(0, 0, 1);
		}
		else {
			cout << "not this case" << endl;
			normal = Vector3f(1, 1, 1);

		}


		return intersect;
	}


	bool rayIntersect(Point3f p0, Point3f p1, Point3f p2, const Ray3f &ray, float &u, float &v, float &t) const {

		/* Find vectors for two edges sharing v[0] */
		Vector3f edge1 = p1 - p0, edge2 = p2 - p0;

		/* Begin calculating determinant - also used to calculate U parameter */
		Vector3f pvec = ray.d.cross(edge2);

		/* If determinant is near zero, ray lies in plane of triangle */
		float det = edge1.dot(pvec);

		if (det > -1e-8f && det < 1e-8f)
			return false;
		float inv_det = 1.0f / det;

		/* Calculate distance from v[0] to ray origin */
		Vector3f tvec = ray.o - p0;

		/* Calculate U parameter and test bounds */
		u = tvec.dot(pvec) * inv_det;
		if (u < 0.0 || u > 1.0)
			return false;

		/* Prepare to test V parameter */
		Vector3f qvec = tvec.cross(edge1);

		/* Calculate V parameter and test bounds */
		v = ray.d.dot(qvec) * inv_det;
		if (v < 0.0 || u + v > 1.0)
			return false;

		/* Ray intersects triangle -> compute t */
		t = edge2.dot(qvec) * inv_det;

		return t >= ray.mint && t <= ray.maxt;
	}

	float intersectFaceA(Point3f p0, Point3f p1, Point3f p2, Point3f p3, const Ray3f &ray, float &u, float &v, float &t) const {
		float t_ = -1;

		if (rayIntersect(p0, p1, p2, ray, u, v, t)) {
			t_ = t;
		}
		else if (rayIntersect(p2, p3, p0, ray, u, v, t)) {
			t_ = t;
		}

		return t_;
	}

	bool rayIntersectFace(const Ray3f &ray, Vector3f &normal, int &index) const {

		bool intersect = false;
		Point3f* corners = getCorners();
		float tmax = std::numeric_limits<float>::max();

		float u, v, t;

		for (int i = 0; i < 6; i++) {
			int index0 = pointFace[i][0];
			int index1 = pointFace[i][1];
			int index2 = pointFace[i][2];
			int index3 = pointFace[i][3];

			float t_ = intersectFaceA(corners[index0], corners[index1], corners[index2], corners[index3], ray, u, v, t);
			if (t_ > 0) {
				if (t_ < tmax) {
					tmax = t_;
					if (i == 0 || i == 1) {
						normal = Vector3f(1, 0, 0);
						//z
					}
					if (i == 2 || i == 3) {
						normal = Vector3f(0, 1, 0);
						//x
					}
					if (i == 4 || i == 5) {
						normal = Vector3f(0, 0, 1);
						//y
					}

				}
			}

		}
		/*if (rayIntersect(corners[5], corners[2], corners[4], ray, u, v, t) || rayIntersect(corners[4], corners[7], corners[5], ray, uu, vv, tt)) {

			normal = Vector3f(1, 0, 0);

		}
		else if (rayIntersect(corners[1], corners[0], corners[2], ray, u, v, t) || rayIntersect(corners[1], corners[4], corners[2], ray, uu, vv, tt)) {
			normal = Vector3f(0, 1, 0);

		}
		else {

			normal = Vector3f(1, 1, 1);
		}*/
		return intersect;
	}

	AXIS rayIntersectAxis(const Ray3f &ray) const {

		bool intersect = false;
		Point3f* corners = getCorners();
		float tmax = std::numeric_limits<float>::max();
		AXIS axis = X;

		float u, v, t;

		for (int i = 0; i < 6; i++) {
			int index0 = pointFace[i][0];
			int index1 = pointFace[i][1];
			int index2 = pointFace[i][2];
			int index3 = pointFace[i][3];

			float t_ = intersectFaceA(corners[index0], corners[index1], corners[index2], corners[index3], ray, u, v, t);
			if (t_ > 0) {
				if (t_ < tmax) {
					tmax = t_;
					if (i == 0 || i == 1) {
						axis = Z;
						//z
					}
					if (i == 2 || i == 3) {
						//normal = Vector3f(0, 1, 0);

						axis = X;
						//x
					}
					if (i == 4 || i == 5) {

						//normal = Vector3f(0, 0, 1);
						axis = Y;
						//y
					}

				}
			}

		}

		delete[] corners;
		return axis;
	}

	/*IntersectionPlaneNormal(const Ray3f &ray) const {



	}
*/
///*
//// offset origin by tMin
//		r = new Ray(r.Origin + r.Direction * tMin, r.Direction, r.Time);

//		distance = 0;
//		normal = 0;

//		// from "A Ray-Box Intersection Algorithm and Efficient Dynamic Voxel Rendering" (Majercik et al.)
//		// http://jcgt.org/published/0007/03/04/
//		float3 rayDirection = r.Direction;
//		float winding = cmax(abs(r.Origin) * box.InverseExtents) < 1 ? -1 : 1;
//		float3 sgn = -sign(rayDirection);
//		float3 distanceToPlane = (box.Extents * winding * sgn - r.Origin) / rayDirection;

//		bool3 test = distanceToPlane >= 0 & bool3(
//			all(abs(r.Origin.yz + rayDirection.yz * distanceToPlane.x) < box.Extents.yz),
//			all(abs(r.Origin.zx + rayDirection.zx * distanceToPlane.y) < box.Extents.zx),
//			all(abs(r.Origin.xy + rayDirection.xy * distanceToPlane.z) < box.Extents.xy));

//		sgn = test.x ? float3(sgn.x, 0, 0) : test.y ? float3(0, sgn.y, 0) : float3(0, 0, test.z ? sgn.z : 0);
//		bool3 nonZero = sgn != 0;
//		if (!any(nonZero)) return false;

//		distance = nonZero.x ? distanceToPlane.x : nonZero.y ? distanceToPlane.y : distanceToPlane.z;
//		distance += tMin;

//		if (distance > tMax) return false;

//		normal = sgn;
//		return true;
//*/

//bool rayIntersectrayIntersectionMajercik(const Ray3f &ray, float &nearT, float &farT, float &distance, Point3f &normal) const {
//	
//	Ray3f _ray;

//	_ray.o = ray.o + ray.d * ray.mint;
//	_ray.d = ray.d;
//	_ray.mint = ray.mint;
//	_ray.maxt = ray.maxt;
//	_ray.update();

//	distance = 0;
//	normal = Point3f(0,0,0);

//	Point3f rayDirection = _ray.d;
//	Point3f inverseExtents = getInverseExtents();
//	Point3f absOrigin(abs(ray.o.x()), abs(ray.o.y()), abs(ray.o.z()));
//	Point3f inverseWinder(absOrigin.x() * inverseExtents.x(), absOrigin.y() * inverseExtents.y(), absOrigin.z() * inverseExtents.z());
//	float windMax = 0;
//	int largest = 0;
//	for (int i = 1; i < 3; ++i) {
//		if (inverseWinder[i] > inverseWinder[largest])
//			largest = i;
//	}
//	windMax = inverseWinder[largest];
//		
//	float winding = windMax < 1 ? -1 : 1;
//	Point3f sgn( -sign(rayDirection.x()), -sign(rayDirection.y()), -sign(rayDirection.z()));
//	//Point3f distanceToPlane = (getExtents() * winding * sgn - r.o) / rayDirection;

//	bool testX = distanceToPlane >= 0 && all(abs(r.o.yz + rayDirection.yz * distanceToPlane.x) < getExtents().yz);
//	bool testY = distanceToPlane >= 0 && all(abs(r.o.zx + rayDirection.zx * distanceToPlane.y) < getExtents().zx);
//	bool testX = distanceToPlane >= 0 && all(abs(r.o.xy + rayDirection.xy * distanceToPlane.z) < getExtents().xy);

//	/*bool3 test = distanceToPlane >= 0 & bool3(
//		all(abs(r.Origin.yz + rayDirection.yz * distanceToPlane.x) < box.Extents.yz),
//		all(abs(r.Origin.zx + rayDirection.zx * distanceToPlane.y) < box.Extents.zx),
//		all(abs(r.Origin.xy + rayDirection.xy * distanceToPlane.z) < box.Extents.xy));*/

//	//sgn = test.x ? float3(sgn.x, 0, 0) : test.y ? float3(0, sgn.y, 0) : float3(0, 0, test.z ? sgn.z : 0);*/
//	/*bool3 nonZero = sgn != 0;
//	if (!any(nonZero)) return false;

//	distance = nonZero.x ? distanceToPlane.x : nonZero.y ? distanceToPlane.y : distanceToPlane.z;
//	distance += tMin;

//	if (distance > tMax) return false;

//	normal = sgn;*/
//	return true;


//	return true;
//}



	bool rayIntersection(const Ray3f &ray, float &nearT, float &farT) const {
		/*nearT = -std::numeric_limits<float>::infinity();
		farT = std::numeric_limits<float>::infinity();*/

		//PointType tMin = (min - ray.o) / ray.d;
		//PointType tMax = (max - ray.o) / ray.d;
		//PointType t1 = min(tMin, tMax);
		//PointType t2 = max(tMin, tMax);
		//nearT = max(max(t1[0], t1[1]), t1[2]);
		//farT = min(min(t2[0], t2[1]), t2[2]);
		////return vec2(tNear, tFar);

		//if (nearT > farT) {
		//	return false;
		//}else {
		//	return true;
		//}
		PointType tMin;
		PointType tMax;
		PointType t1;
		PointType t2;

		for (int i = 0; i < 3; i++) {

			tMin[i] = (min[i] - ray.o[i]) / ray.d[i];
			tMax[i] = (max[i] - ray.o[i]) / ray.d[i];
			t1[i] = min(tMin[i], tMax[i]);
			t2[i] = max(tMin[i], tMax[i]);



		}

		nearT = max(max(t1[0], t1[1]), t1[2]);
		farT = min(min(t2[0], t2[1]), t2[2]);
		//return vec2(tNear, tFar);

		if (nearT > farT) {
			return false;
		}
		else {
			return true;
		}

		/*for (int i = 0; i < 3; i++) {
			float origin = ray.o[i];
			float minVal = min[i], maxVal = max[i];

			if (ray.d[i] == 0) {
				if (origin < minVal || origin > maxVal)
					return false;
			}
			else {
				float t1 = (minVal - origin) * ray.dRcp[i];
				float t2 = (maxVal - origin) * ray.dRcp[i];

				if (t1 > t2)
					std::swap(t1, t2);

				nearT = std::max(t1, nearT);
				farT = std::min(t2, farT);

				if (!(nearT <= farT))
					return false;
			}
		}*/

		return true;
	}


	int pointFace[6][4] = {
   {2,0,1,4} , // FRONT (z)
   {5,3,6,7} , // BACK (z)
   {5,2,4,7} , // RIGHT (x)
   {0,1,6,3} , // LEFT (x)
   {0,3,5,2} , // BOTTOM (y)
   {1,6,7,4} }; //UP (y)

	PointType min; ///< Component-wise minimum 
	PointType max; ///< Component-wise maximum 
};


NORI_NAMESPACE_END
