// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Lionel Heng <lionel.heng@ieee.org>
// Copyright (C) 2017 Andrew Hundt <ATHundt@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
#ifndef EIGEN_DUALQUATERNION_H
#define EIGEN_DUALQUATERNION_H

#include <cmath>

template<typename T>
T sinc(T x) { return (x == 0) ? 1 : std::sin(x) / x; }

namespace Eigen {

template<typename T>
class DualQuaternion;

typedef DualQuaternion<float> DualQuaternionf;
typedef DualQuaternion<double> DualQuaterniond;

template<typename T>
EIGEN_DEVICE_FUNC
DualQuaternion<T>
operator+(const DualQuaternion<T>& dq1, const DualQuaternion<T>& dq2);

template<typename T>
EIGEN_DEVICE_FUNC
DualQuaternion<T>
operator-(const DualQuaternion<T>& dq1, const DualQuaternion<T>& dq2);

template<typename T>
EIGEN_DEVICE_FUNC
DualQuaternion<T>
operator*(T scale, const DualQuaternion<T>& dq);

/** \geometry_module \ingroup Geometry_Module
  * \class DualQuaternion
  * \brief Class for dual quaternion expressions
  * \tparam Scalar type
  * \sa class DualQuaternion
  */
template<typename T>
class DualQuaternion {
public:
	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	DualQuaternion() {
		m_real = Eigen::Quaternion<T>(1, 0, 0, 0);
		m_dual = Eigen::Quaternion<T>(0, 0, 0, 0);
	}

	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	DualQuaternion(const Eigen::Quaternion<T>& r, const Eigen::Quaternion<T>& d) {
		m_real = r;
		m_dual = d;
	}

	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	DualQuaternion(const Eigen::Quaternion<T>& r, const Eigen::Matrix<T, 3, 1>& t) {
		m_real = r.normalized();
		m_dual = Eigen::Quaternion<T>(T(0.5) * (Eigen::Quaternion<T>(T(0), t(0), t(1), t(2)) * m_real).coeffs());
	}

	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	DualQuaternion<T> conjugate() const {
		return DualQuaternion<T>(m_real.conjugate(), m_dual.conjugate());
	}

	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	Eigen::Quaternion<T> dual() const {
		return m_dual;
	}

	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	DualQuaternion<T> exp() const {
		Eigen::Quaternion<T> real = expq(m_real);
		Eigen::Quaternion<T> dual = real * m_dual;

		return DualQuaternion<T>(real, dual);
	}

	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	void fromScrew(T theta, T d,
	               const Eigen::Matrix<T, 3, 1>& l,
	               const Eigen::Matrix<T, 3, 1>& m) {
		m_real = Eigen::AngleAxis<T>(theta, l);
		m_dual.w() = -d / 2.0 * std::sin(theta / 2.0);
		m_dual.vec() = std::sin(theta / 2.0) * m + d / 2.0 * std::cos(theta / 2.0) * l;
	}

	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	static DualQuaternion<T> identity() {
		return DualQuaternion<T>(Eigen::Quaternion<T>::Identity(),
		                         Eigen::Quaternion<T>(0, 0, 0, 0));
	}

	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	DualQuaternion<T> inverse() const {
		T sqrLen0 = m_real.squaredNorm();
		T sqrLenE = 2.0 * (m_real.coeffs().dot(m_dual.coeffs()));

		if (sqrLen0 > 0.0) {
			T invSqrLen0 = 1.0 / sqrLen0;
			T invSqrLenE = -sqrLenE / (sqrLen0 * sqrLen0);

			DualQuaternion<T> conj = conjugate();
			conj.m_real.coeffs() = invSqrLen0 * conj.m_real.coeffs();
			conj.m_dual.coeffs() = invSqrLen0 * conj.m_dual.coeffs() + invSqrLenE * conj.m_real.coeffs();

			return conj;
		} else {
			return DualQuaternion<T>::zeros();
		}
	}

	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	DualQuaternion<T> log() const {
		Eigen::Quaternion<T> real = logq(m_real);
		Eigen::Quaternion<T> dual = m_real.conjugate() * m_dual;
		T scale = T(1) / m_real.squaredNorm();
		dual.coeffs() *= scale;

		return DualQuaternion<T>(real, dual);
	}

	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	void norm(T& real, T& dual) const {
		real = m_real.norm();
		dual = m_real.coeffs().dot(m_dual.coeffs()) / real;
	}

	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	void normalize() {
		T length = m_real.norm();
		T lengthSqr = m_real.squaredNorm();

		// real part is of unit length
		m_real.coeffs() /= length;

		// real and dual parts are orthogonal
		m_dual.coeffs() /= length;
		m_dual.coeffs() -= (m_real.coeffs().dot(m_dual.coeffs()) * lengthSqr) * m_real.coeffs();
	}

	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	DualQuaternion<T> normalized() const {
		DualQuaternion<T> dq = *this;
		dq.normalize();

		return dq;
	}

	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	Eigen::Matrix<T, 3, 1> transformPoint(const Eigen::Matrix<T, 3, 1>& point) const {
		DualQuaternion<T> dq = (*this)
		                       * DualQuaternion<T>(Eigen::Quaternion<T>(1, 0, 0, 0),
		                                           Eigen::Quaternion<T>(0, point(0, 0), point(1, 0), point(2, 0)))
		                       * conjugate();

		Eigen::Matrix<T, 3, 1> p(dq.m_dual.x(), dq.m_dual.y(), dq.m_dual.z());

		// translation
		p += 2.0 * (m_real.w() * m_dual.vec() - m_dual.w() * m_real.vec() + m_real.vec().cross(m_dual.vec()));

		return p;
	}

	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	Eigen::Matrix<T, 3, 1> transformVector(const Eigen::Matrix<T, 3, 1>& vector) const{
		DualQuaternion<T> dq = (*this)
		                       * DualQuaternion<T>(Eigen::Quaternion<T>(1, 0, 0, 0),
		                                           Eigen::Quaternion<T>(0, vector(0, 0), vector(1, 0), vector(2, 0)))
		                       * conjugate();

		return Eigen::Matrix<T, 3, 1>(dq.m_dual.x(), dq.m_dual.y(), dq.m_dual.z());
	}

	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	Eigen::Quaternion<T> real() const {
		return m_real;
	}


	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	Eigen::Quaternion<T> rotation() const{
		return m_real;
	}
	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	Eigen::Matrix<T, 3, 1> translation() const {
		Eigen::Quaternion<T> t(2.0 * (m_dual * m_real.conjugate()).coeffs());

		Eigen::Matrix<T, 3, 1> tvec;
		tvec << t.x(), t.y(), t.z();

		return tvec;
	}

	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	Eigen::Quaternion<T> translationQuaternion() const {
		Eigen::Quaternion<T> t(2.0 * (m_dual * m_real.conjugate()).coeffs());
		return t;
	}

	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	Eigen::Matrix<T, 4, 4> toMatrix() const{
		Eigen::Matrix<T, 4, 4> H = Eigen::Matrix<T, 4, 4>::Identity();

		H.block(0, 0, 3, 3) = m_real.toRotationMatrix();

		Eigen::Quaternion<T> t(2.0 * (m_dual * m_real.conjugate()).coeffs());
		H(0, 3) = t.x();
		H(1, 3) = t.y();
		H(2, 3) = t.z();

		return H;
	}

	EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
	static DualQuaternion<T> zeros(){
		return DualQuaternion<T>(Eigen::Quaternion<T>(T(0), T(0), T(0), T(0)),
		                         Eigen::Quaternion<T>(T(0), T(0), T(0), T(0)));
	}

	EIGEN_DEVICE_FUNC
	DualQuaternion<T> operator*(T scale) const;
	EIGEN_DEVICE_FUNC
	DualQuaternion<T> operator*(const DualQuaternion<T>& other) const;

	friend DualQuaternion<T> operator+<>(const DualQuaternion<T>& dq1, const DualQuaternion<T>& dq2);
	friend DualQuaternion<T> operator-<>(const DualQuaternion<T>& dq1, const DualQuaternion<T>& dq2);

private:
	Eigen::Quaternion<T> m_real; // real part
	Eigen::Quaternion<T> m_dual; // dual part
};

template<typename T>
Eigen::Quaternion<T> expq(const Eigen::Quaternion<T>& q) {
	T a = q.vec().norm();
	T exp_w = std::exp(q.w());

	if (a == T(0)) {
		return Eigen::Quaternion<T>(exp_w, 0, 0, 0);
	}

	Eigen::Quaternion<T> res;
	res.w() = exp_w * T(std::cos(a));
	res.vec() = exp_w * T(sinc(a)) * q.vec();

	return res;
}

template<typename T>
Eigen::Quaternion<T> logq(const Eigen::Quaternion<T>& q) {
	T exp_w = q.norm();
	T w = std::log(exp_w);
	T a = std::acos(q.w() / exp_w);

	if (a == T(0)) {
		return Eigen::Quaternion<T>(w, T(0), T(0), T(0));
	}

	Eigen::Quaternion<T> res;
	res.w() = w;
	res.vec() = q.vec() / exp_w / (std::sin(a) / a);

	return res;
}


template<typename T>
EIGEN_DEVICE_FUNC
DualQuaternion<T>
DualQuaternion<T>::operator*(T scale) const {
	return DualQuaternion<T>(Eigen::Quaternion<T>(scale * m_real.coeffs()),
	                         Eigen::Quaternion<T>(scale * m_dual.coeffs()));
}

template<typename T>
EIGEN_DEVICE_FUNC
DualQuaternion<T>
DualQuaternion<T>::operator*(const DualQuaternion<T>& other) const {
	return DualQuaternion<T>(m_real * other.m_real,
	                         Eigen::Quaternion<T>((m_real * other.m_dual).coeffs() +
	                                              (m_dual * other.m_real).coeffs()));
}

template<typename T>
EIGEN_DEVICE_FUNC
DualQuaternion<T>
operator+(const DualQuaternion<T>& dq1, const DualQuaternion<T>& dq2) {
	return DualQuaternion<T>(Eigen::Quaternion<T>(dq1.m_real.coeffs() + dq2.m_real.coeffs()),
	                         Eigen::Quaternion<T>(dq1.m_dual.coeffs() + dq2.m_dual.coeffs()));
}

template<typename T>
EIGEN_DEVICE_FUNC
DualQuaternion<T>
operator-(const DualQuaternion<T>& dq1, const DualQuaternion<T>& dq2) {
	return DualQuaternion<T>(Eigen::Quaternion<T>(dq1.m_real.coeffs() - dq2.m_real.coeffs()),
	                         Eigen::Quaternion<T>(dq1.m_dual.coeffs() - dq2.m_dual.coeffs()));
}

template<typename T>
EIGEN_DEVICE_FUNC
DualQuaternion<T>
operator*(T scale, const DualQuaternion<T>& dq) {
	return DualQuaternion<T>(Eigen::Quaternion<T>(scale * dq.real().coeffs()),
	                         Eigen::Quaternion<T>(scale * dq.dual().coeffs()));
}

template<typename T>
EIGEN_DEVICE_FUNC
DualQuaternion<T>
expdq(const std::pair<Eigen::Quaternion<T>, Eigen::Quaternion<T> >& v8x1) {
	Eigen::Quaternion<T> real = expq(v8x1.first);
	Eigen::Quaternion<T> dual = real * v8x1.second;
	return DualQuaternion<T>(real, dual);
}

template<typename T>
EIGEN_DEVICE_FUNC
DualQuaternion<T>
logdq(const DualQuaternion<T>& dq) {
	Eigen::Quaternion<T> real = logq(dq.real());
	Eigen::Quaternion<T> dual = dq.real().inverse() * dq.dual();
	return DualQuaternion<T>(real, dual);
}

} // end namespace Eigen



#endif
