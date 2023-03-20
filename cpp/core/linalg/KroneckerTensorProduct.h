// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Kolja Brix <brix@igpm.rwth-aachen.de>
// Copyright (C) 2011 Andreas Platen <andiplaten@gmx.de>
// Copyright (C) 2012 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace nnrt::core::linalg::kernel{
template<typename Lhs, typename Rhs, typename Dest>
NNRT_DEVICE_WHEN_CUDACC
inline void ComputeKroneckerProduct(Dest& dst, const Lhs& m_A, const Rhs& m_B) {
	const int BlockRows = Rhs::RowsAtCompileTime,
			BlockCols = Rhs::ColsAtCompileTime;
	const Eigen::Index Br = m_B.rows(),
			Bc = m_B.cols();
	for (Eigen::Index i = 0; i < m_A.rows(); ++i)
		for (Eigen::Index j = 0; j < m_A.cols(); ++j)
			Eigen::Block<Dest, BlockRows, BlockCols>(dst, i * Br, j * Bc, Br, Bc) = m_A.coeff(i, j) * m_B;
}

} // namespace nnrt::core::linalg::kernel
