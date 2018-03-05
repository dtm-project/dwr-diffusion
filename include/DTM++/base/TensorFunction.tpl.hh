/**
 * @file TensorFunction.tpl.hh
 * @author Uwe Koecher (UK)
 * @date 2015-02-24, UK
 *
 * @brief Abstract tensor function interface IR^d x IR -> T_rank[IR^d]
 */

/*  Copyright (C) 2012-2015 by Uwe Koecher                                    */
/*                                                                            */
/*  This file is part of DTM++.                                               */
/*                                                                            */
/*  DTM++ is free software: you can redistribute it and/or modify             */
/*  it under the terms of the GNU Lesser General Public License as            */
/*  published by the Free Software Foundation, either                         */
/*  version 3 of the License, or (at your option) any later version.          */
/*                                                                            */
/*  DTM++ is distributed in the hope that it will be useful,                  */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/*  GNU Lesser General Public License for more details.                       */
/*                                                                            */
/*  You should have received a copy of the GNU Lesser General Public License  */
/*  along with DTM++.  If not, see <http://www.gnu.org/licenses/>.            */

#ifndef __TensorFunction_tpl_hh
#define __TensorFunction_tpl_hh

// MPI includes

// DEAL.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/vector.h>

// C++ includes
#include <vector>

namespace DTM {

template<int rank, int dim, class NumberType=double>
class TensorFunction : public dealii::TensorFunction<rank,dim,NumberType> {
public:
	typedef dealii::Tensor<rank,dim,NumberType> value_type;
	
	TensorFunction(NumberType initial_time = NumberType(0.)) :
		dealii::TensorFunction<rank,dim,NumberType> (initial_time) {
	}
	
	virtual ~TensorFunction() = default;
	
	/// get value from a function evaluation
	virtual
	value_type
	value(
		const dealii::Point<dim,NumberType> &x
	) const;
	
	/// get values from multiple function evaluations
	virtual
	void
	value_list(
		const std::vector< dealii::Point<dim,NumberType> > &,
		std::vector< value_type > &
	) const;
};


} // namespace
#endif
