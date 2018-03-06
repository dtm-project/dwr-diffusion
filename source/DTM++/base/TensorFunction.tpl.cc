/**
 * @file Function.tpl.cc
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


// PROJECT includes
#include <DTM++/base/TensorFunction.tpl.hh>

// MPI includes

// DEAL.II includes

// C++ includes

namespace DTM {

template<int rank, int dim, class NumberType>
typename TensorFunction<rank,dim,NumberType>::value_type
TensorFunction<rank,dim,NumberType>::
value(
	const dealii::Point<dim,NumberType> &
) const {
	// make sure, that we do not call this base function.
	AssertThrow(false, dealii::ExcNotImplemented());
	
	// to avoid -Wreturn-type warning, we
	return value_type ();
}


template<int rank, int dim, class NumberType>
void
TensorFunction<rank,dim,NumberType>::
value_list(
	const std::vector< dealii::Point<dim,NumberType> > &xs,
	std::vector< value_type > &ys
) const {
	// check input and output data
	Assert(
		xs.size() == ys.size(),
		dealii::ExcDimensionMismatch(xs.size(), ys.size())
	);
	
	// create iterators on input and output data structures
	typename std::vector<
		dealii::Point<dim>, std::allocator<dealii::Point<dim,NumberType> >
	>::const_iterator x(xs.begin());
	
	typename std::vector< value_type >::iterator y(ys.begin());
	
	for ( ; x != xs.end(); ++x, ++y) {
		*y = value(*x);
	}
}

} // namespace

#include "TensorFunction.inst.in"
