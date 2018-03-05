/**
 * @file Function.tpl.cc
 * @author Uwe Koecher (UK)
 * @date 2015-02-23, UK
 * @date 2014-11-19, UK
 *
 * @brief Abstract function interface IR^d x IR -> IR^s
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
#include <DTM++/base/Function.tpl.hh>

// MPI includes

// DEAL.II includes

// C++ includes

namespace DTM {

template<int dim, int n_components>
double
Function<dim,n_components>::
value(
	const dealii::Point<dim> &x,
	const unsigned int c
) const {
	Assert((c < n_components), dealii::ExcIndexRange(c,0,n_components));
	
	dealii::Vector<double> y(n_components); y=0;
	vector_value(x, y);
	
	return y(c);
}


template<int dim, int n_components>
void
Function<dim,n_components>::
value_list(
	const std::vector< dealii::Point<dim> > &xs,
	std::vector<double> &ys,
	const unsigned int c
) const {
	Assert((c < n_components), dealii::ExcIndexRange(c,0,n_components));
	
	// check input and output data
	Assert(
		xs.size() == ys.size(),
		dealii::ExcDimensionMismatch(xs.size(), ys.size())
	);
	
	// create iterators on input and output data structures
	typename std::vector<
		dealii::Point<dim>, std::allocator<dealii::Point<dim> >
	>::const_iterator x(xs.begin());
	
	typename std::vector<double>::iterator y(ys.begin());
	
	for ( ; x != xs.end(); ++x, ++y) {
		*y = value(*x, c);
	}
}


template<int dim, int n_components>
void
Function<dim,n_components>::
vector_value_list(
	const std::vector< dealii::Point<dim> > &xs,
	std::vector< dealii::Vector<double> > &ys
) const {
	// check input and output data
	Assert(
		xs.size() == ys.size(),
		dealii::ExcDimensionMismatch(xs.size(), ys.size())
	);
	
	// create iterators on input and output data structures
	typename std::vector<
		dealii::Point<dim>, std::allocator<dealii::Point<dim> >
	>::const_iterator x(xs.begin());
	
	typename std::vector< dealii::Vector<double> >::iterator y(ys.begin());

	for ( ; x != xs.end(); ++x, ++y) {
		vector_value(*x, *y);
	}
}

} // namespace

#include "Function.inst.in"
