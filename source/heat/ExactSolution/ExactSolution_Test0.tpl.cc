/**
 * @file ExactSolution_Test0.tpl.cc
 * @author Uwe Koecher (UK)
 * 
 * @date 2018-03-09, UK
 * @date 2018-03-08, included from ewave, UK
 * @date 2017-10-25, UK
 * @date 2013-08-15, DTM++ v1, UK
 */

/*  Copyright (C) 2012-2018 by Uwe Koecher                                    */
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
/*  along with DTM++.   If not, see <http://www.gnu.org/licenses/>.           */

#include <heat/ExactSolution/ExactSolution_Test0.tpl.hh>

// DEAL.II includes

// C++ includes

namespace heat {
namespace ExactSolution {

template<int dim>
double
Test0<dim>::
value(
	const dealii::Point<dim> &x,
	[[maybe_unused]]const unsigned int c
) const {
	Assert(c==0, dealii::ExcMessage("you want to get component value which is not implemented"));
	Assert(dim==2, dealii::ExcNotImplemented());
	
	const double t{this->get_time()};
	
	double res{0.};
	res =
		std::sin(M_PI * t) * 16 * x[0]*(x[0]-1) * x[1]*(x[1]-1);;
	
	return res;
}

}} //namespaces

#include "ExactSolution_Test0.inst.in"
