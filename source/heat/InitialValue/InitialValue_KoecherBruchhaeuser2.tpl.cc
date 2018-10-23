/**
 * @file InitialValue_KoecherBruchhaeuser2.tpl.cc
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * @date 2018-10-23, UK
 */

/*  Copyright (C) 2012-2018 by Uwe Koecher and contributors                   */
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

#include <heat/InitialValue/InitialValue_KoecherBruchhaeuser2.tpl.hh>

namespace heat {
namespace initial_value {

template<int dim>
double
KoecherBruchhaeuser2<dim>::
value(
	const dealii::Point<dim> &x,
	[[maybe_unused]]const unsigned int c
) const {
	Assert(
		c==0,
		dealii::ExcMessage(
			"you want to get component value which is not implemented"
		)
	);
	Assert(dim==2, dealii::ExcNotImplemented());
	Assert(this->get_time() >= 0., dealii::ExcNotImplemented());
	
	// get \f$ t \in [0, 1.0) \f$
	const double t{this->get_time() - std::floor(this->get_time())};
	
	const double omega{2.*M_PI};
	const double x0 = .5 + .25*std::cos(omega*t);
	const double y0 = .5 + .25*std::sin(omega*t);
	
	////////////////////////////////////////////////////////////////////////////
	const double u1{
		(t < .5) ?
		-std::atan( 10.*M_PI/2. * (4.*t-1.) ) :
		std::atan( 10.*M_PI/2. * (4.*(t-.5)-1.) )
	};
	
	////////////////////////////////////////////////////////////////////////////
	const double u2{
		1./(1. + a*(x[0]-x0)*(x[0]-x0) + a*(x[1]-y0)*(x[1]-y0))
	};
	
	////////////////////////////////////////////////////////////////////////////
	return u1 * u2;
}

}} //namespaces

#include "InitialValue_KoecherBruchhaeuser2.inst.in"
