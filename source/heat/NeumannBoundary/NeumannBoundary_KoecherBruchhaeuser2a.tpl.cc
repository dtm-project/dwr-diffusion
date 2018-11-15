/**
 * @file NeumannBoundary_KoecherBruchhaeuser2a.tpl.cc
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * @date 2018-11-15, UK
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

#include <heat/NeumannBoundary/NeumannBoundary_KoecherBruchhaeuser2a.tpl.hh>

namespace heat {
namespace neumann_boundary {

template<int dim>
double
KoecherBruchhaeuser2a<dim>::
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
		-s * std::atan( 10.*M_PI_2 * (4.*t-1.) ) :
		s * std::atan( 10.*M_PI_2 * (4.*(t-.5)-1.) )
	};
	
	////////////////////////////////////////////////////////////////////////////
	// \nabla u_2 * n; n=[-1; 0]^T
	const double dn_u2{
		2.*a*(x[0]-x0) /(
			(1. + a*(x[0]-x0)*(x[0]-x0) + a*(x[1]-y0)*(x[1]-y0)) *
			(1. + a*(x[0]-x0)*(x[0]-x0) + a*(x[1]-y0)*(x[1]-y0))
		)
	};
	
	////////////////////////////////////////////////////////////////////////////
	return u1 * (epsilon * dn_u2);
}

}} //namespaces

#include "NeumannBoundary_KoecherBruchhaeuser2a.inst.in"
