/**
 * @file Force_KoecherBruchhaeuser1.tpl.cc
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * @date 2018-10-23, UK
 * @date 2018-09-14, MPB
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

#include <diffusion/Force/Force_KoecherBruchhaeuser1.tpl.hh>

namespace diffusion {
namespace force {

template<int dim>
double
KoecherBruchhaeuser1<dim>::
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
	// NOTE:
	// - \f$ f = \rho \partial_t u - \epsilon (u_xx + u_yy) \f$
	// - u = u1(t) * u2(x,y,t)
	
	////////////////////////////////////////////////////////////////////////////
	const double u1{
		s * std::atan( 10.*M_PI_2 * (2.*t-1.) )
	};
	
	const double dt_u1{
		s * 10.*M_PI / ( 10.*M_PI_2 * (2.*t-1.) * 10.*M_PI_2 * (2.*t-1.) +1. )
	};
	
	////////////////////////////////////////////////////////////////////////////
	const double u2_div{
		(1. + a*(x[0]-x0)*(x[0]-x0) + a*(x[1]-y0)*(x[1]-y0))
	};
	
	const double u2_div2{
		u2_div * u2_div
	};
	
	const double u2_div3{
		u2_div2 * u2_div
	};
	
	const double u2{
		1./ u2_div
	};
	
	const double dt_u2{ 2 * a * (
			(x[0]-x0) * (.25*std::sin(omega*t)*omega) +
			(x[1]-y0) * (-.25*std::cos(omega*t)*omega)
		) / (-u2_div2)
	};
	
	const double dxx_u2{
		8. * a * a * (x[0]-x0) * (x[0]-x0) / u2_div3
		-2. * a / u2_div2
	};
	
	const double dyy_u2{
		8. * a * a * (x[1]-y0) * (x[1]-y0) / u2_div3
		-2. * a / u2_div2
	};
	
	////////////////////////////////////////////////////////////////////////////
	return rho * (dt_u1 * u2 + u1 * dt_u2) - epsilon * u1 * ( dxx_u2 + dyy_u2 );
}

}} //namespaces

#include "Force_KoecherBruchhaeuser1.inst.in"
