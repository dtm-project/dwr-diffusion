/**
 * @file Force_KoecherBruchhaeuser1.tpl.cc
 *
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 *
 * @date 2018-09-14, MPB, UK
 */

/*  Copyright (C) 2012-2018 by Uwe Koecher, Marius Paul Bruchhaeuser          */
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

#include <heat/Force/Force_KoecherBruchhaeuser1.tpl.hh>

namespace heat {
namespace force {

template<int dim>
double
KoecherBruchhaeuser1<dim>::
value(
	const dealii::Point<dim> &x,
	[[maybe_unused]]const unsigned int c
) const {
	Assert(c==0, dealii::ExcMessage("you want to get component value which is not implemented"));
	Assert(dim==2, dealii::ExcNotImplemented());
	
	const double t{this->get_time()};
	
	const double x0 = 0.5+0.25*std::cos(2.*M_PI*t);
	const double x1 = 0.5+0.25*std::sin(2.*M_PI*t);
	
	const double t0 = 10.*(M_PI_2)*(2.*t-1.);
	
	const double Nenner = 1. + a*( (x[0]-x0)*(x[0]-x0) + (x[1]-x1)*(x[1]-x1) );
	
	double dtu =
		// Quotient rule (f'*g - f*g') / g^2 with f=atan(t0), g=Nenner
// 		( (10.*M_PI / (t0*t0+1)) * Nenner
// 		- std::atan(t0)*( ( a * (x[0]-x0) * M_PI * std::sin(2.*M_PI*t) )
// 		- ( a * (x[1]-x1) * M_PI * std::cos(2.*M_PI*t)) ) )/
// 		(Nenner*Nenner);
		
		// Product rule f'*g + f*g' with f=atan(t0), g=1/Nenner 
// 		( (10.*M_PI / (1. + t0*t0)) * (1. / Nenner) )
// 		+
// 		(std::atan(t0) * ( ( a*( (x[1]-x1)*M_PI*std::cos(2.*M_PI*t)-(x[0]-x0)*M_PI*std::sin(2.*M_PI*t) ) )/
// 		(Nenner*Nenner) ) );
		
		// MAPLE
		( 10.*M_PI / ( Nenner*(25.*M_PI*M_PI*(2.*t-1)*(2.*t-1)+1) ) )
		-
		( ( std::atan(t0)*( a*M_PI*(x[0]-x0)*std::sin(2.*M_PI*t)-a*M_PI*(x[1]-x1)*std::cos(2.*M_PI*t) ) )/
		(Nenner*Nenner) );
	
	const double u_xx =
// 		std::atan(t0)*
// 		(-2.*a*( 1./(Nenner*Nenner)
// 		+ (x[0]-x0) * (-2./(Nenner*Nenner*Nenner)*2.*a*(x[0]-x0)) ) );

		// MAPLE
		( ( 8.*std::atan(t0)*a*a*(x[0]-x0)*(x[0]-x0) ) / ( Nenner*Nenner*Nenner ) )
		-
		( ( 2.*std::atan(t0)*a ) / ( Nenner*Nenner ) );
		
		
	const double u_yy =
// 		std::atan(t0)*
// 		(-2.*a*( 1./(Nenner*Nenner)
// 		+ (x[1]-x1) * (-2./(Nenner*Nenner*Nenner)*2.*a*(x[1]-x1)) ) );

		// MAPLE
		( ( 8.*std::atan(t0)*a*a*(x[1]-x1)*(x[1]-x1) ) / ( Nenner*Nenner*Nenner ) )
		-
		( ( 2.*std::atan(t0)*a ) / ( Nenner*Nenner ) );
	
	return dtu - epsilon * (u_xx+u_yy);
}

}} //namespaces

#include "Force_KoecherBruchhaeuser1.inst.in"
