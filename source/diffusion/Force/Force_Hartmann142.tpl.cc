/**
 * @file Force_Hartmann142.tpl.cc
 * @author Uwe Koecher (UK)
 * 
 * @date 2018-07-20, Hartmann142, MPB, UK
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

#include <diffusion/Force/Force_Hartmann142.tpl.hh>

// DEAL.II includes

// C++ includes

namespace diffusion {
namespace force {

template<int dim>
double
Hartmann142<dim>::
value(
	const dealii::Point<dim> &x,
	[[maybe_unused]]const unsigned int c
) const {
	Assert(c==0, dealii::ExcMessage("you want to get component value which is not implemented"));
	Assert(dim==2, dealii::ExcNotImplemented());
	
	const double t{this->get_time()};
	
// 	const double epsilon = function.epsilon->value(x, 0);
	
	const double x0 = 0.5+0.25*std::cos(2.*M_PI*t);
	const double x1 = 0.5+0.25*std::sin(2.*M_PI*t);
	
	const double Nenner = 1. + a*( (x[0]-x0)*(x[0]-x0) + (x[1]-x1)*(x[1]-x1) );
	
	double dtu =
		-( ( a * (x[0]-x0) * M_PI * std::sin(2.*M_PI*t) ) - ( a * (x[1]-x1) * M_PI * std::cos(2.*M_PI*t)) ) /
		(Nenner*Nenner);
	
	const double u_xx =
		-2.*a*( 1./(Nenner*Nenner)
		+ (x[0]-x0) * (-2./(Nenner*Nenner*Nenner)*2.*a*(x[0]-x0)) );
		
	const double u_yy =
		-2.*a*( 1./(Nenner*Nenner)
		+ (x[1]-x1) * (-2./(Nenner*Nenner*Nenner)*2.*a*(x[1]-x1)) );
	
	return dtu - epsilon * (u_xx+u_yy);
}

}} //namespaces

#include "Force_Hartmann142.inst.in"
