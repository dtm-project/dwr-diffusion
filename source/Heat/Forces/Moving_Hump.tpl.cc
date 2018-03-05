/**
 * @file Moving_Hump.tpl.cc
 * @author Marius Paul Bruchhäuser (MPB), Uwe Koecher (UK)
 * @date 2017-08-01, MPB, UK
 *
 * @brief Heat Problem
 */

/*  Copyright (C) 2012-2017 by Marius Paul Bruchhäuser, Uwe Köcher            */
/*                                                                            */
/*  This file is part of DTM++/aWave. (Acoustic Wave Equation Solver)         */
/*                                                                            */
/*  DTM++/aWave is free software: you can redistribute it and/or modify       */
/*  it under the terms of the GNU Lesser General Public License as            */
/*  published by the Free Software Foundation, either                         */
/*  version 3 of the License, or (at your option) any later version.          */
/*                                                                            */
/*  DTM++/aWave is distributed in the hope that it will be useful,            */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/*  GNU Lesser General Public License for more details.                       */
/*                                                                            */
/*  You should have received a copy of the GNU Lesser General Public License  */
/*  along with DTM++/aWave.   If not, see <http://www.gnu.org/licenses/>.     */

// PROJECT includes
#include <Heat/Forces/Moving_Hump.tpl.hh>

// DEAL.II includes
#include <deal.II/base/function.h>

// C++ includes
#include <fstream>
#include <vector>

namespace Heat {

template <int dim>
Moving_Hump<dim>::Moving_Hump (std::shared_ptr< dealii::Function<dim> > _epsilon,
				   std::shared_ptr< dealii::Function<dim> > _BoundaryValues) 
	: 
	epsilon(_epsilon),
	BoundaryValues(_BoundaryValues)
	{}

////////////////////////////////////////////////////////////////////////////////
//////////////// Moving Hump ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <int dim>
double Moving_Hump<dim>::value (
	const dealii::Point<dim> &p,
	const unsigned int /* component */
) const {
	double return_value = 0.0;
	
	// time variable t
	double t;
	t = this->get_time();
	
	double pi = dealii::numbers::PI;  // Werte für die exakte Lösung:u_MovHump
	double c_1 = (16./pi)*sin(pi*t); // *sin(pi*t) CHANGE TEST cos statt sin
	double c_2 = 2./(sqrt(epsilon->value(p, 0)));
	double c_3 = c_2*((1./16.)-((p(0)-(0.5))*(p(0)-(0.5)))-((p(1)-(0.5))*(p(1)-(0.5))));
	double xminxx = p(0)-(p(0)*p(0));
	double yminyy = p(1)-(p(1)*p(1));
	double xdc = ((c_2*((1.)-(2.*p(0))))/(1.+(c_3*c_3)));
	double ydc = ((c_2*((1.)-(2.*p(1))))/(1.+(c_3*c_3)));

// 	double u_x = c_1*(yminyy*((((pi/2.)+atan(c_3))*((1.)-(2.*p(0))))+
// 					(xminxx*xdc)));
// 	double u_y = c_1*(xminxx*((((pi/2.)+atan(c_3))*((1.)-(2.*p(1))))+
// 					(yminyy*ydc)));
	double dtu_E = (16.*cos(pi*t)*xminxx*yminyy*((pi/2.)+atan(c_3))); //CHANGE TEST -sin statt cos
	
	double u_xx = c_1*yminyy*(
					((-2.)*((pi/2.)+atan(c_3)))+(2.*(((1.)-(2.)*p(0))*xdc))+
					(xminxx*(((-8.*c_2*c_2*c_3*(p(0)-(0.5))*(p(0)-(0.5)))/((1.+c_3*c_3)*(1.+c_3*c_3)))-
					((2.*c_2)/(1.+c_3*c_3))))
					);
	double u_yy = c_1*xminxx*(
					((-2.)*((pi/2.)+atan(c_3)))+(2.*(((1.)-(2.)*p(1))*ydc))+
					(yminyy*(((-8.*c_2*c_2*c_3*(p(1)-(0.5))*(p(1)-(0.5)))/((1.+c_3*c_3)*(1.+c_3*c_3)))-
					((2.*c_2)/(1.+c_3*c_3))))
					);

	
	// conv.-diff.-eq. for u_MovingHump
	return_value = dtu_E - ((epsilon->value(p, 0))*(u_xx+u_yy));
					
	return return_value;
}



} // namespace

#include "Moving_Hump.inst.in"
