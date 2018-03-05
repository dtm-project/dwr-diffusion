/**
 * @file Testfunktion.tpl.cc
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
#include <Heat/Forces/Testfunktion.tpl.hh>

// DEAL.II includes
#include <deal.II/base/function.h>

// C++ includes
#include <fstream>
#include <vector>

namespace Heat {

template <int dim>
Testfunktion<dim>::Testfunktion (std::shared_ptr< dealii::Function<dim> > _epsilon,
				   std::shared_ptr< dealii::Function<dim> > _BoundaryValues) 
	: 
	epsilon(_epsilon),
	BoundaryValues(_BoundaryValues)
	{}

////////////////////////////////////////////////////////////////////////////////
//////////////// Moving Hump ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <int dim>
double Testfunktion<dim>::value (
	const dealii::Point<dim> &p,
	const unsigned int /* component */
) const {
	double return_value = 0.0;
	double pi = dealii::numbers::PI;
	// time variable t
	double t;
	t = this->get_time();
	
// 	double u_x = (2.*(p(1)*p(1))*p(0))-(2.*p(1)*p(0))-(p(1)*p(1))+p(1);
// 	double u_y = (2.*(p(0)*p(0))*p(1))-(2.*p(0)*p(1))-(p(0)*p(0))+p(0);
// 	double dtu_E = 3.*((t+1)*(t+1))*(((p(0)*p(0))-p(0))*((p(1)*p(1))-p(1))); //CHANGE TEST -sin statt cos
// 	double dtu_E = 3.*((t+1)*(t+1))*(sin(2.*pi*p(0))*sin(2.*pi*p(1))*cos(2.*pi*p(0))*cos(2.*pi*p(1)));
	double dtu_E = /*3.*((t+1)*(t+1))*/2.*pi*cos(2.*pi*t)*((1-(p(0)*p(0)))*(1-(p(1)*p(1)))*sin(4.*p(0))*sin(4.*p(1)));
// 	double u_xx = ((t+1)*(t+1)*(t+1))*((2.*(p(1)*p(1)))-(2.*p(1)));
// 	double u_yy = ((t+1)*(t+1)*(t+1))*((2.*(p(0)*p(0)))-(2.*p(0)));
// 	double u_xx = -4.*pi*pi*sin(2.*pi*p(0))*sin(2.*pi*p(1));
// 	double u_yy = -4.*pi*pi*sin(2.*pi*p(0))*sin(2.*pi*p(1));
// 	double u_xx = ((t+1)*(t+1)*(t+1))*
// 	(-16.*pi*pi*sin(2.*pi*p(0))*sin(2.*pi*p(1))*cos(2.*pi*p(0))*cos(2.*pi*p(1)));
// 	double u_yy = ((t+1)*(t+1)*(t+1))*
// 	(-16.*pi*pi*sin(2.*pi*p(0))*sin(2.*pi*p(1))*cos(2.*pi*p(0))*cos(2.*pi*p(1)));
	double u_xx = /*((t+1)*(t+1)*(t+1))*/sin(2.*pi*t)*(
					(1-(p(1)*p(1)))*sin(4.*p(1))*(-16.*p(0)*cos(4.*p(0))
					-16.*(1-(p(0)*p(0)))*sin(4.*p(0))
					-2.*sin(4.*p(0))));
	double u_yy = /*((t+1)*(t+1)*(t+1))*/sin(2.*pi*t)*(
					(1-(p(0)*p(0)))*sin(4.*p(0))*(-16.*p(1)*cos(4.*p(1))
					-16.*(1-(p(1)*p(1)))*sin(4.*p(1))
					-2.*sin(4.*p(1))));
	
	// conv.-diff.-eq. for u_Testfunktion
	return_value = dtu_E - ((epsilon->value(p, 0))*(u_xx+u_yy));
					
	return return_value;
}



} // namespace

#include "Testfunktion.inst.in"
