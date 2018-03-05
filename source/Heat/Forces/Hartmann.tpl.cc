/**
 * @file Hartmann.tpl.cc
 * @author Marius Paul Bruchhäuser (MPB), Uwe Koecher (UK)
 * @date 2017-08-22, MPB, UK
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
#include <Heat/Forces/Hartmann.tpl.hh>

// DEAL.II includes
#include <deal.II/base/function.h>

// C++ includes
#include <fstream>
#include <vector>

namespace Heat {

template <int dim>
Hartmann<dim>::Hartmann (std::shared_ptr< dealii::Function<dim> > _epsilon,
						 std::shared_ptr< dealii::Function<dim> > _alpha,
						 std::shared_ptr< dealii::Function<dim> > _BoundaryValues) 
	: 
	epsilon(_epsilon),
	alpha(_alpha),
	BoundaryValues(_BoundaryValues)
	{}

////////////////////////////////////////////////////////////////////////////////
//////////////// Hartmann ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <int dim>
double Hartmann<dim>::value (
	const dealii::Point<dim> &p,
	const unsigned int /* component */
) const {
	double return_value = 0.0;
	
	// time variable t
	double t;
	t = this->get_time();
	
	double x = p(0);
	double y = p(1);
	double pi = dealii::numbers::PI;
	double x_0t = 0.5+0.25*cos(2.*pi*t); 
	double y_0t = 0.5+0.25*sin(2.*pi*t);
	double xminx_0t = (x-x_0t);
	double yminy_0t = (y-y_0t);
	double Nenner = (1.+alpha->value(p, 0)*((xminx_0t*xminx_0t)+(yminy_0t*yminy_0t)));
	
	double dtu = -((alpha->value(p, 0)*xminx_0t*pi*sin(2.*pi*t))
				-(alpha->value(p, 0)*yminy_0t*pi*cos(2.*pi*t)))/
				(Nenner*Nenner);

	double u_xx = 	((-2.*alpha->value(p, 0)*(Nenner*Nenner))
					+(2.*alpha->value(p, 0)*xminx_0t)*
					((4.*alpha->value(p, 0)*xminx_0t)
						+(4.*(alpha->value(p, 0)*alpha->value(p, 0))*(xminx_0t*xminx_0t*xminx_0t))
						+(4.*(alpha->value(p, 0)*alpha->value(p, 0))*(yminy_0t*yminy_0t)*xminx_0t))
					)/
					(Nenner*Nenner*Nenner*Nenner);
	double u_yy = 	((-2.*alpha->value(p, 0)*(Nenner*Nenner))
					+(2.*alpha->value(p, 0)*yminy_0t)*
					((4.*alpha->value(p, 0)*yminy_0t)
						+(4.*(alpha->value(p, 0)*alpha->value(p, 0))*(yminy_0t*yminy_0t*yminy_0t))
						+(4.*(alpha->value(p, 0)*alpha->value(p, 0))*(xminx_0t*xminx_0t)*yminy_0t))
					)/
					(Nenner*Nenner*Nenner*Nenner);

	
	return_value = dtu-((epsilon->value(p, 0))*(u_xx+u_yy));
					
	return return_value;
}



} // namespace

#include "Hartmann.inst.in"
