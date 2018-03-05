/**
 * @file Tangens_hyperbolicus.tpl.cc
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
#include <Heat/Forces/Tangens_hyperbolicus.tpl.hh>

// DEAL.II includes
#include <deal.II/base/function.h>

// C++ includes
#include <fstream>
#include <vector>

namespace Heat {

template <int dim>
Tangens_hyperbolicus<dim>::Tangens_hyperbolicus (std::shared_ptr< dealii::Function<dim> > _epsilon,
				   std::shared_ptr< dealii::Function<dim> > _BoundaryValues) 
	: 
	epsilon(_epsilon),
	BoundaryValues(_BoundaryValues)
	{}


////////////////////////////////////////////////////////////////////////////////
/////////////// Tangens Hyperbolicus ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <int dim>
double Tangens_hyperbolicus<dim>::value (
	const dealii::Point<dim> &p,
	const unsigned int /* component */
) const {
	double return_value = 0.0;
	double c_1 = (1./(sqrt(5.*epsilon->value(p, 0))));
	double c_2 = (1./(5.*epsilon->value(p, 0)));
	double tanhyp = tanh((2*p(0)-p(1)-0.25)*c_1);

// 	double u_exact = 0.5*(1.-tanhyp);
// 	double u_x = -c_1*(1-tanhyp*tanhyp);
// 	double u_y = 0.5*c_1*(1-tanhyp*tanhyp);
	double u_xx = 4.*c_2*tanhyp*(1-tanhyp*tanhyp);
	double u_yy = c_2*tanhyp*(1-tanhyp*tanhyp);
	
	return_value = -((epsilon->value(p, 0))*(u_xx+u_yy));	// -grad(epsilon*grad(u))

	

	return return_value;
}

} // namespace

#include "Tangens_hyperbolicus.inst.in"
