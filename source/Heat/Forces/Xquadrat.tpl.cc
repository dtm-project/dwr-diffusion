/**
 * @file Xquadrat.tpl.cc
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
#include <Heat/Forces/Xquadrat.tpl.hh>

// DEAL.II includes
#include <deal.II/base/function.h>

// C++ includes
#include <fstream>
#include <vector>

namespace Heat {

template <int dim>
Xquadrat<dim>::Xquadrat (std::shared_ptr< dealii::Function<dim> > _epsilon) 
	: 
	epsilon(_epsilon)
	{}

////////////////////////////////////////////////////////////////////////////////
//////////////// Moving Hump ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <int dim>
double Xquadrat<dim>::value (
	const dealii::Point<dim> &p,
	const unsigned int /* component */
) const {
	double return_value = 0.0;
	double x = p(0); // just for better reading aspects
	double y = p(1); // just for better reading aspects

	double u_xx = 2.*((y*y)-y);
	double u_yy = 2.*((x*x)-x);


	// diff.-eq. for u_Sinus_Sinus
	return_value =  -((epsilon->value(p, 0))*(u_xx+u_yy));
					
	return return_value;
}



} // namespace

#include "Xquadrat.inst.in"
