/**
 * @file Sinus_Sinus.tpl.cc
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
#include <Heat/Forces/Sinus_Sinus.tpl.hh>

// DEAL.II includes
#include <deal.II/base/function.h>

// C++ includes
#include <fstream>
#include <vector>

namespace Heat {

template <int dim>
Sinus_Sinus<dim>::Sinus_Sinus (std::shared_ptr< dealii::Function<dim> > _epsilon) 
	: 
	epsilon(_epsilon)
	{}

////////////////////////////////////////////////////////////////////////////////
//////////////// Moving Hump ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <int dim>
double Sinus_Sinus<dim>::value (
	const dealii::Point<dim> &p,
	const unsigned int /* component */
) const {
	double return_value = 0.0;
	// time variable t
	double t;
	t = this->get_time();
	
	double pi = dealii::numbers::PI;  // Werte für die exakte Lösung:u_Sinus_Sinus

	double dtu = 2.*t*sin(2.*pi*p(0))*sin(2.*pi*p(1)); 
	
	double u_xx = (t*t)*(-4.)*pi*pi*sin(2.*pi*p(0))*sin(2.*pi*p(1));
	double u_yy = (t*t)*(-4.)*pi*pi*sin(2.*pi*p(0))*sin(2.*pi*p(1));


	// diff.-eq. for u_Sinus_Sinus
	return_value =  -dtu - ((epsilon->value(p, 0))*(u_xx+u_yy));
					
	return return_value;
}



} // namespace

#include "Sinus_Sinus.inst.in"
