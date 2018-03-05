/**
 * @file BoundaryValues_Hartmann.tpl.cc
 * @author Uwe Koecher (UK), Marius Paul Bruchhäuser (MPB)
 * @date 2017-08-22, UK
 *
 * @brief Heat Problem
 */

/*  Copyright (C) 2012-2017 by Uwe Köcher, Marius Paul Bruchhäuser            */
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
#include <Heat/BoundaryValues/BoundaryValues_Hartmann.tpl.hh>

// DEAL.II includes
#include <deal.II/base/function.h>

// C++ includes
#include <fstream>
#include <vector>

namespace Heat {

template <int dim>
BoundaryValues_Hartmann<dim>::BoundaryValues_Hartmann (std::shared_ptr< dealii::Function<dim> > _alpha) 
	: 
	alpha(_alpha)
	{}
	
	
////////////////////////////////////////////////////////////////////////////////
//////////////// Moving Hump ///////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <int dim>
double
BoundaryValues_Hartmann<dim>::
value (const dealii::Point<dim>   &p,
       const unsigned int  /*component*/) const
{
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
	
	return_value = 1./Nenner; //u_Hartmann exakt (Diplom thesis p. 20)
	
	return return_value;
}


template <int dim>
dealii::Tensor<1,dim> 
BoundaryValues_Hartmann<dim>::
gradient (const dealii::Point<dim> &p,
		const unsigned int) const
{
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
	
	dealii::Tensor<1, dim> return_value;
	
	return_value[0] = (-2.*alpha->value(p, 0)*xminx_0t)/(Nenner*Nenner);
	return_value[1] = (-2.*alpha->value(p, 0)*yminy_0t)/(Nenner*Nenner);
	
	return return_value;
}


} // namespace

#include "BoundaryValues_Hartmann.inst.in"
