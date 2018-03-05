/**
 * @file BoundaryValues_Sinus.tpl.cc
 * @author Uwe Koecher (UK), Marius Paul Bruchhäuser (MPB)
 * @date 2017-08-01, UK
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
#include <Heat/BoundaryValues/BoundaryValues_Sinus.tpl.hh>

// DEAL.II includes
#include <deal.II/base/function.h>

// C++ includes
#include <fstream>
#include <vector>

namespace Heat {

template <int dim>
BoundaryValues_Sinus<dim>::BoundaryValues_Sinus (std::shared_ptr< dealii::Function<dim> > _epsilon) 
	: 
	epsilon(_epsilon)
	{}
	
	
////////////////////////////////////////////////////////////////////////////////
//////////////// Moving Hump ///////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <int dim>
double
BoundaryValues_Sinus<dim>::
value (const dealii::Point<dim>   &p,
       const unsigned int  /*component*/) const
{
	double return_value = 0.0;
	// time variable t
	double t;
	t = this->get_time();
	
	double pi = dealii::numbers::PI;

	
	return_value = (t*t)*sin(2.*pi*p(0))*sin(2.*pi*p(1));
	
	return return_value;
}


template <int dim>
dealii::Tensor<1,dim> 
BoundaryValues_Sinus<dim>::
gradient (const dealii::Point<dim> &p,
		const unsigned int) const
{
	// time variable t
	double t;
	t = this->get_time();
	
	double pi = dealii::numbers::PI;

	
	dealii::Tensor<1, dim> return_value;
	
	return_value[0] = (t*t)*2.*pi*sin(2.*pi*p(1))*cos(2.*pi*p(0));
	return_value[1] = (t*t)*2.*pi*sin(2.*pi*p(0))*cos(2.*pi*p(1));
	
	return return_value;
}


} // namespace

#include "BoundaryValues_Sinus.inst.in"
