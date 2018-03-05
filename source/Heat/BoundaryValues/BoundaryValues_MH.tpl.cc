/**
 * @file BoundaryValues_MH.tpl.cc
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
#include <Heat/BoundaryValues/BoundaryValues_MH.tpl.hh>

// DEAL.II includes
#include <deal.II/base/function.h>

// C++ includes
#include <fstream>
#include <vector>

namespace Heat {

template <int dim>
BoundaryValues_MH<dim>::BoundaryValues_MH (std::shared_ptr< dealii::Function<dim> > _epsilon) 
	: 
	epsilon(_epsilon)
	{}
	
	
////////////////////////////////////////////////////////////////////////////////
//////////////// Moving Hump ///////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <int dim>
double
BoundaryValues_MH<dim>::
value (const dealii::Point<dim>   &p,
       const unsigned int  /*component*/) const
{
	double return_value = 0.0;
	
	// time variable t
	double t;
	t = this->get_time();
	
	double pi = dealii::numbers::PI;
	double c_1 = (16./pi)*sin(pi*t); // *sin(pi*t) CHANGE TEST cos statt sin
	double c_2 = 2./(sqrt(epsilon->value(p, 0)));
	double c_3 = c_2*((1./16.)-((p(0)-0.5)*(p(0)-0.5))-((p(1)-0.5)*(p(1)-0.5)));
	double xminxx = p(0)-(p(0)*p(0));
	double yminyy = p(1)-(p(1)*p(1));
	
	return_value = c_1*((xminxx)*(yminyy)*((pi/2.)+atan(c_3))); //u_MovHumpexakt
	
	return return_value;
}


template <int dim>
dealii::Tensor<1,dim> 
BoundaryValues_MH<dim>::
gradient (const dealii::Point<dim> &p,
		const unsigned int) const
{
	// time variable t
	double t;
	t = this->get_time();
	
	double pi = dealii::numbers::PI;
	double c_1 = (16./pi)*sin(pi*t); // *sin(pi*t) CHANGE TEST cos statt si
	double c_2 = 2./(sqrt(epsilon->value(p, 0)));
	double c_3 = c_2*((1./16.)-((p(0)-(0.5))*(p(0)-(0.5)))-((p(1)-(0.5))*(p(1)-(0.5))));
	double xminxx = p(0)-(p(0)*p(0));
	double yminyy = p(1)-(p(1)*p(1));
	double xdc = ((c_2*((1.)-(2.*p(0))))/(1.+(c_3*c_3)));
	double ydc = ((c_2*((1.)-(2.*p(1))))/(1.+(c_3*c_3)));
	
	dealii::Tensor<1, dim> return_value;
	
	return_value[0] = c_1*(yminyy*((((pi/2.)+atan(c_3))*((1.)-(2.*p(0))))+(xminxx*xdc)));
	return_value[1] = c_1*(xminxx*((((pi/2.)+atan(c_3))*((1.)-(2.*p(1))))+(yminyy*ydc)));
	
	return return_value;
}


} // namespace

#include "BoundaryValues_MH.inst.in"
