/**
 * @file ControlVolume_hyper_rectangle.tpl.cc
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * @date 2018-11-19, UK
 */

/*  Copyright (C) 2012-2018 by Uwe Koecher and contributors                   */
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

#include <diffusion/ControlVolume/ControlVolume_hyper_rectangle.tpl.hh>

namespace diffusion {
namespace control_volume {

template<int dim>
double
hyper_rectangle<dim>::
value(
	const dealii::Point<dim> &x,
	[[maybe_unused]]const unsigned int c
) const {
	Assert(
		c==0,
		dealii::ExcMessage(
			"you want to get component value which is not implemented"
		)
	);
	
	double result{1.0};
	
	const double t{this->get_time()};
	
	if ((t >= t1) && (t <= t2)) {
		// time window of \Omega_c x I_c
		
		switch (dim) {
		case 3:
			result *= ((x[2] >= p1[2]) && (x[2] <= p2[2])) ? 1. : 0.;
			[[fallthrough]];
		
		case 2:
			result *= ((x[1] >= p1[1]) && (x[1] <= p2[1])) ? 1. : 0.;
			[[fallthrough]];
			
		case 1:
			result *= ((x[0] >= p1[0]) && (x[0] <= p2[0])) ? 1. : 0.;
			break;
			
		default:
			Assert(false, dealii::ExcInvalidState());
		}
	}
	else {
		result = 0.;
	}
	
	return result;
}

}} //namespaces

#include "ControlVolume_hyper_rectangle.inst.in"
