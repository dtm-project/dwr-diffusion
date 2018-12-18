/**
 * @file ControlVolume_moving_hyper_rectangle.tpl.cc
 * @author Uwe Koecher (UK)
 * 
 * @date 2018-12-13, UK
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

#include <diffusion/ControlVolume/ControlVolume_moving_hyper_rectangle.tpl.hh>

namespace diffusion {
namespace control_volume {

template<int dim>
double
moving_hyper_rectangle<dim>::
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
		
		// get \f$ \hat t \in [0, 1.0) \f$
		const double t_hat{t - std::floor(t)};
		const double omega{2.*M_PI};
		const double phi{omega*t_hat};
		
		std::array<double,3> Mc = {
			r1*std::cos(phi)+.5,
			r1*std::sin(phi)+.5,
			0. // NOTE: change this (z-component) to your needs.
		};
		
		for (unsigned int d{dim-1}; d > 0; --d) {
			result *=
				((x[d] >= (p1[d]+Mc[d])) && (x[d] <= (p2[d]+Mc[d]))) ? 1. : 0.;
		}
	}
	else {
		result = 0.;
	}
	
	return result;
}

}} //namespaces

#include "ControlVolume_moving_hyper_rectangle.inst.in"
