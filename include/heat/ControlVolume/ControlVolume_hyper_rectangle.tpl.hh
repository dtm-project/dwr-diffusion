/**
 * @file ControlVolume_hyper_rectangle.tpl.hh
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

#ifndef __ControlVolume_hyper_rectangle_tpl_hh
#define __ControlVolume_hyper_rectangle_tpl_hh

// DEAL.II includes
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

namespace heat {
namespace control_volume {

/**
 * Control volume \f$ \mathcal{Q}_c:=\Omega_c \times I_c \f$, with \f$ \Omega_c \subseteq \Omega \f$
 * and \f$ I_c \subseteq I \f$.
 */
template<int dim>
class hyper_rectangle : public dealii::Function<dim> {
public:
	hyper_rectangle(
		const dealii::Point<dim> &p1,
		const dealii::Point<dim> &p2,
		const double &t1,
		const double &t2
	) : dealii::Function<dim> (1), p1(p1), p2(p2), t1(t1), t2(t2)
	{};
	
	virtual ~hyper_rectangle() = default;
	
	virtual
	double
	value(
		const dealii::Point<dim> &x,
		const unsigned int c
	) const;

private:
	const dealii::Point<dim> p1;
	const dealii::Point<dim> p2;
	
	const double t1;
	const double t2;
};

}}

#endif
