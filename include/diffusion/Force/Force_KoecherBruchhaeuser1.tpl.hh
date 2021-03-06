/**
 * @file Force_KoecherBruchhaeuser1.tpl.hh
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * @date 2018-10-23, UK
 * @date 2018-09-14, MPB
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

#ifndef __Force_KoecherBruchhaeuser1_tpl_hh
#define __Force_KoecherBruchhaeuser1_tpl_hh

// DEAL.II includes
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

namespace diffusion {
namespace force {

template<int dim>
class KoecherBruchhaeuser1 : public dealii::Function<dim> {
public:
	KoecherBruchhaeuser1(
		const double &s,
		const double &a,
		const double &rho,
		const double &epsilon
	) : dealii::Function<dim> (1), s(s), a(a), rho(rho), epsilon(epsilon)
	{ };
	
	virtual ~KoecherBruchhaeuser1() = default;
	
	virtual
	double
	value(
		const dealii::Point<dim> &x,
		const unsigned int c
	) const;

private:
	const double s;
	const double a;
	const double rho;
	const double epsilon;
};

}}

#endif
