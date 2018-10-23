/**
 * @file ExactSolution_KoecherBruchhaeuser2.tpl.hh
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * @date 2018-10-23, UK
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

#ifndef __ExactSolution_KoecherBruchhaeuser2_tpl_hh
#define __ExactSolution_KoecherBruchhaeuser2_tpl_hh

// DEAL.II includes
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

namespace heat {
namespace exact_solution {

template<int dim>
class KoecherBruchhaeuser2 : public dealii::Function<dim> {
public:
	KoecherBruchhaeuser2(
		const double &s,
		const double &a
	) : dealii::Function<dim> (1), s(s), a(a)
	{};
	
	virtual ~KoecherBruchhaeuser2() = default;
	
	virtual
	double
	value(
		const dealii::Point<dim> &x,
		const unsigned int c
	) const;

private:
	const double s;
	const double a;
};

}}

#endif
