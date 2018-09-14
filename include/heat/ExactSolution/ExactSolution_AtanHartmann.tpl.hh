/**
 * @file ExactSolution_AtanHartmann.tpl.hh
 * @author Marius Paul Bruchhaeuser (MPB)
 * @author Uwe Koecher (UK)
 * 
 * @date 2018-09-14, MPB, UK
 * @date 2018-03-09, UK
 * @date 2018-03-08, included from ewave, UK
 * @date 2017-10-25, UK
 * @date 2013-08-15, DTM++ v1, UK
 */

/*  Copyright (C) 2012-2018 by Uwe Koecher, Marius Paul Bruchhaeuser          */
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

#ifndef __ExactSolution_AtanHartmann_tpl_hh
#define __ExactSolution_AtanHartmann_tpl_hh

// DEAL.II includes
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

// C++ includes

namespace heat {
namespace exact_solution {

template<int dim>
class AtanHartmann : public dealii::Function<dim> {
public:
	AtanHartmann(const double &a) : dealii::Function<dim> (1), a(a) {};
	
	virtual ~AtanHartmann() = default;
	
	/// get value (of a specific component) from a function evaluation
	virtual
	double
	value(
		const dealii::Point<dim> &x,
		const unsigned int c
	) const;

private:
	const double a;
};

}}

#endif
