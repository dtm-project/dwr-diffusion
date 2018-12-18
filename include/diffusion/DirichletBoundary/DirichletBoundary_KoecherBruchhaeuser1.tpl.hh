/**
 * @file DirichletBoundary_KoecherBruchhaeuser1.tpl.hh
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

#ifndef __DirichletBoundary_KoecherBruchhaeuser1_tpl_hh
#define __DirichletBoundary_KoecherBruchhaeuser1_tpl_hh

// DEAL.II includes
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

namespace diffusion {
namespace dirichlet_boundary {

/**
 * Implements the analytic solution \f$ u : \Omega \times I \to \mathbb{R} \f$,
 * \f$ \Omega \subset \mathbb{R}^2 \f$, as given by:
 * \f[
 * u(x,y,t) :=
 * s \frac{\arctan(10\frac{\pi}{2}(2t-1))}{1+a\big(x-\frac{1}{2}-\frac{1}{4}\cos(2\pi t)\big)^2+
 * a\big(y-\frac{1}{2}-\frac{1}{4}\sin(2\pi t)\big)^2}\,,
 * \f]
 * with the parameter values \f$ s =\frac{1}{3} \f$ and \f$ a = 50 \f$ for example.
 */
template<int dim>
class KoecherBruchhaeuser1 : public dealii::Function<dim> {
public:
	KoecherBruchhaeuser1(
		const double &s,
		const double &a
	) : dealii::Function<dim> (1), s(s), a(a)
	{};
	
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
};

}}

#endif
