/**
 * @file NeumannBoundary_KoecherBruchhaeuser2a.tpl.hh
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * @date 2018-11-15, UK
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

#ifndef __NeumannBoundary_KoecherBruchhaeuser2a_tpl_hh
#define __NeumannBoundary_KoecherBruchhaeuser2a_tpl_hh

// DEAL.II includes
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

namespace diffusion {
namespace neumann_boundary {

/**
 * Implements the analytic solution \f$ h : \Omega \times I \to \mathbb{R} \f$,
 * \f$ \Omega \subset \mathbb{R}^2 \f$, as given by:
 * \f[
 * h(x,y,t) := \varepsilon(\boldsymbol x) \nabla u(x,y,t)\cdot\boldsymbol n
 * \f]
 * where \f$ u(x,y,t) \f$ denotes the exact analytical solution
 * and \f$ \boldsymbol n \f$ denotes the outward directed normal vector.
 */
template<int dim>
class KoecherBruchhaeuser2a : public dealii::Function<dim> {
public:
	KoecherBruchhaeuser2a(
		const double &s,
		const double &a,
		const double &epsilon
	) : dealii::Function<dim> (1), s(s), a(a), epsilon(epsilon)
	{};
	
	virtual ~KoecherBruchhaeuser2a() = default;
	
	virtual
	double
	value(
		const dealii::Point<dim> &x,
		const unsigned int c
	) const;

private:
	const double s;
	const double a;
	const double epsilon;
};

}}

#endif
