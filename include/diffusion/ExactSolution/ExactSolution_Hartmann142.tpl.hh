/**
 * @file ExactSolution_Hartmann142.tpl.hh
 * @author Uwe Koecher (UK)
 * 
 * @date 2018-07-20, Hartmann142, MPB, UK
 * @date 2018-03-09, UK
 * @date 2018-03-08, included from ewave, UK
 * @date 2017-10-25, UK
 * @date 2013-08-15, DTM++ v1, UK
 */

/*  Copyright (C) 2012-2018 by Uwe Koecher                                    */
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

#ifndef __ExactSolution_Hartmann142_tpl_hh
#define __ExactSolution_Hartmann142_tpl_hh

// DEAL.II includes
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

// C++ includes

namespace diffusion {
namespace exact_solution {

/**
 * Implements the analytic solution \f$ u : \Omega \times I \to \mathbb{R} \f$,
 * \f$ \Omega \subset \mathbb{R}^2 \f$, as given by:
 * \f[
 * u(x,y,t) :=
 * \frac{ 1 }{1+a\big(x-\frac{1}{2}-\frac{1}{4}\cos(2\pi t)\big)^2+
 * a\big(y-\frac{1}{2}-\frac{1}{4}\sin(2\pi t)\big)^2}\,,
 * \f]
 * with the parameter value \f$ a = 50 \f$ for example, found in the literature
 * reference: R. Hartmann: A-posteriori Fehlersch&auml;tzung und adaptive
 * Schrittweiten- und Ortsgittersteuerung bei Galerkin-Verfahren f&uuml;r die
 * W&auml;rmeleitungsgleichung. Sec. 1.4.2, p. 20, Diploma thesis,
 * Supervisor: Prof. Dr. R. Rannacher,
 * Faculty for Mathematics, University of Heidelberg, Germany, 1998.
 */
template<int dim>
class Hartmann142 : public dealii::Function<dim> {
public:
	Hartmann142(const double &a) : dealii::Function<dim> (1), a(a) {};
	
	virtual ~Hartmann142() = default;
	
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
