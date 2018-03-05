/**
 * @file Function.tpl.hh
 * @author Uwe Koecher (UK)
 * @date 2015-02-23, UK
 * @date 2014-11-19, UK
 *
 * @brief Abstract function interface IR^d x IR -> IR^s
 */

/*  Copyright (C) 2012-2015 by Uwe Koecher                                    */
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
/*  along with DTM++.  If not, see <http://www.gnu.org/licenses/>.            */

#ifndef __Function_tpl_hh
#define __Function_tpl_hh

// MPI includes

// DEAL.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>

// C++ includes
#include <vector>

namespace DTM {

template<int dim, int n_components>
class Function : public dealii::Function<dim> {
public:
	Function() : dealii::Function<dim> (n_components) {}
	
	virtual ~Function() = default;
	
	/// get value (of a specific component) from a function evaluation
	virtual
	double
	value(
		const dealii::Point<dim> &x,
		const unsigned int c = 0
	) const;
	
	/// get values (of a specific component) from multiple function evaluations
	virtual
	void
	value_list(
		const std::vector<dealii::Point<dim> > &,
		std::vector<double> &,
		const unsigned int c
	) const;
	
	/// get vector-value (of all components) from a function evaluation
	virtual
	void
	vector_value(
		const dealii::Point<dim> &,
		dealii::Vector<double> &
	) const {
		Assert(false, dealii::ExcNotImplemented());
	}
	
	/// get vector-values (of all components) from multiple function evaluations
	virtual
	void
	vector_value_list(
		const std::vector< dealii::Point<dim> > &,
		std::vector< dealii::Vector<double> > &
	) const;
};


} // namespace
#endif
