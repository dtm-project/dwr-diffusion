/**
 * @file BoundaryValues_Xquadrat.tpl.hh
 * @author Marius Paul Bruchhäuser (MPB), Uwe Koecher (UK)
 * @date 2017-08-22, MPB, UK
 *
 * @brief Heat Problem
*/

/*  Copyright (C) 2012-2017 by Marius Paul Bruchhäuser , Uwe Köcher           */
/*                                                                            */
/*  This file is part of DTM++/aWave. (Acoustic Wave Equation Solver)         */
/*                                                                            */
/*  DTM++/aWave is free software: you can redistribute it and/or modify       */
/*  it under the terms of the GNU Lesser General Public License as            */
/*  published by the Free Software Foundation, either                         */
/*  version 3 of the License, or (at your option) any later version.          */
/*                                                                            */
/*  DTM++/aWave is distributed in the hope that it will be useful,            */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/*  GNU Lesser General Public License for more details.                       */
/*                                                                            */
/*  You should have received a copy of the GNU Lesser General Public License  */
/*  along with DTM++/aWave.   If not, see <http://www.gnu.org/licenses/>.     */

#ifndef __BoundaryValues_Xquadrat_tpl_hh
#define __BoundaryValues_Xquadrat_tpl_hh

// PROJECT includes


// DEAL.II includes
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

// C++ includes
#include <fstream>
#include <vector>
#include <memory>

namespace Heat {
	
template<int dim>
class BoundaryValues_Xquadrat : public dealii::Function<dim> {
	public:
	BoundaryValues_Xquadrat (std::shared_ptr< dealii::Function<dim> > epsilon);
	virtual ~BoundaryValues_Xquadrat() = default;
	
	virtual double value (
		const dealii::Point<dim> &p,
		const unsigned int component = 0
	) const;
	virtual dealii::Tensor<1,dim> gradient (const dealii::Point<dim> &p,
											const unsigned int component = 0)
											const;
	
	private:
	std::shared_ptr< dealii::Function<dim> > epsilon;	
};


} // namespace
#endif
