/**
 * @file slabs.tpl.hh
 * 
 * @author Uwe Koecher (UK)
 * 
 * @date 2018-03-06, UK
 * 
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


#ifndef __slabs_hh
#define __slabs_hh

// DEAL.II includes
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

namespace DTM {
namespace types {
namespace spacetime {
namespace dwr {

/// slab: collects data structures and functions of a space-time slab for dwr
template <int dim>
struct s_slab {
	std::shared_ptr< dealii::Triangulation<dim> > tria;
	double t_m; ///< left endpoint of \f$ I_n=(t_m, t_n) \f$
	double t_n; ///< right endpoint of \f$ I_n=(t_m, t_n) \f$
	
	struct {
		std::shared_ptr< dealii::DoFHandler<dim> > dof;
		std::shared_ptr< dealii::FiniteElement<dim> > fe;
		std::shared_ptr< dealii::Mapping<dim> >  mapping;
		
		std::shared_ptr< dealii::ConstraintMatrix > constraints;
		std::shared_ptr< dealii::SparsityPattern > sp;
	} primal;
	
	struct {
		std::shared_ptr< dealii::DoFHandler<dim> > dof;
		std::shared_ptr< dealii::FiniteElement<dim> > fe;
		std::shared_ptr< dealii::Mapping<dim> >  mapping;
		
		std::shared_ptr< dealii::ConstraintMatrix > constraints;
		std::shared_ptr< dealii::SparsityPattern > sp;
	} dual;
	
	double tau_n() const { return (t_n-t_m); };
};

template <int dim>
using slab = struct s_slab<dim>;

template <int dim>
using slabs = std::list< slab<dim> >;

}}}}

#endif
