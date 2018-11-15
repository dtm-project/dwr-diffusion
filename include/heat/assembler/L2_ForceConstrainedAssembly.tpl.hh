/**
 * @file L2_ForceConstrainedAssembly.tpl.hh
 * @author Uwe Koecher (UK)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 * 
 * @date 2018-03-08, included from ewave, UK
 * @date 2017-10-26, auto mode, UK
 * @date 2017-09-13, xwave/ewave, UK
 * @date 2015-05-19, AWAVE/C++.11, UK
 * @date 2012-08-31
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


#ifndef __L2_ForceConstrainedAssembly_tpl_hh
#define __L2_ForceConstrainedAssembly_tpl_hh

// DEAL.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/vector.h>

// C++ includes
#include <iterator>
#include <functional>
#include <memory>
#include <vector>


namespace heat {
namespace Assemble {
namespace L2 {
namespace ForceConstrained {

namespace Assembly {
namespace Scratch {

template<int dim>
struct ForceConstrainedAssembly {
	ForceConstrainedAssembly(
		const dealii::FiniteElement<dim> &fe,
		const dealii::Mapping<dim> &mapping,
		const dealii::Quadrature<dim> &quad,
		const dealii::UpdateFlags &uflags
	);
	
	ForceConstrainedAssembly(const ForceConstrainedAssembly &scratch);
	
	dealii::FEValues<dim> fe_values;
	std::vector<double>   phi;
	double                JxW;
	double                f;
	
	// other
	unsigned int q;
	unsigned int component;
	unsigned int k;
	unsigned int i;
};

} // namespace Scratch
namespace CopyData {

template<int dim>
struct ForceConstrainedAssembly {
	ForceConstrainedAssembly(const dealii::FiniteElement<dim> &fe);
	ForceConstrainedAssembly(const ForceConstrainedAssembly &copydata);
	
	dealii::Vector<double> fi_vi_vector;
	std::vector<unsigned int> local_dof_indices;
};

} // namespace CopyData
} // namespace Assembly
////////////////////////////////////////////////////////////////////////////////


template<int dim>
class Assembler {
public:
	Assembler(
		std::shared_ptr< dealii::Vector<double> > f,
		std::shared_ptr< dealii::DoFHandler<dim> > dof,
		std::shared_ptr< dealii::FiniteElement<dim> > fe,
		std::shared_ptr< dealii::Mapping<dim> > mapping,
		std::shared_ptr< dealii::ConstraintMatrix > constraints
	);
	
	~Assembler() = default;
	
	void set_function(
		std::shared_ptr< dealii::Function<dim> > force
	);
	
	/** Assemble vector.
	 *  If @param n_quadrature_points = 0 is given,
	 *  the dynamic default fe.tensor_degree()+1 will be used.
	 */
	void assemble(
		const double time,
		const unsigned int n_quadrature_points = 0,
		const bool quadrature_points_auto_mode = true
	);
	
protected:
	void local_assemble_cell(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		Assembly::Scratch::ForceConstrainedAssembly<dim> &scratch,
		Assembly::CopyData::ForceConstrainedAssembly<dim> &copydata
	);

	void copy_local_to_global_cell(
		const Assembly::CopyData::ForceConstrainedAssembly<dim> &copydata
	);
	
private:
	std::shared_ptr< dealii::Vector<double> > f;
	
	std::shared_ptr< dealii::DoFHandler<dim> > dof;
	std::shared_ptr< dealii::FiniteElement<dim> > fe;
	std::shared_ptr< dealii::Mapping<dim> > mapping;
	std::shared_ptr< dealii::ConstraintMatrix > constraints;
	dealii::UpdateFlags uflags;
	
	struct {
		std::shared_ptr< dealii::Function<dim> > f;
	} function;
};

}}}} // namespaces

#endif
