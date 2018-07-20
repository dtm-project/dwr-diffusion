/**
 * @file L2_ForceConstrainedAssembly.tpl.cc
 * @author Uwe Koecher (UK)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 * 
 * @date 2018-03-08, included from ewave, UK
 * @date 2017-10-26, auto mode, UK
 * @date 2017-09-13, xwave/ewave, UK
 * @date 2015-05-19, AWAVE/C++.11, UK
 * @date 2012-08-31
 *
 * @brief Purpose: Assemble L2-function.
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

// PROJECT includes
#include <heat/assembler/L2_ForceConstrainedAssembly.tpl.hh>

namespace heat {
namespace Assemble {
namespace L2 {
namespace ForceConstrained {

namespace Assembly {
namespace Scratch {

/// (Struct-) Constructor.
template<int dim>
ForceConstrainedAssembly<dim>::ForceConstrainedAssembly(
	const dealii::FiniteElement<dim> &fe,
	const dealii::Mapping<dim> &mapping,
	const dealii::Quadrature<dim> &quad,
	const dealii::UpdateFlags &uflags) :
	fe_values(mapping, fe, quad, uflags),
	phi(fe.dofs_per_cell),
	JxW(0),
	f(0) {
}


/// (Struct-) Copy constructor.
template<int dim>
ForceConstrainedAssembly<dim>::ForceConstrainedAssembly(
	const ForceConstrainedAssembly &scratch) :
	fe_values(
		scratch.fe_values.get_mapping(),
		scratch.fe_values.get_fe(),
		scratch.fe_values.get_quadrature(),
		scratch.fe_values.get_update_flags()),
	phi(scratch.phi),
	JxW(scratch.JxW),
	f(scratch.f),
	q(scratch.q),
	component(scratch.component),
	k(scratch.k),
	i(scratch.i) {
}

}

namespace CopyData {

/// (Struct-) Constructor.
template<int dim>
ForceConstrainedAssembly<dim>::ForceConstrainedAssembly(
	const dealii::FiniteElement<dim> &fe) :
	fi_vi_vector(fe.dofs_per_cell),
	local_dof_indices(fe.dofs_per_cell) {
}


/// (Struct-) Copy constructor.
template<int dim>
ForceConstrainedAssembly<dim>::ForceConstrainedAssembly(
	const ForceConstrainedAssembly& copydata) :
	fi_vi_vector(copydata.fi_vi_vector),
	local_dof_indices(copydata.local_dof_indices) {
}

}}
////////////////////////////////////////////////////////////////////////////////


/// Constructor.
template<int dim>
Assembler<dim>::
Assembler(
	std::shared_ptr< dealii::Vector<double> > f,
	std::shared_ptr< dealii::DoFHandler<dim> > dof,
	std::shared_ptr< dealii::FiniteElement<dim> > fe,
	std::shared_ptr< dealii::Mapping<dim> > mapping,
	std::shared_ptr< dealii::ConstraintMatrix > constraints) :
	f(f),
	dof(dof),
	fe(fe),
	mapping(mapping),
	constraints(constraints) {
	// init update flags
	uflags =
		dealii::update_quadrature_points |
		dealii::update_values |
		dealii::update_JxW_values;
}


template<int dim>
void
Assembler<dim>::set_function(
	std::shared_ptr< dealii::Function<dim> > fun) {
	function.f = fun;
}


template<int dim>
void Assembler<dim>::assemble(
	const double time,
	const unsigned int q,
	const bool quadrature_points_auto_mode) {
	// check
	Assert( function.f.use_count(), dealii::ExcNotInitialized() );
	function.f->set_time(time);
	
	// assemble matrix
	const dealii::QGauss<dim> quad{
		(quadrature_points_auto_mode ? (fe->tensor_degree()+1) : q)
	};
	
	if (!quad.size()) {
		return;
	}
	
	typedef dealii::FilteredIterator<
		const typename dealii::DoFHandler<dim>::active_cell_iterator
	> CellFilter;
	
	// Using WorkStream to assemble.
	dealii::WorkStream::
	run(
		CellFilter(
			dealii::IteratorFilters::LocallyOwnedCell(), dof->begin_active()
		),
		CellFilter(
			dealii::IteratorFilters::LocallyOwnedCell(), dof->end()
		),
		std::bind (
			&Assembler<dim>::local_assemble_cell,
			this,
			std::placeholders::_1,
			std::placeholders::_2,
			std::placeholders::_3
		),
		std::bind (
			&Assembler<dim>::copy_local_to_global_cell,
			this,
			std::placeholders::_1
		),
		Assembly::Scratch::ForceConstrainedAssembly<dim> (*fe, *mapping, quad, uflags),
		Assembly::CopyData::ForceConstrainedAssembly<dim> (*fe)
	);
}


/// Local assemble on cell.
template<int dim>
void Assembler<dim>::local_assemble_cell(
	const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
	Assembly::Scratch::ForceConstrainedAssembly<dim> &scratch,
	Assembly::CopyData::ForceConstrainedAssembly<dim> &copydata) {
	
	// reinit scratch and data to current cell
	scratch.fe_values.reinit(cell);
	cell->get_dof_indices(copydata.local_dof_indices);
	
	// initialize local matrix with zeros
	copydata.fi_vi_vector = 0;
	
	// assemble cell terms
	for (scratch.q=0; scratch.q < scratch.fe_values.n_quadrature_points; ++scratch.q) {
		scratch.JxW = scratch.fe_values.JxW(scratch.q);
		
		// loop over all components of this finite element
		for (scratch.component=0;
			scratch.component < scratch.fe_values.get_fe().n_components();
			++scratch.component) {
			scratch.f = function.f->value(
				scratch.fe_values.quadrature_point(scratch.q),
				scratch.component
			);
			
			// loop over all basis functions to get the shape values
			for (scratch.k=0; scratch.k < scratch.fe_values.get_fe().dofs_per_cell;
				++scratch.k) {
				scratch.phi[scratch.k] =
					scratch.fe_values.shape_value_component(
						scratch.k,
						scratch.q,
						scratch.component
					);
			}
			
			// loop over all basis function combinitions to get the assembly
			for (scratch.i=0; scratch.i < scratch.fe_values.get_fe().dofs_per_cell;
				 ++scratch.i) {
				copydata.fi_vi_vector[scratch.i] += (
					scratch.f *
					scratch.phi[scratch.i] *
					scratch.JxW
				);
			} // for i
		} // for component
	} // for q
}


/// Copy local assembly to global matrix.
template<int dim>
void Assembler<dim>::copy_local_to_global_cell(
	const Assembly::CopyData::ForceConstrainedAssembly<dim> &copydata) {
	constraints->distribute_local_to_global(
		copydata.fi_vi_vector, copydata.local_dof_indices, *f
	);
}


}}}}

#include "L2_ForceConstrainedAssembly.inst.in"
