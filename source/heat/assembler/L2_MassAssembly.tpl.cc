/**
 * @file L2_MassAssembly.tpl.cc
 * @author Uwe Koecher (UK)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 *
 * @date 2018-03-08, included from ewave, UK
 * @date 2017-12-08, cG/L2, UK
 * @date 2017-09-13, xwave/ewave, UK
 * @date 2015-05-18, UK
 * @date 2012-10-26, 2013-08-15, 2014-04-30, UK
 *
 * @brief Purpose: Assemble Mass Matrix (L2-integrals)
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
#include <heat/assembler/L2_MassAssembly.tpl.hh>

namespace heat {
namespace Assemble {
namespace L2 {
namespace Mass {

namespace Assembly {
namespace Scratch {

/// (Struct-) Constructor.
template<int dim>
MassAssembly<dim>::MassAssembly(
	const dealii::FiniteElement<dim> &fe,
	const dealii::Mapping<dim> &mapping,
	const dealii::Quadrature<dim> &quad,
	const dealii::UpdateFlags &uflags_cell) :
	fe_values(mapping, fe, quad, uflags_cell),
	phi(fe.dofs_per_cell),
	density(1.),
	JxW(0),
	dofs_per_cell(0),
	n_components(0) {
}


/// (Struct-) Copy constructor.
template<int dim>
MassAssembly<dim>::MassAssembly(const MassAssembly &scratch) :
	fe_values(
		scratch.fe_values.get_mapping(),
		scratch.fe_values.get_fe(),
		scratch.fe_values.get_quadrature(),
		scratch.fe_values.get_update_flags()),
	phi(scratch.phi),
	density(scratch.density),
	JxW(scratch.JxW),
	dofs_per_cell(scratch.dofs_per_cell),
	n_components(scratch.n_components),
	q(scratch.q),
	component(scratch.component),
	k(scratch.k),
	i(scratch.i),
	j(scratch.j) {
}

} // namespace Scratch
namespace CopyData {

/// (Struct-) Constructor.
template<int dim>
MassAssembly<dim>::MassAssembly(
	const dealii::FiniteElement<dim> &fe) :
	ui_vi_matrix(fe.dofs_per_cell, fe.dofs_per_cell),
	local_dof_indices(fe.dofs_per_cell) {
}


/// (Struct-) Copy constructor.
template<int dim>
MassAssembly<dim>::MassAssembly(const MassAssembly &copydata) :
	ui_vi_matrix(copydata.ui_vi_matrix),
	local_dof_indices(copydata.local_dof_indices) {
}

} // namespace CopyData
} // namespace Assembly
////////////////////////////////////////////////////////////////////////////////


/// Constructor.
template<int dim>
Assembler<dim>::
Assembler(
	std::shared_ptr< dealii::SparseMatrix<double> > M,
	std::shared_ptr< dealii::DoFHandler<dim> > dof,
	std::shared_ptr< dealii::FiniteElement<dim> > fe,
	std::shared_ptr< dealii::Mapping<dim> > mapping,
	std::shared_ptr< dealii::ConstraintMatrix > constraints) :
	M(M),
	dof(dof),
	fe(fe),
	mapping(mapping),
	constraints(constraints) {
	// init UpdateFlags
	uflags =
		dealii::update_quadrature_points |
		dealii::update_values |
		dealii::update_JxW_values;
}


template<int dim>
void
Assembler<dim>::set_density(
	std::shared_ptr< dealii::Function<dim> > fun) {
	function.density = fun;
}


template<int dim>
void Assembler<dim>::assemble(
	const unsigned int q) {
	// assemble matrix
	const dealii::QGauss<dim> quad{ (q ? q : (fe->tensor_degree()+1)) };
	
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
		Assembly::Scratch::MassAssembly<dim> (*fe, *mapping, quad, uflags),
		Assembly::CopyData::MassAssembly<dim> (*fe)
	);
}


/// Local assemble on cell.
template<int dim>
void Assembler<dim>::local_assemble_cell(
	const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
	Assembly::Scratch::MassAssembly<dim> &scratch,
	Assembly::CopyData::MassAssembly<dim> &copydata) {
	
	// reinit scratch and data to current cell
	scratch.fe_values.reinit(cell);
	scratch.dofs_per_cell = scratch.fe_values.get_fe().dofs_per_cell;
	scratch.n_components = scratch.fe_values.get_fe().n_components();
	cell->get_dof_indices(copydata.local_dof_indices);
	
	// initialize local matrix with zeros
	copydata.ui_vi_matrix = 0;
	
	// assemble cell terms
	for (scratch.q=0; scratch.q < scratch.fe_values.n_quadrature_points;
		++scratch.q) {
		scratch.density = function.density->value(
			scratch.fe_values.quadrature_point(scratch.q),0
		);
		scratch.JxW = scratch.fe_values.JxW(scratch.q);
		
		// loop over all components of this finite element
		for (scratch.component=0; scratch.component < scratch.n_components;
			++scratch.component) {
		
		// loop over all basis functions to get the shape values
		for (scratch.k=0; scratch.k < scratch.dofs_per_cell; ++scratch.k) {
			scratch.phi[scratch.k] =
				scratch.fe_values.shape_value_component(
					scratch.k,
					scratch.q,
					scratch.component
				);
		}
		
		// loop over all test & trial function combinitions to get the assembly
		for (scratch.i=0; scratch.i < scratch.dofs_per_cell; ++scratch.i)
		for (scratch.j=0; scratch.j < scratch.dofs_per_cell; ++scratch.j) {
			copydata.ui_vi_matrix(scratch.i,scratch.j) += (
				scratch.density *
				(scratch.phi[scratch.i] * scratch.phi[scratch.j]) *
				scratch.JxW
			);
		} // for ij
		
		} // for component
	} // for q
}


/// Copy local assembly to global matrix.
template<int dim>
void Assembler<dim>::copy_local_to_global_cell(
	const Assembly::CopyData::MassAssembly<dim> &copydata) {
	// copy mass matrix
	constraints->distribute_local_to_global(
		copydata.ui_vi_matrix,
		copydata.local_dof_indices, copydata.local_dof_indices,
		*M
	);
}

}}}} // namespaces

#include "L2_MassAssembly.inst.in"
