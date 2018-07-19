/**
 * @file L2_Je_global_L2L2_Assembly.tpl.cc
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhäuser (MPB)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 * 
 * @date 2018-03-09, derived from L2_Je_global_L2L2_Assembly and MPB code, UK
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

// PROJECT includes
#include <heat/assembler/L2_Je_global_L2L2_Assembly.tpl.hh>

namespace heat {
namespace Assemble {
namespace L2 {
namespace Je_global_L2L2 {

namespace Assembly {
namespace Scratch {

/// (Struct-) Constructor.
template<int dim>
Je_global_L2L2Assembly<dim>::Je_global_L2L2Assembly(
	const dealii::FiniteElement<dim> &fe,
	const dealii::Mapping<dim> &mapping,
	const dealii::Quadrature<dim> &quad,
	const dealii::UpdateFlags &uflags) :
	fe_values(mapping, fe, quad, uflags),
	phi(fe.dofs_per_cell),
	JxW(0),
	u_E(0),
	u_h(0) {
}


/// (Struct-) Copy constructor.
template<int dim>
Je_global_L2L2Assembly<dim>::Je_global_L2L2Assembly(
	const Je_global_L2L2Assembly &scratch) :
	fe_values(
		scratch.fe_values.get_mapping(),
		scratch.fe_values.get_fe(),
		scratch.fe_values.get_quadrature(),
		scratch.fe_values.get_update_flags()),
	phi(scratch.phi),
	JxW(scratch.JxW),
	u_E(scratch.u_E),
	u_h(scratch.u_h) {
}

}

namespace CopyData {

/// (Struct-) Constructor.
template<int dim>
Je_global_L2L2Assembly<dim>::Je_global_L2L2Assembly(
	const dealii::FiniteElement<dim> &fe) :
	vi_Jei_vector(fe.dofs_per_cell),
	local_dof_indices(fe.dofs_per_cell) {
}


/// (Struct-) Copy constructor.
template<int dim>
Je_global_L2L2Assembly<dim>::Je_global_L2L2Assembly(
	const Je_global_L2L2Assembly& copydata) :
	vi_Jei_vector(copydata.vi_Jei_vector),
	local_dof_indices(copydata.local_dof_indices) {
}

}}
////////////////////////////////////////////////////////////////////////////////


/// Constructor.
template<int dim>
Assembler<dim>::
Assembler(
	std::shared_ptr< dealii::DoFHandler<dim> > dof,
	std::shared_ptr< dealii::FiniteElement<dim> > fe,
	std::shared_ptr< dealii::Mapping<dim> > mapping,
	std::shared_ptr< dealii::ConstraintMatrix > constraints) :
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
void Assembler<dim>::assemble(
	std::shared_ptr< dealii::Vector<double> > _Je,
	const double time,
	std::shared_ptr< dealii::Function<dim> > _u_E,
	std::shared_ptr< dealii::Vector<double> > _u_h,
	const unsigned int q,
	const bool quadrature_points_auto_mode) {
	// init
	Je = _Je;
	Assert( Je.use_count(), dealii::ExcNotInitialized() );
	
	function.u_E = _u_E;
	Assert( function.u_E.use_count(), dealii::ExcNotInitialized() );
	function.u_E->set_time(time);
	
	u_h = _u_h;
	Assert( u_h.use_count(), dealii::ExcNotInitialized() );
	
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
		Assembly::Scratch::Je_global_L2L2Assembly<dim> (*fe, *mapping, quad, uflags),
		Assembly::CopyData::Je_global_L2L2Assembly<dim> (*fe)
	);
}


/// Local assemble on cell.
template<int dim>
void Assembler<dim>::local_assemble_cell(
	const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
	Assembly::Scratch::Je_global_L2L2Assembly<dim> &scratch,
	Assembly::CopyData::Je_global_L2L2Assembly<dim> &copydata) {
	
	// reinit scratch and data to current cell
	scratch.fe_values.reinit(cell);
	cell->get_dof_indices(copydata.local_dof_indices);
	
	// initialize local matrix with zeros
	copydata.vi_Jei_vector = 0;
	
	// assemble cell terms
	for (unsigned int q{0}; q < scratch.fe_values.n_quadrature_points; ++q) {
		scratch.JxW = scratch.fe_values.JxW(q);
		
		// loop over all components of this finite element
		for (unsigned int component{0};
			component < scratch.fe_values.get_fe().n_components();
			++component) {
			// loop over all basis functions to get the shape values
			for (unsigned int k{0}; k < scratch.fe_values.get_fe().dofs_per_cell; ++k) {
				scratch.phi[k] =
					scratch.fe_values.shape_value_component(k,q,component);
			}
			
			scratch.u_E = function.u_E->value(
				scratch.fe_values.quadrature_point(q),
				component
			);
			
			// loop over all basis functions to get u_h
			scratch.u_h = 0;
			for (unsigned int j{0}; j < scratch.fe_values.get_fe().dofs_per_cell; ++j) {
				scratch.u_h +=
					(*u_h)[ copydata.local_dof_indices[j] ] *
					scratch.phi[j];
			}
			
			
			// loop over all basis function combinitions to get the assembly
			for (unsigned int i{0}; i < scratch.fe_values.get_fe().dofs_per_cell; ++i) {
				copydata.vi_Jei_vector[i] += (
					scratch.phi[i] *
					(scratch.u_E - scratch.u_h) *
					scratch.JxW
				);
			} // for i
		} // for component
	} // for q
}


/// Copy local assembly to global matrix.
template<int dim>
void Assembler<dim>::copy_local_to_global_cell(
	const Assembly::CopyData::Je_global_L2L2Assembly<dim> &copydata) {
	constraints->distribute_local_to_global(
		copydata.vi_Jei_vector, copydata.local_dof_indices, *Je
	);
}


}}}}

#include "L2_Je_global_L2L2_Assembly.inst.in"
