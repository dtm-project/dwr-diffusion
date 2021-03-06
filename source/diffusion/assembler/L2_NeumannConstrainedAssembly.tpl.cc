/**
 * @file L2_NeumannConstrainedAssembly.tpl.cc
 * @author Uwe Koecher (UK)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 * 
 * @date 2018-11-15, Neumann, UK
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
#include <diffusion/assembler/L2_NeumannConstrainedAssembly.tpl.hh>
#include <diffusion/types/boundary_id.hh>

// DEAL.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/filtered_iterator.h>

// C++ includes
#include <functional>
#include <iterator>

namespace diffusion {
namespace Assemble {
namespace L2 {
namespace NeumannConstrained {

namespace Assembly {
namespace Scratch {

/// (Struct-) Constructor.
template<int dim>
NeumannConstrainedAssembly<dim>::NeumannConstrainedAssembly(
	const dealii::FiniteElement<dim> &fe,
	const dealii::Mapping<dim> &mapping,
	const dealii::Quadrature<dim-1> &quad_face,
	const dealii::UpdateFlags &uflags_face) :
	fe_face_values(mapping, fe, quad_face, uflags_face),
	phi(fe.dofs_per_cell) {
}


/// (Struct-) Copy constructor.
template<int dim>
NeumannConstrainedAssembly<dim>::NeumannConstrainedAssembly(
	const NeumannConstrainedAssembly &scratch) :
	fe_face_values(
		scratch.fe_face_values.get_mapping(),
		scratch.fe_face_values.get_fe(),
		scratch.fe_face_values.get_quadrature(),
		scratch.fe_face_values.get_update_flags()),
	phi(scratch.phi),
	JxW(scratch.JxW),
	u_N(scratch.u_N),
	face_no(scratch.face_no),
	q(scratch.q),
	k(scratch.k),
	i(scratch.i) {
}

}

namespace CopyData {

/// (Struct-) Constructor.
template<int dim>
NeumannConstrainedAssembly<dim>::NeumannConstrainedAssembly(
	const dealii::FiniteElement<dim> &fe) :
	fi_vi_vector(fe.dofs_per_cell),
	local_dof_indices(fe.dofs_per_cell) {
}


/// (Struct-) Copy constructor.
template<int dim>
NeumannConstrainedAssembly<dim>::NeumannConstrainedAssembly(
	const NeumannConstrainedAssembly& copydata) :
	fi_vi_vector(copydata.fi_vi_vector),
	local_dof_indices(copydata.local_dof_indices) {
}

}}
////////////////////////////////////////////////////////////////////////////////


template<int dim>
Assembler<dim>::
Assembler(
	std::shared_ptr< dealii::Vector<double> > u_N,
	std::shared_ptr< dealii::DoFHandler<dim> > dof,
	std::shared_ptr< dealii::FiniteElement<dim> > fe,
	std::shared_ptr< dealii::Mapping<dim> > mapping,
	std::shared_ptr< dealii::ConstraintMatrix > constraints) :
	u_N(u_N),
	dof(dof),
	fe(fe),
	mapping(mapping),
	constraints(constraints) {
	uflags_face =
		dealii::update_quadrature_points |
		dealii::update_values |
		dealii::update_JxW_values;
}


template<int dim>
void
Assembler<dim>::set_function(
	std::shared_ptr< dealii::Function<dim> > fun) {
	function.u_N = fun;
}


template<int dim>
void Assembler<dim>::assemble(
	const double time,
	const unsigned int q) {
	// init. global vector
	Assert(u_N.use_count(), dealii::ExcNotInitialized());
	*u_N = 0;
	
	// check
	Assert(function.u_N.use_count(), dealii::ExcNotInitialized());
	function.u_N->set_time(time);
	
	// setup quadrature; return if q==0
	const dealii::QGauss<dim-1> quad_face{q};
	if (!quad_face.size())
		return;
	
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
		Assembly::Scratch::NeumannConstrainedAssembly<dim> (
			*fe,
			*mapping,
			quad_face,
			uflags_face
		),
		Assembly::CopyData::NeumannConstrainedAssembly<dim> (*fe)
	);
}


template<int dim>
void Assembler<dim>::local_assemble_cell(
	const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
	Assembly::Scratch::NeumannConstrainedAssembly<dim> &scratch,
	Assembly::CopyData::NeumannConstrainedAssembly<dim> &copydata) {
	// get global indices
	cell->get_dof_indices(copydata.local_dof_indices);
	
	// initialize local vector
	copydata.fi_vi_vector = 0;
	
	if (cell->at_boundary())
	for (scratch.face_no=0;
		scratch.face_no < dealii::GeometryInfo<dim>::faces_per_cell;
		++scratch.face_no)
	if ((cell->face(scratch.face_no)->at_boundary()) &&
		(cell->face(scratch.face_no)->boundary_id() ==
		static_cast<dealii::types::boundary_id> (
		diffusion::types::boundary_id::Neumann))) {
		////////////////////////////////////////////////////////////////////////////
		// initialise
		//
		
		// reinit scratch and data to current cell
		scratch.fe_face_values.reinit(cell,scratch.face_no);
		
		////////////////////////////////////////////////////////////////////////////
		// assemble terms
		//
		
		for (scratch.q=0;
			scratch.q < scratch.fe_face_values.n_quadrature_points; ++scratch.q) {
			////////////////////////////////////////////////////////////////////////
			// prefetch data on the current quadrature point
			
			scratch.JxW = scratch.fe_face_values.JxW(scratch.q);
		
			scratch.u_N = function.u_N->value(
				scratch.fe_face_values.quadrature_point(scratch.q),
				0 // component
			);
			
			// prefetch data
			for (scratch.k=0;
				scratch.k < scratch.fe_face_values.get_fe().dofs_per_cell;
				++scratch.k) {
				scratch.phi[scratch.k] =
					scratch.fe_face_values.shape_value_component(
						scratch.k,
						scratch.q,
						0 // component
					);
			}
			
			////////////////////////////////////////////////////////////////////////
			// assemble
			//
			
			// u_N (Neumann boundary function)
			for (scratch.i=0;
				scratch.i < scratch.fe_face_values.get_fe().dofs_per_cell;
				++scratch.i) {
				copydata.fi_vi_vector[scratch.i] += (
					scratch.phi[scratch.i] *
					scratch.u_N *
					scratch.JxW
				);
			}
		}
	}
}


template<int dim>
void Assembler<dim>::copy_local_to_global_cell(
	const Assembly::CopyData::NeumannConstrainedAssembly<dim> &copydata) {
	constraints->distribute_local_to_global(
		copydata.fi_vi_vector, copydata.local_dof_indices, *u_N
	);
}


}}}}

#include "L2_NeumannConstrainedAssembly.inst.in"
