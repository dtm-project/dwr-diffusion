/**
 * @file DWR_ErrorEstimator.tpl.cc
 * @author Uwe Koecher (UK), Marius Paul Bruchhaeuser (MPB)
 * @date 2017-11-08, ErrorEstimator class, UK, MPB
 */

/*  Copyright (C) 2012-2017 by Uwe Koecher, Marius Paul Bruchhaeuser          */  
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
#include <DTM++/base/LogStream.hh>

#include <Poisson/ErrorEstimator/DWR_ErrorEstimator.tpl.hh>

// DEAL.II includes
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_refinement.h>

// C++ includes
#include <cmath>
#include <fstream>
#include <vector>

namespace Poisson {
namespace DWR {

namespace Assembly {
namespace Scratch {

/// (Struct-) Constructor.
template<int dim>
ErrorEstimateOnCell<dim>::ErrorEstimateOnCell(
	const dealii::FiniteElement<dim> &fe,
	const dealii::Mapping<dim> &mapping,
	const dealii::Quadrature<dim> &quad,
	const dealii::UpdateFlags &uflags) :
	fe_values(mapping, fe, quad, uflags),
	rhs_values(quad.size()),
	cell_laplacians(quad.size()),
	cell_values(quad.size()),
	dual_weights(quad.size()),
	R_u_h(quad.size()),
	cell_gradients(quad.size()),
	dual_weights_gradients(quad.size()) {
}


/// (Struct-) Copy constructor.
template<int dim>
ErrorEstimateOnCell<dim>::ErrorEstimateOnCell(const ErrorEstimateOnCell &scratch) :
	fe_values(
		scratch.fe_values.get_mapping(),
		scratch.fe_values.get_fe(),
		scratch.fe_values.get_quadrature(),
		scratch.fe_values.get_update_flags()),
	rhs_values(scratch.rhs_values),
	cell_laplacians(scratch.cell_laplacians), 
	cell_values(scratch.cell_values),
	dual_weights(scratch.dual_weights),
	R_u_h(scratch.R_u_h),
	cell_gradients(scratch.cell_gradients),
	dual_weights_gradients(scratch.dual_weights_gradients) {
}


/// (Struct-) Constructor.
template<int dim>
ErrorEstimateOnFace<dim>::ErrorEstimateOnFace(
	const dealii::FiniteElement<dim> &fe,
	const dealii::Mapping<dim> &mapping,
	const dealii::Quadrature<dim-1> &quad,
	const dealii::UpdateFlags &uflags) :
	fe_face_values(mapping, fe, quad, uflags),
	fe_face_values_neighbor(mapping, fe, quad, uflags),
	fe_subface_values(mapping, fe, quad, dealii::UpdateFlags::update_gradients),
	boundary_values(quad.size()),
	g_h(quad.size()),
	jump_residuals(quad.size()),
	dual_weights(quad.size()),
	inhom_dirichlet_difference(quad.size()),
	dual_solution_gradients(quad.size()),
	cell_grads(quad.size()),
	neighbor_grads(quad.size()) {
}


/// (Struct-) Copy constructor.
template<int dim>
ErrorEstimateOnFace<dim>::ErrorEstimateOnFace(const ErrorEstimateOnFace &scratch) :
	fe_face_values(
		scratch.fe_face_values.get_mapping(),
		scratch.fe_face_values.get_fe(),
		scratch.fe_face_values.get_quadrature(),
		scratch.fe_face_values.get_update_flags()),
	fe_face_values_neighbor(
		scratch.fe_face_values_neighbor.get_mapping(),
		scratch.fe_face_values_neighbor.get_fe(),
		scratch.fe_face_values_neighbor.get_quadrature(),
		scratch.fe_face_values_neighbor.get_update_flags()),
	fe_subface_values(
		scratch.fe_subface_values.get_mapping(),
		scratch.fe_subface_values.get_fe(),
		scratch.fe_subface_values.get_quadrature(),
		scratch.fe_subface_values.get_update_flags()),
	boundary_values(scratch.boundary_values),
	g_h(scratch.g_h),
	jump_residuals(scratch.jump_residuals),
	dual_weights(scratch.dual_weights),
	inhom_dirichlet_difference(scratch.inhom_dirichlet_difference),
	dual_solution_gradients(scratch.dual_solution_gradients),
	cell_grads(scratch.cell_grads),
	neighbor_grads(scratch.neighbor_grads) {
}


/// (Struct-) Constructor.
template<int dim>
ErrorEstimates<dim>::ErrorEstimates(
	const dealii::FiniteElement<dim> &fe,
	const dealii::Mapping<dim>       &mapping,
	const dealii::Quadrature<dim>    &quad_cell,
	const dealii::Quadrature<dim-1>  &quad_face,
	const dealii::UpdateFlags        &uflags_cell,
	const dealii::UpdateFlags        &uflags_face) :
	cell(fe, mapping, quad_cell, uflags_cell),
	face(fe, mapping, quad_face, uflags_face) {
}


/// (Struct-) Copy constructor.
template<int dim>
ErrorEstimates<dim>::ErrorEstimates(const ErrorEstimates &scratch) :
	cell(scratch.cell),
	face(scratch.face) {
}

}

namespace CopyData {

/// (Struct-) Constructor.
template<int dim>
ErrorEstimateOnCell<dim>::ErrorEstimateOnCell() {
}


/// (Struct-) Copy constructor.
template<int dim>
ErrorEstimateOnCell<dim>::ErrorEstimateOnCell(const ErrorEstimateOnCell &copydata) :
	cell(copydata.cell),
	value(copydata.value) {
}


/// (Struct-) Constructor.
template<int dim>
ErrorEstimateOnFace<dim>::ErrorEstimateOnFace() {
}


/// (Struct-) Copy constructor.
template<int dim>
ErrorEstimateOnFace<dim>::ErrorEstimateOnFace(const ErrorEstimateOnFace &copydata) :
	face(copydata.face),
	value(copydata.value) {
}

/// (Struct-) Constructor.
template<int dim>
ErrorEstimates<dim>::ErrorEstimates() {
}


/// (Struct-) Copy constructor.
template<int dim>
ErrorEstimates<dim>::ErrorEstimates(const ErrorEstimates &copydata) :
	cell(copydata.cell),
	face(copydata.face) {
}


} // namespace CopyData
} // namespace Assembly
////////////////////////////////////////////////////////////////////////////////


template<int dim>
void
ErrorEstimator<dim>::
set_objects(
	std::shared_ptr< Poisson::Grid_DWR<dim,1> > _grid,
	std::shared_ptr< dealii::Function<dim> > _epsilon,
	std::shared_ptr< dealii::Function<dim> > _BoundaryValues,
	std::shared_ptr< dealii::Function<dim> > f) {
	grid = _grid;
	epsilon = _epsilon;
	BoundaryValues = _BoundaryValues;
	function.f = f;
}


template<int dim>
void
ErrorEstimator<dim>::
estimate(
	std::shared_ptr< dealii::Vector<double> > u, ///< primal problem solution on dual space
	std::shared_ptr< dealii::Vector<double> > z, ///< dual problem solution
	std::shared_ptr< dealii::Vector<double> > error_indicators
) {
	Assert(u.use_count(), dealii::ExcNotInitialized());
	Assert(z.use_count(), dealii::ExcNotInitialized());
	Assert(error_indicators.use_count(), dealii::ExcNotInitialized());
	Assert(function.f.use_count(), dealii::ExcNotInitialized());
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	
	dual.u = u;
	dual.z = z;

	dual_weights.reinit(grid->dual.dof->n_dofs());
	
	dealii::FETools::interpolation_difference(
		*grid->dual.dof,
		*grid->dual.constraints,
		*dual.z,
		*grid->primal.dof,
		*grid->primal.constraints,
		dual_weights
	);
	
// 	std::cout << "dual.z = " << *dual.z << std::endl;
	
////////////////////////////////////////////////////////////////////////////////
/////// Interpolation of dual.z into primal FE-room: //TEST
	primal_z.reinit(grid->primal.dof->n_dofs());
	dealii::FETools::interpolate(
		*(grid->dual.dof),
		*(dual.z),
		*(grid->primal.dof),
		*(grid->primal.constraints),
		primal_z
	);
// 	std::cout << "primal_z = " << primal_z << std::endl;
	
	primal_z_dual.reinit(grid->dual.dof->n_dofs());
	dealii::FETools::interpolate(
		*(grid->primal.dof),
		primal_z,
		*(grid->dual.dof),
		*(grid->dual.constraints),
		primal_z_dual
	);
// 	std::cout << "primal_z_dual = " << primal_z_dual << std::endl;
////////////////////////////////////////////////////////////////////////////////
/////// Interpolation of g (u = g on dOmega):
	dealii::Vector<double> vec_g(grid->primal.dof->n_dofs());
	dealii::VectorTools::interpolate(*(grid->primal.mapping),
									 *(grid->primal.dof),
									 *BoundaryValues,
									 vec_g);
	grid->primal.constraints->distribute(vec_g);
	
	g_interpolated.reinit(grid->dual.dof->n_dofs());
	dealii::FETools::interpolate(
		*(grid->primal.dof),
		vec_g,
		*(grid->dual.dof),
		*grid->dual.constraints,
		g_interpolated
	);
	
// 	g_interpolated.reinit(grid->dual.dof->n_dofs());
// 	dealii::VectorTools::interpolate(*(grid->dual.mapping),
// 									 *(grid->dual.dof),
// 									 *BoundaryValues,
// 									 g_interpolated);
// 	grid->primal.constraints->distribute(g_interpolated);
////////////////////////////////////////////////////////////////////////////////	
	
	{
		auto cell = grid->dual.dof->begin_active();
		auto endc = grid->dual.dof->end();
		for ( ; cell != endc; ++cell) {
			cell_integrals[cell] = std::numeric_limits< double >::quiet_NaN();
			for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<dim>::faces_per_cell; ++face_no) {
				face_integrals[cell->face(face_no)] = std::numeric_limits< double >::quiet_NaN();
			}
		}
	}
	
	std::cout << "number of cells: " << cell_integrals.size() << std::endl;
	std::cout << "number of faces: " << face_integrals.size() << std::endl;
	
	dealii::QGauss<dim> quad_cell(grid->dual.fe->tensor_degree()+2);
	dealii::QGauss<dim-1> quad_face(grid->dual.fe->tensor_degree()+2);
	
	dealii::WorkStream::run(
		grid->dual.dof->begin_active(),
		grid->dual.dof->end(),
		std::bind (
			&ErrorEstimator<dim>::assemble_local_error,
			this,
			std::placeholders::_1,
			std::placeholders::_2,
			std::placeholders::_3
		),
		std::bind (
			&ErrorEstimator<dim>::copy_local_error,
			this,
			std::placeholders::_1
		),
		Assembly::Scratch::ErrorEstimates<dim> (
			*grid->dual.fe,
			*grid->dual.mapping,
			quad_cell,
			quad_face,
			//
			dealii::update_values |
			dealii::update_gradients |
			dealii::update_hessians |
			dealii::update_quadrature_points |
			dealii::update_JxW_values,
			//
			dealii::update_values |
			dealii::update_gradients |
			dealii::update_quadrature_points |
			dealii::update_normal_vectors |
			dealii::update_JxW_values),
		Assembly::CopyData::ErrorEstimates<dim> ()
	);
	
	(*error_indicators) = 0;
	{
		auto cell = grid->dual.dof->begin_active();
		auto endc = grid->dual.dof->end();
		
		for (unsigned int cell_no=0; cell != endc; ++cell, ++cell_no) {
			(*error_indicators)[cell_no] = cell_integrals[cell];
			
			for (unsigned int face_no=0;
				face_no < dealii::GeometryInfo<dim>::faces_per_cell; ++face_no) {
				Assert(
					(face_integrals.find(cell->face(face_no)) != face_integrals.end()),
					dealii::ExcMessage("Your face iterator does not exist in your map(face_it,double). \
					Please check if you have assembled this error integral.")
				);
				
				(*error_indicators)[cell_no] -= 0.5 * face_integrals[cell->face(face_no)];
			}
		}
	}
	
	// check if error_indicators vector has valid entries only
	{
		for (unsigned int cell_no=0; cell_no < error_indicators->size(); ++cell_no) {
			Assert(
				!std::isnan((*error_indicators)[cell_no]),
				dealii::ExcMessage("Your error indicator has quiet_NaN entries. \
				Please check if you have assembled cell_integrals and face_integrals correctly.")
			);
			(void)cell_no;
		}
	}
	cell_integrals.clear();
	face_integrals.clear();
}


template<int dim>
void
ErrorEstimator<dim>::
assemble_local_error(
	const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
	Assembly::Scratch::ErrorEstimates<dim> &scratch,
	Assembly::CopyData::ErrorEstimates<dim> &copydata) {
	
	assemble_error_on_cell(cell, scratch.cell, copydata.cell);
	
	for (unsigned int face_no=0; face_no < dealii::GeometryInfo<dim>::faces_per_cell; ++face_no) {
		// handle boundary faces
		if (cell->face(face_no)->at_boundary()) {

			assemble_error_on_boundary_face(
				cell,
				face_no,
				scratch.face,
				copydata.face
			);
			continue;
		}
		////////////////////////////////////////////////////////////////////////
		// interior faces only:
		
		// skip face with same refinement level where the neighbor cell index
		// is smaller than this ones
		if ((cell->index() > cell->neighbor(face_no)->index()) &&
			(cell->neighbor(face_no)->has_children() == false) &&
			(cell->level() == cell->neighbor(face_no)->level())) {
			// integrate face value from the neighbor cell
			continue;
		}
		
		// integrate from coarser cell
		if (cell->level() > cell->neighbor(face_no)->level()) {
			continue;
		}
		
		if (cell->face(face_no)->has_children() == false) {
			assemble_error_on_regular_face(
				cell,
				face_no,
				scratch.face,
				copydata.face
			);
		}
		else {
			assemble_error_on_irregular_face(
				cell,
				face_no,
				scratch.face,
				copydata.face
			);
		}
	}
}


template<int dim>
void
ErrorEstimator<dim>::
assemble_error_on_cell(
	const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
	Assembly::Scratch::ErrorEstimateOnCell<dim> &scratch,
	Assembly::CopyData::ErrorEstimateOnCell<dim> &copydata) {

	// reinit scratch and data to current cell
	scratch.fe_values.reinit(cell);

	function.f->value_list(								// calculates f
		scratch.fe_values.get_quadrature_points(),
		scratch.rhs_values
	);
	
	scratch.fe_values.get_function_laplacians(			// calculates Laplace(u_h)
		*dual.u,
		scratch.cell_laplacians
	);
	
	scratch.fe_values.get_function_values(				// calculates "z-phi_h" respectively z_h-Interp(z_h)
		dual_weights,
		scratch.dual_weights
	);

	for (unsigned int q=0; q < scratch.fe_values.n_quadrature_points; ++q) {
		scratch.R_u_h[q] = (
					scratch.rhs_values[q] + 
					(epsilon->value(scratch.fe_values.quadrature_point(q), 0)*scratch.cell_laplacians[q]) 
					);

	}
	
	// initialize local matrix with zeros
	copydata.cell = cell;
	copydata.value = 0;
	
	for (unsigned int q=0; q < scratch.fe_values.n_quadrature_points; ++q) {
		copydata.value += 
			((
			scratch.R_u_h[q]*
			scratch.dual_weights[q]
			)*
			scratch.fe_values.JxW(q));
	}
	
	cell_integrals[copydata.cell] = copydata.value;

}


template<int dim>
void
ErrorEstimator<dim>::
assemble_error_on_boundary_face(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		const unsigned int face_no,
		Assembly::Scratch::ErrorEstimateOnFace<dim> &scratch,
		Assembly::CopyData::ErrorEstimateOnFace<dim> &copydata) {
	Assert(
		(cell->face(face_no).state() == dealii::IteratorState::valid),
		dealii::ExcInternalError()
	);

	scratch.fe_face_values.reinit(cell, face_no);
	
	BoundaryValues->value_list(scratch.fe_face_values.get_quadrature_points(),
							   scratch.boundary_values);
	
	scratch.fe_face_values.get_function_gradients(
	*dual.z,
	scratch.dual_solution_gradients
	);
	
	scratch.fe_face_values.get_function_values(
		g_interpolated,
		scratch.g_h
	);
	
	for (unsigned int q=0;q<scratch.fe_face_values.n_quadrature_points; ++q) {
		scratch.inhom_dirichlet_difference[q] = (scratch.boundary_values[q] - scratch.g_h[q]);
	}
	
	copydata.face  = cell->face(face_no);
	copydata.value = 0;
	
	for (unsigned int q=0; q < scratch.fe_face_values.n_quadrature_points; ++q) { // (g-g_h, epsilon*grad(z_h)*n)_dOmega
		copydata.value +=
			2.*(													//multiplied with 2, because within the estimate() function
			(scratch.inhom_dirichlet_difference[q] *						//the whole faces contribution will be subtracted  by the factor
			(scratch.fe_face_values.normal_vector(q)*
			(epsilon->value(scratch.fe_face_values.quadrature_point(q), 0)*
			scratch.dual_solution_gradients[q])))*
			scratch.fe_face_values.JxW(q));
	}

	face_integrals[copydata.face] = copydata.value;
}


template<int dim>
void
ErrorEstimator<dim>::
assemble_error_on_regular_face(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		const unsigned int face_no,
		Assembly::Scratch::ErrorEstimateOnFace<dim> &scratch,
		Assembly::CopyData::ErrorEstimateOnFace<dim> &copydata) {
	Assert(
		(cell->neighbor(face_no).state() == dealii::IteratorState::valid),
		dealii::ExcInternalError()
	);
	
	scratch.fe_face_values.reinit(cell, face_no);
	
	scratch.fe_face_values_neighbor.reinit(
		cell->neighbor(face_no),
		cell->neighbor_of_neighbor(face_no)
	);
	
	scratch.fe_face_values.get_function_gradients(
		*dual.u,
		scratch.cell_grads
	);
	
	scratch.fe_face_values_neighbor.get_function_gradients(
		*dual.u,
		scratch.neighbor_grads
	);
	
	for (unsigned int q=0; q < scratch.fe_face_values.n_quadrature_points; ++q) {
		scratch.jump_residuals[q] = (
			(epsilon->value(scratch.fe_face_values.quadrature_point(q), 0)*
			(scratch.cell_grads[q] - scratch.neighbor_grads[q])) *
			scratch.fe_face_values.normal_vector(q)
		);
	}
	
	scratch.fe_face_values.get_function_values(
		dual_weights,
		scratch.dual_weights
	);
	
	copydata.face  = cell->face(face_no);
	copydata.value = 0;
	
	for (unsigned int q=0; q < scratch.fe_face_values.n_quadrature_points; ++q) {
		copydata.value += (
			scratch.jump_residuals[q] * scratch.dual_weights[q] *
			scratch.fe_face_values.JxW(q)
		);
	}
	
	face_integrals[copydata.face] = copydata.value;
}


template<int dim>
void
ErrorEstimator<dim>::
assemble_error_on_irregular_face(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		const unsigned int face_no,
		Assembly::Scratch::ErrorEstimateOnFace<dim> &scratch,
		Assembly::CopyData::ErrorEstimateOnFace<dim> &copydata) {
	Assert(
		(cell->neighbor(face_no).state() == dealii::IteratorState::valid),
		dealii::ExcInternalError()
	);
	
	Assert(
		(cell->neighbor(face_no)->has_children()),
		dealii::ExcInternalError()
	);
	
	for (unsigned int subface_no=0; subface_no < cell->face(face_no)->n_children();
		++subface_no) {
		Assert(
			(cell->neighbor_child_on_subface(face_no,subface_no)->face(
				cell->neighbor_of_neighbor(face_no) ) ==
			cell->face(face_no)->child(subface_no)),
			dealii::ExcInternalError()
		);
		
		scratch.fe_subface_values.reinit(cell, face_no, subface_no);
		
		scratch.fe_face_values_neighbor.reinit(
			cell->neighbor_child_on_subface(face_no, subface_no),
			cell->neighbor_of_neighbor(face_no)
		);
		
		scratch.fe_subface_values.get_function_gradients(
			*dual.u,
			scratch.cell_grads
		);
		
		scratch.fe_face_values_neighbor.get_function_gradients(
			*dual.u,
			scratch.neighbor_grads
		);
		
		for (unsigned int q=0; q < scratch.fe_face_values.n_quadrature_points; ++q) {
			scratch.jump_residuals[q] = (
				(epsilon->value(scratch.fe_face_values.quadrature_point(q), 0)*
				(scratch.neighbor_grads[q] - scratch.cell_grads[q])) *
				scratch.fe_face_values_neighbor.normal_vector(q)
			);
		}
		
		scratch.fe_face_values_neighbor.get_function_values(
			dual_weights,
			scratch.dual_weights
		);
		
		copydata.face  = cell->face(face_no)->child(subface_no);
		copydata.value = 0;
		
		for (unsigned int q=0; q < scratch.fe_face_values.n_quadrature_points; ++q) {
			copydata.value += (
				scratch.jump_residuals[q] * scratch.dual_weights[q] *
				scratch.fe_face_values_neighbor.JxW(q)
			);
		}
		
		face_integrals[copydata.face] = copydata.value;
	}
	
	// compute integral value of big face as sum of the subface integral values
	copydata.face  = cell->face(face_no);
	copydata.value = 0;
	
	for (unsigned int subface_no=0; subface_no < cell->face(face_no)->n_children();
		++subface_no) {
		copydata.value += face_integrals[cell->face(face_no)->child(subface_no)];
	}
	
	face_integrals[cell->face(face_no)] = copydata.value;
}


template<int dim>
void
ErrorEstimator<dim>::copy_local_error(
	const Assembly::CopyData::ErrorEstimates<dim> &) {
}


}} // namespace

#include "DWR_ErrorEstimator.inst.in"
