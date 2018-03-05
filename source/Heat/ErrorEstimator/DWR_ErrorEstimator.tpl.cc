/**
 * @file DWR_ErrorEstimator.tpl.cc
 * @author Marius Paul Bruchhaeuser (MPB), Uwe Koecher (UK)
 * @date 2017-08-01, ErrorEstimator class, UK
 * @date 2017-08-01, Heat / DWR, UK
 */

/*  Copyright (C) 2012-2017 by Marius Paul Bruchhaeuser, Uwe Koecher          */
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

#include <Heat/ErrorEstimator/DWR_ErrorEstimator.tpl.hh>

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
#include <algorithm>
#include <list>
#include <iterator>

namespace Heat {
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
	rhs_values_old(quad.size()),
	cell_laplacians(quad.size()),
	cell_laplacians_old(quad.size()),
	cell_values(quad.size()),
	cell_values_old(quad.size()),
	dual_weights(quad.size()),
	dual_weights_old(quad.size()),
	R_u_h(quad.size()),
	R_u_h_old(quad.size()),
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
	rhs_values_old(scratch.rhs_values_old),
	cell_laplacians(scratch.cell_laplacians), 
	cell_laplacians_old(scratch.cell_laplacians_old),
	cell_values(scratch.cell_values),
	cell_values_old(scratch.cell_values_old),
	dual_weights(scratch.dual_weights),
	dual_weights_old(scratch.dual_weights_old),
	R_u_h(scratch.R_u_h),
	R_u_h_old(scratch.R_u_h_old),
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
	boundary_values_old(quad.size()),
	g_h(quad.size()),
	g_h_old(quad.size()),
	jump_residuals(quad.size()),
	jump_residuals_old(quad.size()),
	dual_weights(quad.size()),
	dual_weights_old(quad.size()),
	inhom_dirichlet_difference(quad.size()),
	inhom_dirichlet_difference_old(quad.size()),
	dual_solution_gradients(quad.size()),
	dual_solution_gradients_old(quad.size()),
	cell_grads(quad.size()),
	cell_grads_old(quad.size()),
	neighbor_grads(quad.size()),
	neighbor_grads_old(quad.size()) {
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
	boundary_values_old(scratch.boundary_values_old),
	g_h(scratch.g_h),
	g_h_old(scratch.g_h_old),
	jump_residuals(scratch.jump_residuals),
	jump_residuals_old(scratch.jump_residuals_old),
	dual_weights(scratch.dual_weights),
	dual_weights_old(scratch.dual_weights_old),
	inhom_dirichlet_difference(scratch.inhom_dirichlet_difference),
	inhom_dirichlet_difference_old(scratch.inhom_dirichlet_difference_old),
	dual_solution_gradients(scratch.dual_solution_gradients),
	dual_solution_gradients_old(scratch.dual_solution_gradients_old),
	cell_grads(scratch.cell_grads),
	cell_grads_old(scratch.cell_grads_old),
	neighbor_grads(scratch.neighbor_grads),
	neighbor_grads_old(scratch.neighbor_grads_old) {
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
	std::shared_ptr< Heat::Grid_DWR<dim,1> > _grid,
	std::shared_ptr< dealii::Function<dim> > _epsilon,
	std::shared_ptr< dealii::Function<dim> > _BoundaryValues,
	std::shared_ptr< dealii::Function<dim> > _BoundaryValues_old,
	std::shared_ptr< dealii::Function<dim> > _f,
	std::shared_ptr< dealii::Function<dim> > _f_old
   	) {
	grid = _grid;
	epsilon = _epsilon;
	BoundaryValues = _BoundaryValues;
	BoundaryValues_old = _BoundaryValues_old;
	function.f = _f;
	function.f_old = _f_old;
}


template<int dim>
void
ErrorEstimator<dim>::
estimate(
	std::shared_ptr< storage_data_vectors > _In_u, ///< list of primal problem solution on dual space
	std::shared_ptr< storage_data_vectors > _In_z, ///< list of dual problem solution
	std::shared_ptr< storage_data_vectors > _In_eta ///< list of error_indicators (eta^n)
) {
	In_u = _In_u;
	In_z = _In_z;
	In_eta = _In_eta;
	
	auto Inth(grid->slabs.begin());
// 	auto endIn(grid->slabs.end());
	auto Inth_old(grid->slabs.begin());
// 	auto endIn_old(grid->slabs.end());
	auto In_etath(In_eta->begin());
// 	auto endIn_etath(In_eta->end());
	auto In_uth(In_u->begin());
// 	auto endIn_uth(In_u->end());
	auto In_uth_old(In_u->begin());
// 	auto endIn_uth_old(In_u->end());
	auto In_zth(In_z->begin());
// 	auto endIn_zth(In_z->end());
	auto In_zth_old(In_z->begin());
// 	auto endIn_zth_old(In_z->end());
	
	// Loop over all time-intervals I_n (n=1,..,N) = #grids
	for (unsigned int n{1}; n <= grid->slabs.size(); ++n) {

		if (n == 1) {
			std::cout << "I_n: t_m = " << Inth->t_m << std::endl;
			std::cout << "I_n: t_n = " << Inth->t_n << std::endl;
			std::cout << "I_n: tau_n = " << Inth->tau_n() << std::endl;
			
			++In_uth;
			++In_zth;
			In_etath->x->reinit(Inth->tria->n_active_cells());
			auto error_indicators = std::make_shared< dealii::Vector<double> > ();
			error_indicators->reinit(Inth->tria->n_active_cells());
			dual.u = In_uth->x;
			dual.z = In_zth->x;
			
			dual.u_old = std::make_shared< dealii::Vector<double> > ();
			dual.u_old->reinit(Inth->dual.dof->n_dofs());
			dual.z_old = std::make_shared< dealii::Vector<double> > ();
			dual.z_old->reinit(Inth->dual.dof->n_dofs());
			dealii::VectorTools::interpolate_to_different_mesh(
				*(Inth_old->dual.dof),
				*(In_uth_old->x),
				*(Inth->dual.dof),
				*(Inth->dual.constraints),
				*(dual.u_old)
			);
			dealii::VectorTools::interpolate_to_different_mesh(
				*(Inth_old->dual.dof),
				*(In_zth_old->x),
				*(Inth->dual.dof),
				*(Inth->dual.constraints),
				*(dual.z_old)
			);
			
			Assert(In_uth->x.use_count(), dealii::ExcNotInitialized());
			Assert(In_zth->x.use_count(), dealii::ExcNotInitialized());
			Assert(error_indicators.use_count(), dealii::ExcNotInitialized());
			Assert(function.f.use_count(), dealii::ExcNotInitialized());
			Assert(grid.use_count(), dealii::ExcNotInitialized());
			
			dual_weights.reinit(Inth->dual.dof->n_dofs());
			
			dealii::FETools::interpolation_difference(
				*(Inth->dual.dof),
				*(Inth->dual.constraints),
				*(dual.z),
				*(Inth->primal.dof),
				*(Inth->primal.constraints),
				dual_weights
			);
			
			dual_weights_old.reinit(Inth->dual.dof->n_dofs());
			
			dealii::FETools::interpolation_difference(
				*(Inth->dual.dof),
				*(Inth->dual.constraints),
				*(dual.z_old),
				*(Inth->primal.dof),
				*(Inth->primal.constraints),
				dual_weights_old
			);
			
			// Interpolate dual.z into primal FE-room and afterwards back
			// into dual FE-room:
			// Compute I_h z_H^m,i = primal_z_dual
			
			primal_z_dual.reinit(Inth->dual.dof->n_dofs());
			dealii::FETools::back_interpolate(
				*(Inth->dual.dof),
				*(Inth->dual.constraints),
				*dual.z,
				*(Inth->primal.dof),
				*(Inth->primal.constraints),
				primal_z_dual
			);
			
			////////////////////////////////////////////////////////////////////
			// Interpolation of g (u = g on dOmega) at timepoint t_n:
			BoundaryValues->set_time(Inth->t_n);
			dealii::Vector<double> vec_g(Inth->primal.dof->n_dofs());
			dealii::VectorTools::interpolate(*(Inth->primal.mapping),
											*(Inth->primal.dof),
											*BoundaryValues,
											vec_g);
			Inth->primal.constraints->distribute(vec_g);
			
			g_interpolated.reinit(Inth->dual.dof->n_dofs());
			dealii::FETools::interpolate(
				*(Inth->primal.dof),
				vec_g,
				*(Inth->dual.dof),
				*(Inth->dual.constraints),
				g_interpolated
			);
			
			////////////////////////////////////////////////////////////////////
			// Interpolation of g (u = g on dOmega) at timepoint \f$ t_{n-1} \f$:
			BoundaryValues_old->set_time(Inth->t_m);
			dealii::Vector<double> vec_g_old(Inth->primal.dof->n_dofs());
			dealii::VectorTools::interpolate(*(Inth->primal.mapping),
											*(Inth->primal.dof),
											*BoundaryValues_old,
											vec_g_old);
			Inth->primal.constraints->distribute(vec_g_old);
			
			g_interpolated_old.reinit(Inth->dual.dof->n_dofs());
			dealii::FETools::interpolate(
				*(Inth->primal.dof),
				vec_g_old,
				*(Inth->dual.dof),
				*(Inth->dual.constraints),
				g_interpolated_old
			);
			
			// 	g_interpolated.reinit(Inth->dual.dof->n_dofs());
			// 	dealii::VectorTools::interpolate(*(Inth->dual.mapping),
			// 									 *(Inth->dual.dof),
			// 									 *BoundaryValues,
			// 									 g_interpolated);
			// 	Inth->primal.constraints->distribute(g_interpolated);
			
			////////////////////////////////////////////////////////////////////
			function.f->set_time(Inth->t_n);
			function.f_old->set_time(Inth->t_m);
			
			tau_n = Inth->tau_n();
			time = Inth->t_n;
			time_old = Inth->t_m;
			
			{
				auto cell = Inth->dual.dof->begin_active();
				auto endc = Inth->dual.dof->end();
				
				for ( ; cell != endc; ++cell) {
					cell_integrals[cell] = std::numeric_limits< double >::quiet_NaN();
					
					for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<dim>::faces_per_cell; ++face_no) {
						face_integrals[cell->face(face_no)] = std::numeric_limits< double >::quiet_NaN();
					}
				}
			}
			
			std::cout << "number of cells: " << cell_integrals.size() << std::endl;
			std::cout << "number of faces: " << face_integrals.size() << std::endl;
			
			dealii::QGaussLobatto<dim> quad_cell(Inth->dual.fe->tensor_degree()+2);
			dealii::QGaussLobatto<dim-1> quad_face(Inth->dual.fe->tensor_degree()+2);
			
			dealii::WorkStream::run(
				Inth->dual.dof->begin_active(),
				Inth->dual.dof->end(),
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
					*(Inth->dual.fe),
					*(Inth->dual.mapping),
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
				auto cell = Inth->dual.dof->begin_active();
				auto endc = Inth->dual.dof->end();
				
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
			//TEST
			*(In_etath->x) = *(error_indicators);
			cell_integrals.clear();
			face_integrals.clear();
			++In_uth_old;
			++In_zth_old;
		} // if (n==1)
		else {
			++Inth;
			++In_etath;
			++In_uth;
			++In_zth;
			
			std::cout << "I_n: t_m = " << Inth->t_m << std::endl;
			std::cout << "I_n: t_n = " << Inth->t_n << std::endl;
			std::cout << "I_n: tau_n = " << Inth->tau_n() << std::endl;
			
			In_etath->x->reinit(Inth->tria->n_active_cells());
			auto error_indicators = std::make_shared< dealii::Vector<double> > ();
			error_indicators->reinit(Inth->tria->n_active_cells());
			dual.u = In_uth->x;
			dual.z = In_zth->x;
			
			dual.u_old = std::make_shared< dealii::Vector<double> > ();
			dual.u_old->reinit(Inth->dual.dof->n_dofs());
			dual.z_old = std::make_shared< dealii::Vector<double> > ();
			dual.z_old->reinit(Inth->dual.dof->n_dofs());
			dealii::VectorTools::interpolate_to_different_mesh(
				*(Inth_old->dual.dof),
				*(In_uth_old->x),
				*(Inth->dual.dof),
				*(Inth->dual.constraints),
				*(dual.u_old)
			);
			dealii::VectorTools::interpolate_to_different_mesh(
				*(Inth_old->dual.dof),
				*(In_zth_old->x),
				*(Inth->dual.dof),
				*(Inth->dual.constraints),
				*(dual.z_old)
			);
			
			Assert(In_uth->x.use_count(), dealii::ExcNotInitialized());
			Assert(In_zth->x.use_count(), dealii::ExcNotInitialized());
			Assert(error_indicators.use_count(), dealii::ExcNotInitialized());
			Assert(function.f.use_count(), dealii::ExcNotInitialized());
			Assert(grid.use_count(), dealii::ExcNotInitialized());
			
			dual_weights.reinit(Inth->dual.dof->n_dofs());
			
			dealii::FETools::interpolation_difference(
				*(Inth->dual.dof),
				*(Inth->dual.constraints),
				*(dual.z),
				*(Inth->primal.dof),
				*(Inth->primal.constraints),
				dual_weights
			);
			
			dual_weights_old.reinit(Inth->dual.dof->n_dofs());
			
			dealii::FETools::interpolation_difference(
				*(Inth->dual.dof),
				*(Inth->dual.constraints),
				*(dual.z_old),
				*(Inth->primal.dof),
				*(Inth->primal.constraints),
				dual_weights_old
			);
			// Interpolate dual.z into primal FE-room and afterwards back
			// into dual FE-room:
			// Compute I_h z_H^m,i = primal_z_dual
			
			primal_z_dual.reinit(Inth->dual.dof->n_dofs());
			dealii::FETools::back_interpolate(
				*(Inth->dual.dof),
				*(Inth->dual.constraints),
				*dual.z,
				*(Inth->primal.dof),
				*(Inth->primal.constraints),
				primal_z_dual
			);
			
			////////////////////////////////////////////////////////////////////
			// Interpolation of g (u = g on dOmega):
			BoundaryValues->set_time(Inth->t_n);
			dealii::Vector<double> vec_g(Inth->primal.dof->n_dofs());
			dealii::VectorTools::interpolate(*(Inth->primal.mapping),
											*(Inth->primal.dof),
											*BoundaryValues,
											vec_g);
			Inth->primal.constraints->distribute(vec_g);
			
			g_interpolated.reinit(Inth->dual.dof->n_dofs());
			dealii::FETools::interpolate(
				*(Inth->primal.dof),
				vec_g,
				*(Inth->dual.dof),
				*(Inth->dual.constraints),
				g_interpolated
			);
			
			////////////////////////////////////////////////////////////////////
			// Interpolation of g (u = g on dOmega) at timepoint \f$ t_{n-1} \f$:
			BoundaryValues_old->set_time(Inth->t_m);
			dealii::Vector<double> vec_g_old(Inth->primal.dof->n_dofs());
			dealii::VectorTools::interpolate(*(Inth->primal.mapping),
											*(Inth->primal.dof),
											*BoundaryValues_old,
											vec_g_old);
			Inth->primal.constraints->distribute(vec_g_old);
			
			g_interpolated_old.reinit(Inth->dual.dof->n_dofs());
			dealii::FETools::interpolate(
				*(Inth->primal.dof),
				vec_g_old,
				*(Inth->dual.dof),
				*(Inth->dual.constraints),
				g_interpolated_old
			);
			
			// 	g_interpolated.reinit(Inth->dual.dof->n_dofs());
			// 	dealii::VectorTools::interpolate(*(Inth->dual.mapping),
			// 									 *(Inth->dual.dof),
			// 									 *BoundaryValues,
			// 									 g_interpolated);
			// 	Inth->primal.constraints->distribute(g_interpolated);
			
			////////////////////////////////////////////////////////////////////
			function.f->set_time(Inth->t_n);
			function.f_old->set_time(Inth->t_m);
			
			tau_n = Inth->tau_n();
			time = Inth->t_n;
			time_old = Inth->t_m;
			
			{
				auto cell = Inth->dual.dof->begin_active();
				auto endc = Inth->dual.dof->end();
				
				for ( ; cell != endc; ++cell) {
					cell_integrals[cell] = std::numeric_limits< double >::quiet_NaN();
					
					for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<dim>::faces_per_cell; ++face_no) {
						face_integrals[cell->face(face_no)] = std::numeric_limits< double >::quiet_NaN();
					}
				}
			}
			
			std::cout << "number of cells: " << cell_integrals.size() << std::endl;
			std::cout << "number of faces: " << face_integrals.size() << std::endl;
			
			dealii::QGauss<dim> quad_cell(Inth->dual.fe->tensor_degree()+2); //GaussLobatto
			dealii::QGauss<dim-1> quad_face(Inth->dual.fe->tensor_degree()+2);
			
			dealii::WorkStream::run(
				Inth->dual.dof->begin_active(),
				Inth->dual.dof->end(),
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
					*(Inth->dual.fe),
					*(Inth->dual.mapping),
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
				auto cell = Inth->dual.dof->begin_active();
				auto endc = Inth->dual.dof->end();
				
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
			*(In_etath->x) = *(error_indicators);
			cell_integrals.clear();
			face_integrals.clear();
			++In_uth_old;
			++In_zth_old;
			++Inth_old;
		} // end else
	} //end for loop In
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
		// TODO: inhomogeneous Dirichlet boundary ---> DONE!
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

	scratch.fe_values.reinit(cell);
	
	function.f->value_list(							// calculates f
		scratch.fe_values.get_quadrature_points(),
		scratch.rhs_values
	);
	//TODO
	function.f_old->value_list(							// calculates f_old
		scratch.fe_values.get_quadrature_points(),
		scratch.rhs_values_old
	);
	
	scratch.fe_values.get_function_laplacians(		// calculates Laplace(u_h^n)
		*(dual.u),
		scratch.cell_laplacians
	);
	scratch.fe_values.get_function_laplacians(		// calculates Laplace(u_h^(n-1))
		*(dual.u),
		scratch.cell_laplacians_old
	);
	scratch.fe_values.get_function_values(			// calculates "z-phi_h" respectively n: z_h-Interp(z_h)
		dual_weights,
		scratch.dual_weights
	);
	scratch.fe_values.get_function_values(			// calculates "z-phi_h" respectively n-1: z_h-Interp(z_h)
		dual_weights_old,
		scratch.dual_weights_old
	);
	
	scratch.fe_values.get_function_values(			// calculates u_h^n
		*(dual.u),
		scratch.cell_values
	);
	scratch.fe_values.get_function_values(			// calculates u_h^(n-1)
		*(dual.u_old),
		scratch.cell_values_old
	);
	
	for (unsigned int q=0; q < scratch.fe_values.n_quadrature_points; ++q) {
		scratch.R_u_h[q] = (
					scratch.rhs_values[q] + 
					(epsilon->value(scratch.fe_values.quadrature_point(q), 0)*scratch.cell_laplacians[q])//TODO epsilon(x)
					);
		scratch.R_u_h_old[q] = (
					scratch.rhs_values_old[q] + 
					(epsilon->value(scratch.fe_values.quadrature_point(q), 0)*scratch.cell_laplacians_old[q])//TODO epsilon(x)
					);
	}
	
	// initialize local matrix with zeros
	copydata.cell = cell;
	copydata.value = 0;
	
	for (unsigned int q=0; q < scratch.fe_values.n_quadrature_points; ++q) {
		copydata.value += (
			(
			((tau_n*0.5)*(
			(
			scratch.R_u_h[q]*
			scratch.dual_weights[q]
			)
			+
			(
			scratch.R_u_h_old[q]*
			scratch.dual_weights_old[q]
			)
			)
			)
			-((scratch.cell_values[q]-scratch.cell_values_old[q])*scratch.dual_weights_old[q])
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
	BoundaryValues->set_time(time);
	BoundaryValues->value_list(scratch.fe_face_values.get_quadrature_points(),
							   scratch.boundary_values);
	//TODO second BondaryV-Function defined in exactly the same way as BV
	BoundaryValues_old->set_time(time_old);
	BoundaryValues_old->value_list(scratch.fe_face_values.get_quadrature_points(), 
							   scratch.boundary_values_old);
	
	scratch.fe_face_values.get_function_gradients(
	*(dual.z),
	scratch.dual_solution_gradients
	);
	scratch.fe_face_values.get_function_gradients(
	*(dual.z_old),
	scratch.dual_solution_gradients_old
	);
	
	scratch.fe_face_values.get_function_values(
		g_interpolated,
		scratch.g_h
	);
	scratch.fe_face_values.get_function_values(
		g_interpolated_old,
		scratch.g_h_old
	);
	
	for (unsigned int q=0;q<scratch.fe_face_values.n_quadrature_points; ++q) {
		scratch.inhom_dirichlet_difference[q] = (scratch.boundary_values[q] - scratch.g_h[q]);
		scratch.inhom_dirichlet_difference_old[q] = (scratch.boundary_values_old[q] - scratch.g_h_old[q]);
	}
	
	copydata.face  = cell->face(face_no);
	copydata.value = 0;
	
	for (unsigned int q=0; q < scratch.fe_face_values.n_quadrature_points; ++q) { // (g-g_h, epsilon*grad(z_h)*n)_dOmega
		copydata.value += 
			2.*(													//multiplied with 2, because within the estimate() function
			(tau_n*0.5)*((
			(scratch.inhom_dirichlet_difference[q] *						//the whole faces contribution will be subtracted  by the factor
			(scratch.fe_face_values.normal_vector(q)*
			(epsilon->value(scratch.fe_face_values.quadrature_point(q), 0)*		// 1/2.
			scratch.dual_solution_gradients[q])))
			+
			(scratch.inhom_dirichlet_difference_old[q] *						//the whole faces contribution will be subtracted  by the factor
			(scratch.fe_face_values.normal_vector(q)*
			(epsilon->value(scratch.fe_face_values.quadrature_point(q), 0)*		// 1/2.
			scratch.dual_solution_gradients_old[q])))
			)*
			scratch.fe_face_values.JxW(q)));
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
		*(dual.u),
		scratch.cell_grads
	);
	scratch.fe_face_values.get_function_gradients(
		*(dual.u),
		scratch.cell_grads_old
	);

	scratch.fe_face_values_neighbor.get_function_gradients(
		*(dual.u),
		scratch.neighbor_grads
	);
	scratch.fe_face_values_neighbor.get_function_gradients(
		*(dual.u),
		scratch.neighbor_grads_old
	);
	
	for (unsigned int q=0; q < scratch.fe_face_values.n_quadrature_points; ++q) {
		scratch.jump_residuals[q] = (
			(epsilon->value(scratch.fe_face_values.quadrature_point(q), 0)*
			(scratch.cell_grads[q] - scratch.neighbor_grads[q])) *
			scratch.fe_face_values.normal_vector(q)
		);
		scratch.jump_residuals_old[q] = (
			(epsilon->value(scratch.fe_face_values.quadrature_point(q), 0)*
			(scratch.cell_grads_old[q] - scratch.neighbor_grads_old[q])) *
			scratch.fe_face_values.normal_vector(q)
		);
	}
	
	scratch.fe_face_values.get_function_values(
		dual_weights,
		scratch.dual_weights
	);
	scratch.fe_face_values.get_function_values(
		dual_weights_old,
		scratch.dual_weights_old
	);
	
	copydata.face  = cell->face(face_no);
	copydata.value = 0;
	
	for (unsigned int q=0; q < scratch.fe_face_values.n_quadrature_points; ++q) {
		copydata.value += (
			((tau_n*0.5)*((scratch.jump_residuals[q] * scratch.dual_weights[q])
			+(scratch.jump_residuals_old[q] * scratch.dual_weights_old[q])
			  )
			)*
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
			*(dual.u),
			scratch.cell_grads
		);
		scratch.fe_subface_values.get_function_gradients(
			*(dual.u),
			scratch.cell_grads_old
		);
		
		scratch.fe_face_values_neighbor.get_function_gradients(
			*(dual.u),
			scratch.neighbor_grads
		);
		scratch.fe_face_values_neighbor.get_function_gradients(
			*(dual.u),
			scratch.neighbor_grads_old
		);
		
		for (unsigned int q=0; q < scratch.fe_face_values.n_quadrature_points; ++q) {
			scratch.jump_residuals[q] = (
				(epsilon->value(scratch.fe_face_values.quadrature_point(q), 0)*
				(scratch.neighbor_grads[q] - scratch.cell_grads[q])) *
				scratch.fe_face_values_neighbor.normal_vector(q)
			);
			scratch.jump_residuals_old[q] = (
				(epsilon->value(scratch.fe_face_values.quadrature_point(q), 0)*
				(scratch.neighbor_grads_old[q] - scratch.cell_grads_old[q])) *
				scratch.fe_face_values_neighbor.normal_vector(q)
			);
		}
		
		scratch.fe_face_values_neighbor.get_function_values(
			dual_weights,
			scratch.dual_weights
		);
		scratch.fe_face_values_neighbor.get_function_values(
			dual_weights_old,
			scratch.dual_weights_old
		);
		
		copydata.face  = cell->face(face_no)->child(subface_no);
		copydata.value = 0;
		
		for (unsigned int q=0; q < scratch.fe_face_values.n_quadrature_points; ++q) {
			copydata.value += (
				((tau_n*0.5)*(
				(scratch.jump_residuals[q] * scratch.dual_weights[q]) 
				+
				(scratch.jump_residuals_old[q] * scratch.dual_weights_old[q])
				)
				)*
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
