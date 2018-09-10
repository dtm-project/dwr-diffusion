/**
 * @file ErrorEstimator_DWR__cGp_dG0__cGq_cG1.tpl.cc
 *
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 *
 * @date 2018-07-19, bugfix: irregular face, UK, MPB
 * @date 2018-03-16, ErrorEstimator class for heat (final), UK, MPB
 * @date 2018-03-13, new development ErrorEstimator class for heat (begin), UK, MPB
 * @date 2018-03-13, fork from DTM++/dwr-poisson, UK
 *
 * @date 2017-11-08, ErrorEstimator class (Poisson), UK, MPB
 * @date 2016-08-16, ErrorEstimator class (Poisson), UK
 * @date 2016-08-11, Poisson / DWR from deal.II/step-14 and DTM++, UK
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

// PROJECT includes
#include <heat/ErrorEstimator/ErrorEstimator_DWR__cGp_dG0__cGq_cG1.tpl.hh>

// DTM++ includes
#include <DTM++/base/LogStream.hh>

// DEAL.II includes
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/fe/fe_tools.h>

// C++ includes

namespace heat {
namespace dwr {

namespace cGp_dG0 { // primal
namespace cGq_cG1 { // dual

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
	local_dof_indices(fe.dofs_per_cell),
	phi(fe.dofs_per_cell),
	grad_phi(fe.dofs_per_cell),
	laplace_phi(fe.dofs_per_cell),
	local_u0(fe.dofs_per_cell),
	local_z0(fe.dofs_per_cell),
	local_Rz0(fe.dofs_per_cell),
	local_um(fe.dofs_per_cell),
	local_up(fe.dofs_per_cell),
	local_zm(fe.dofs_per_cell),
	local_Rzm(fe.dofs_per_cell) {
}


/// (Struct-) Copy constructor.
template<int dim>
ErrorEstimateOnCell<dim>::ErrorEstimateOnCell(const ErrorEstimateOnCell &scratch) :
	fe_values(
		scratch.fe_values.get_mapping(),
		scratch.fe_values.get_fe(),
		scratch.fe_values.get_quadrature(),
		scratch.fe_values.get_update_flags()),
	local_dof_indices(scratch.local_dof_indices),
	phi(scratch.phi),
	grad_phi(scratch.grad_phi),
	hessian_phi(scratch.hessian_phi),
	laplace_phi(scratch.laplace_phi),
	local_u0(scratch.local_u0),
	local_z0(scratch.local_z0),
	local_Rz0(scratch.local_Rz0),
	local_um(scratch.local_um),
	local_up(scratch.local_up),
	local_zm(scratch.local_zm),
	local_Rzm(scratch.local_Rzm),
	value_f(scratch.value_f),
	value_diffusion_epsilon(scratch.value_diffusion_epsilon),
	grad_diffusion_epsilon(scratch.grad_diffusion_epsilon),
	val_R_u_kh_j(scratch.val_R_u_kh_j),
	val_u_kh_j(scratch.val_u_kh_j),
	val_z_Rz_j(scratch.val_z_Rz_j),
	JxW(scratch.JxW),
	q(scratch.q),
	d(scratch.d),
// 	k(scratch.k),
	j(scratch.j) {
}


/// (Struct-) Constructor.
template<int dim>
ErrorEstimateOnFace<dim>::ErrorEstimateOnFace(
	const dealii::FiniteElement<dim> &fe,
	const dealii::Mapping<dim> &mapping,
	const dealii::Quadrature<dim-1> &quad,
	const dealii::UpdateFlags &uflags) :
	// data structures of current face on cell (+)
	fe_values_face(mapping, fe, quad, uflags),
	fe_values_subface(mapping, fe, quad, uflags),
	local_dof_indices(fe.dofs_per_cell),
	phi(fe.dofs_per_cell),
	grad_phi(fe.dofs_per_cell),
	local_u0(fe.dofs_per_cell),
	local_z0(fe.dofs_per_cell),
	local_Rz0(fe.dofs_per_cell),
	// data structures of neighboring face of cell (-)
	neighbor_fe_values_face(mapping, fe, quad, dealii::UpdateFlags::update_gradients),
	neighbor_local_dof_indices(fe.dofs_per_cell),
	neighbor_grad_phi(fe.dofs_per_cell),
	neighbor_local_u0(fe.dofs_per_cell){
}


/// (Struct-) Copy constructor.
template<int dim>
ErrorEstimateOnFace<dim>::ErrorEstimateOnFace(const ErrorEstimateOnFace &scratch) :
	// data structures of current face on cell (+)
	fe_values_face(
		scratch.fe_values_face.get_mapping(),
		scratch.fe_values_face.get_fe(),
		scratch.fe_values_face.get_quadrature(),
		scratch.fe_values_face.get_update_flags()
	),
	fe_values_subface(
		scratch.fe_values_subface.get_mapping(),
		scratch.fe_values_subface.get_fe(),
		scratch.fe_values_subface.get_quadrature(),
		scratch.fe_values_subface.get_update_flags()
	),
	local_dof_indices(scratch.local_dof_indices),
	phi(scratch.phi),
	grad_phi(scratch.grad_phi),
	normal_vector(scratch.normal_vector),
	local_u0(scratch.local_u0),
	local_z0(scratch.local_z0),
	local_Rz0(scratch.local_Rz0),
	// data structures of neighboring face of cell (-)
	neighbor_fe_values_face(
		scratch.neighbor_fe_values_face.get_mapping(),
		scratch.neighbor_fe_values_face.get_fe(),
		scratch.neighbor_fe_values_face.get_quadrature(),
		scratch.neighbor_fe_values_face.get_update_flags()
	),
	neighbor_local_dof_indices(scratch.neighbor_local_dof_indices),
	neighbor_grad_phi(scratch.neighbor_grad_phi),
	neighbor_local_u0(scratch.neighbor_local_u0),
	// other
	value_diffusion_epsilon(scratch.value_diffusion_epsilon),
	value_u_D(scratch.value_u_D),
	val_uh(scratch.val_uh),
	val_grad_zh(scratch.val_grad_zh),
	val_face_jump_grad_u(scratch.val_face_jump_grad_u),
	val_z_Rz_j(scratch.val_z_Rz_j),
	JxW(scratch.JxW),
	q(scratch.q),
// 	k(scratch.k),
	j(scratch.j),
	subface_no(scratch.subface_no) {
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
	face(scratch.face),
	face_no(scratch.face_no) {
}

}

namespace CopyData {

/// (Struct-) Copy constructor.
template<int dim>
ErrorEstimateOnCell<dim>::ErrorEstimateOnCell(const ErrorEstimateOnCell &copydata) :
	cell(copydata.cell),
	value(copydata.value) {
}


/// (Struct-) Copy constructor.
template<int dim>
ErrorEstimateOnFace<dim>::ErrorEstimateOnFace(const ErrorEstimateOnFace &copydata) :
	face(copydata.face),
	value(copydata.value) {
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
estimate(
	std::shared_ptr< dealii::Function<dim> > _diffusion_epsilon,
	std::shared_ptr< dealii::Function<dim> > _f,
	std::shared_ptr< dealii::Function<dim> > _u_D,
	std::shared_ptr< dealii::Function<dim> > _u_0,
	std::shared_ptr< heat::Grid_DWR<dim,1> > _grid,
	std::shared_ptr< DTM::types::storage_data_vectors<1> > _u,
	std::shared_ptr< DTM::types::storage_data_vectors<2> > _z,
	std::shared_ptr< DTM::types::storage_data_vectors<1> > _eta
) {
	Assert(_diffusion_epsilon.use_count(), dealii::ExcNotInitialized());
	function.diffusion_epsilon = _diffusion_epsilon;
	
	Assert(_f.use_count(), dealii::ExcNotInitialized());
	function.f = _f;
	
	Assert(_u_D.use_count(), dealii::ExcNotInitialized());
	function.u_D = _u_D;
	
	Assert(_u_0.use_count(), dealii::ExcNotInitialized());
	function.u_0 = _u_0;
	
	Assert(_grid.use_count(), dealii::ExcNotInitialized());
	grid = _grid;
	
	Assert(_u.use_count(), dealii::ExcNotInitialized());
	primal.storage.u = _u;
	
	Assert(_z.use_count(), dealii::ExcNotInitialized());
	dual.storage.z = _z;
	
	Assert(_eta.use_count(), dealii::ExcNotInitialized());
	error_estimator.storage.eta = _eta;
	
	Assert(error_estimator.storage.eta->size(), dealii::ExcNotInitialized());
	Assert(error_estimator.storage.eta->front().x[0].use_count(), dealii::ExcNotInitialized());
	
	
	////////////////////////////////////////////////////////////////////////////
	// do estimate errors over \Omega x (0,T) loop
	//
	
	Assert(grid->slabs.size(), dealii::ExcNotInitialized());
	auto slab = grid->slabs.begin();
	
	auto u = primal.storage.u->begin();
	auto z = dual.storage.z->begin();
	auto eta = error_estimator.storage.eta->begin();
	
	while (slab != grid->slabs.end()) {
		// local time variables
		const double tm = slab->t_m;
		const double t0 = tm + slab->tau_n()/2.;
		
		// local tau_n (used in the local assembly functions internally)
		tau_n = slab->tau_n();
		
		// interpolate primal solution u^-(t_m) to dual solution space
		dual_um_on_tm = std::make_shared< dealii::Vector<double> > ();
		dual_um_on_tm->reinit( slab->dual.dof->n_dofs() );
		
		if (slab == grid->slabs.begin()) {
			// n == 1: interpolate initial value function u_0 to dual space
			
			auto primal_um_on_tm = std::make_shared< dealii::Vector<double> > ();
			primal_um_on_tm->reinit( slab->primal.dof->n_dofs() );
			
			function.u_0->set_time(tm);
			dealii::VectorTools::interpolate(
				*slab->primal.mapping,
				*slab->primal.dof,
				*function.u_0,
				*primal_um_on_tm
			);
			// call hanging nodes to make the result continuous again (Note: after the 
			// first dwr-loop the initial grid could have hanging nodes)
			slab->primal.constraints->distribute(*primal_um_on_tm);
			
			dealii::FETools::interpolate(
				// primal solution
				*slab->primal.dof,
				*primal_um_on_tm,
				// dual solution
				*slab->dual.dof,
				*slab->dual.constraints,
				*dual_um_on_tm
			);
			
			primal_um_on_tm = nullptr;
		}
		else {
			// n > 1
			
			//   get u^-(t_m) from:   Omega_h^primal x I_{n-1} (t_{n-1})
			//   (1) interpolated to: Omega_h^primal x I_{n} (t_m) => primal_um_on_tm
			//   (2) interpolated to: Omega_h^dual x I_{n} (t_m)   => dual_um_on_tm
			
			// (1) interpolate_to_different_mesh (in primal):
			//     - needs the same fe: dof1.get_fe() = dof2.get_fe()
			//     - allow different triangulations: dof1.get_tria() != dof2.get_tria()
			auto primal_um_on_tm = std::make_shared< dealii::Vector<double> > ();
			primal_um_on_tm->reinit( slab->primal.dof->n_dofs() );
			
			dealii::VectorTools::interpolate_to_different_mesh(
				// solution on I_{n-1}:
				*std::prev(slab)->primal.dof,
				*std::prev(u)->x[0],
				// solution on I_n:
				*slab->primal.dof,
				*slab->primal.constraints,
				*primal_um_on_tm
			);
			
			// (2) interpolate primal -> dual:
			//     - needs the same tria: dof1.get_tria() == dof2.get_tria()
			//     - allow different FE-spaces: dof1.get_fe() != dof2.get_fe()
			dealii::FETools::interpolate(
				// primal solution
				*slab->primal.dof,
				*primal_um_on_tm,
				// dual solution
				*slab->dual.dof,
				*slab->dual.constraints,
				*dual_um_on_tm
			);
			
			primal_um_on_tm = nullptr;
		}
		
		dual_get_z_t_on_slab(slab, z, tm, dual_z_on_tm);
		
		dual_get_z_t_on_slab_after_restriction_to_primal_space(
			slab, z, tm, dual_Rz_on_tm
		);
		
		dual_get_z_t_on_slab(slab, z, t0, dual_z_on_t0);
		
		dual_get_z_t_on_slab_after_restriction_to_primal_space(
			slab, z, t0, dual_Rz_on_t0
		);
		
		dual_get_u_t_on_slab(slab, u, t0, dual_u_on_t0);
		
		dual_up_on_tm = dual_u_on_t0;
		
		// interpolate boundary values u_D(t0) to dual solution space
		dual_uD_on_t0 = std::make_shared< dealii::Vector<double> > ();
		dual_uD_on_t0->reinit( slab->dual.dof->n_dofs() );
		
		{
			// get boundary values on Dirichlet nodes by interpolation
			auto boundary_values = std::make_shared< std::map<dealii::types::global_dof_index, double> > ();
			
			Assert(function.u_D.use_count(), dealii::ExcNotInitialized());
			function.u_D->set_time(t0);
			
			dealii::VectorTools::interpolate_boundary_values(
				*slab->primal.mapping,
				*slab->primal.dof,
				static_cast< dealii::types::boundary_id > (
					heat::types::boundary_id::Dirichlet
				),
				*function.u_D,
				*boundary_values
			);
			
			// copy Dirichlet node values into a vector for further use
			auto primal_uD_on_t0 = std::make_shared< dealii::Vector<double> > ();
			primal_uD_on_t0->reinit( slab->primal.dof->n_dofs() );
			*primal_uD_on_t0 = 0.;
			for (auto &boundary_value : *boundary_values) {
				(*primal_uD_on_t0) [boundary_value.first] = boundary_value.second;
			}
			
			boundary_values = nullptr;
			
			// interpolate uD values from primal to dual solution space
			dealii::FETools::interpolate(
				// primal solution
				*slab->primal.dof,
				*primal_uD_on_t0,
				// dual solution
				*slab->dual.dof,
				*slab->dual.constraints,
				*dual_uD_on_t0
			);
			
			primal_uD_on_t0 = nullptr;
		}
		
		// init storage for slab problem
		{
			Assert(
				(cell_integrals.size()==0),
				dealii::ExcMessage("cell_integrals must be empty")
			);
			
			Assert(
				(face_integrals.size()==0),
				dealii::ExcMessage("face_integrals must be empty")
			);
			
			auto cell = slab->dual.dof->begin_active();
			auto endc = slab->dual.dof->end();
			for ( ; cell != endc; ++cell) {
				cell_integrals[cell] = std::numeric_limits< double >::quiet_NaN();
				for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<dim>::faces_per_cell; ++face_no) {
					face_integrals[cell->face(face_no)] = std::numeric_limits< double >::quiet_NaN();
				}
			}
		}
		
		// assemble slab problem
		dealii::QGauss<dim> quad_cell(slab->dual.fe->tensor_degree()+1);
		dealii::QGauss<dim-1> quad_face(slab->dual.fe->tensor_degree()+1);
		
		// set time variable for force function f
		function.f->set_time(t0);
		
		dealii::WorkStream::run(
			slab->dual.dof->begin_active(),
			slab->dual.dof->end(),
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
				*slab->dual.fe,
				*slab->dual.mapping,
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
		
		// copy data
		Assert(eta->x[0].use_count(), dealii::ExcNotInitialized());
		Assert(eta->x[0]->size(), dealii::ExcNotInitialized());
		
		#ifdef DEBUG
		// init *eta->x[0] with quiet_NaN (only for debugging)
		for (unsigned int cell_no{0}; cell_no < eta->x[0]->size(); ++cell_no) {
			(*eta->x[0])[cell_no] = std::numeric_limits<double>::quiet_NaN();
		}
		#endif
		
		{
			// copy the cell and face data assembled
			// from the internal map<> data structure of ErrorEstimator
			// to a dealii::Vector for further usage with deal.II functions
			auto cell = slab->dual.dof->begin_active();
			auto endc = slab->dual.dof->end();
			
			for (unsigned int cell_no{0}; cell != endc; ++cell, ++cell_no) {
				// initialize with the volume integrals
				(*eta->x[0])[cell_no] = cell_integrals[cell];
				
				// add the contributions from the spatial face assemblies
				for (unsigned int face_no{0};
					face_no < dealii::GeometryInfo<dim>::faces_per_cell; ++face_no) {
					Assert(
						(face_integrals.find(cell->face(face_no)) != face_integrals.end()),
						dealii::ExcMessage(
							"Your face iterator does not exist in your map(face_it,double). \
							Please check if you have assembled this error integral."
						)
					);
					
					// NOTE: Here, we can not distingush between an interior face
					//       and boundary face for performance reasons.
					//
					//       Thus, we need to substract 1/2 of the assembly for
					//       all face assemblies, such that the contributions on
					//       boundary faces must be weighted with a factor of 2.0
					//       during the assembly in assemble_error_on_boundary_face() .
					(*eta->x[0])[cell_no] -= (1./2.) * face_integrals[cell->face(face_no)];
				}
			}
		}
		
		#ifdef DEBUG
		// check if error_indicators vector has only valid entries
		for (unsigned int cell_no{0}; cell_no < eta->x[0]->size(); ++cell_no) {
			Assert(
				!std::isnan((*eta->x[0])[cell_no]),
				dealii::ExcMessage("Your error indicator has quiet_NaN entries. \
				Please check if you have assembled cell_integrals and face_integrals correctly.")
			);
		}
		#endif
		
		////////////////////////////////////////////////////////////////////////
		// prepare next I_n slab problem:
		//
		
		cell_integrals.clear();
		face_integrals.clear();
		
		++slab;
		++u; ++z; ++eta;
	}
}


////////////////////////////////////////////////////////////////////////////////
//
//

template<int dim>
void
ErrorEstimator<dim>::
primal_get_u_t_on_slab(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &u,
	const double &t,
	std::shared_ptr< dealii::Vector<double> > &u_result
) {
	Assert( (t > slab->t_m), dealii::ExcInvalidState() );
	Assert( (t <= slab->t_n), dealii::ExcInvalidState() );
	
	u_result = std::make_shared< dealii::Vector<double> > ();
	u_result->reinit(
		slab->primal.dof->n_dofs()
	);
	
	// get time _t on reference time interval I_hat = (0,1)
	[[maybe_unused]] const double _t{ (t - slab->t_m) / slab->tau_n() };
	
	// trial basis functions evaluation on reference interval
	const double zeta0{1.};
	
	u_result->equ(zeta0, *u->x[0]);
}


template<int dim>
void
ErrorEstimator<dim>::
dual_get_u_t_on_slab(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &u,
	const double &t,
	std::shared_ptr< dealii::Vector<double> > &u_result
) {
	// evaluate dof vector u(t) on primal solution space:
	std::shared_ptr< dealii::Vector<double> > primal_u_t;
	primal_get_u_t_on_slab(slab, u, t, primal_u_t);
	
	u_result = std::make_shared< dealii::Vector<double> > ();
	u_result->reinit(
		slab->dual.dof->n_dofs()
	);
	
	// interpolate primal dof vector to dual dof vector
	dealii::FETools::interpolate(
		// primal solution
		*slab->primal.dof,
		*primal_u_t,
		// dual solution
		*slab->dual.dof,
		*slab->dual.constraints,
		*u_result
	);
}


template<int dim>
void
ErrorEstimator<dim>::
dual_get_z_t_on_slab(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<2>::iterator &z,
	const double &t,
	std::shared_ptr< dealii::Vector<double> > &z_result
) {
	Assert( (t >= slab->t_m), dealii::ExcInvalidState() );
	Assert( (t <= slab->t_n), dealii::ExcInvalidState() );
	
	z_result = std::make_shared< dealii::Vector<double> > ();
	z_result->reinit(
		slab->dual.dof->n_dofs()
	);
	
	// get time _t on reference time interval I_hat = (0,1)
	const double _t{ (t - slab->t_m) / slab->tau_n() };
	
	// trial basis functions evaluation on reference interval
	const double xi0{ 1. - _t };
	const double xi1{ _t };
	
	z_result->equ(xi0, *z->x[0]);
	z_result->add(xi1, *z->x[1]);
}


template<int dim>
void
ErrorEstimator<dim>::
dual_get_z_t_on_slab_after_restriction_to_primal_space(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<2>::iterator &z,
	const double &t,
	std::shared_ptr< dealii::Vector<double> > &z_result) {
	////////////////////////////////////////////////////////////////////////////
	// NOTE: this function must know the time discretisation of
	//       the primal problem!
	//
	
	// result is: z^dual(t) = I^dual{ [R^primal(z^dual)] (t) }
	
	Assert( (t >= slab->t_m), dealii::ExcInvalidState() );
	Assert( (t <= slab->t_n), dealii::ExcInvalidState() );
	
	////////////////////////////////////////////////////////////////////////////
	// compute the restriction z^primal(t) = R^primal(z^dual) (t)
	//
	
	/// primal_z_t = R^primal(z^dual) (t)
	std::shared_ptr< dealii::Vector<double> > primal_z_t;
	{
		// get dual_z_on_primal_t0 for t0 of primal problem
		auto dual_z_on_primal_t0 = std::make_shared< dealii::Vector<double> > ();
		dual_z_on_primal_t0->reinit(
			slab->dual.dof->n_dofs()
		);
		
		const double _t{ (t - slab->t_m) / slab->tau_n() };
		
		// evaluate dual trial functions in time on _t
		const double xi0{ 1.-_t };
		const double xi1{ _t };
		
		// evaluate z^dual on time-dof t0 of the primal problem
		dual_z_on_primal_t0->equ(xi0, *z->x[0]);
		dual_z_on_primal_t0->add(xi1, *z->x[1]);
		
		// interpolate dual_z_on_primal_t0 to primal_z_on_primal_t0
		auto primal_z_on_primal_t0 = std::make_shared< dealii::Vector<double> > ();
		primal_z_on_primal_t0->reinit(
			slab->primal.dof->n_dofs()
		);
		
		dealii::FETools::interpolate(
			// dual solution
			*slab->dual.dof,
			*dual_z_on_primal_t0,
			// primal solution
			*slab->primal.dof,
			*slab->primal.constraints,
			*primal_z_on_primal_t0
		);
		
		// evaluate solution for t \in I_n on primal time discretisation:
		// NOTE: primal problem: dG(0)-Q_G(1) discretisation (constant in time):
		//   primal_z_t = zeta0(_t) * primal_z_on_primal_t0
		//              = 1 * primal_z_on_primal_t0
		primal_z_t = primal_z_on_primal_t0;
	}
	
	// interpolate primal_z_t to z_result (on dual space for fixed t)
	Assert(primal_z_t.use_count(), dealii::ExcNotInitialized());
	Assert(primal_z_t->size(), dealii::ExcNotInitialized());
	
	z_result = std::make_shared< dealii::Vector<double> > ();
	z_result->reinit(
		slab->dual.dof->n_dofs()
	);
	
	dealii::FETools::interpolate(
		// primal solution
		*slab->primal.dof,
		*primal_z_t,
		// dual solution
		*slab->dual.dof,
		*slab->dual.constraints,
		*z_result
	);
}


////////////////////////////////////////////////////////////////////////////////
//
//

template<int dim>
void
ErrorEstimator<dim>::
assemble_local_error(
	const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
	Assembly::Scratch::ErrorEstimates<dim> &scratch,
	Assembly::CopyData::ErrorEstimates<dim> &copydata) {
	
	////////////////////////////////////////////////////////////////////////
	// cell integrals:
	//
	assemble_error_on_cell(cell, scratch.cell, copydata.cell);
	
	////////////////////////////////////////////////////////////////////////
	// face integrals:
	//
	for (scratch.face_no=0;
		scratch.face_no < dealii::GeometryInfo<dim>::faces_per_cell;
		++scratch.face_no) {
		////////////////////////////////////////////////////////////////////////
		// handle Dirichlet boundary faces
		if (cell->face(scratch.face_no)->at_boundary()) {
			if (cell->face(scratch.face_no)->boundary_id() ==
				static_cast<dealii::types::boundary_id> (
						heat::types::boundary_id::Dirichlet) ) {
				// only on Dirichlet type boundary face
				assemble_error_on_boundary_face(
					cell,
					scratch.face_no,
					scratch.face,
					copydata.face
				);
			}
			
			continue;
		}
		
		////////////////////////////////////////////////////////////////////////
		// interior faces only:
		Assert(!cell->face(scratch.face_no)->at_boundary(), dealii::ExcInvalidState());
		
		// skip face with same refinement level where the neighbor cell index
		// is smaller than this ones
		if ((cell->index() > cell->neighbor(scratch.face_no)->index()) &&
			(cell->neighbor(scratch.face_no)->has_children() == false) &&
			(cell->level() == cell->neighbor(scratch.face_no)->level())) {
			// integrate face value from the neighbor cell
			continue;
		}
		
		// integrate from coarser cell
		if (cell->level() > cell->neighbor(scratch.face_no)->level()) {
			continue;
		}
		
		if (cell->face(scratch.face_no)->has_children() == false) {
			assemble_error_on_regular_face(
				cell,
				scratch.face_no,
				scratch.face,
				copydata.face
			);
		}
		else {
			assemble_error_on_irregular_face(
				cell,
				scratch.face_no,
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
	
	// fetch local dof data
	cell->get_dof_indices(scratch.local_dof_indices);
	
	for (scratch.j=0; scratch.j < scratch.fe_values.get_fe().dofs_per_cell;
		++scratch.j) {
		scratch.local_u0[scratch.j] =
			(*dual_u_on_t0)[ scratch.local_dof_indices[scratch.j] ];
	}
	
	for (scratch.j=0; scratch.j < scratch.fe_values.get_fe().dofs_per_cell;
		++scratch.j) {
		scratch.local_z0[scratch.j] =
			(*dual_z_on_t0)[ scratch.local_dof_indices[scratch.j] ];
	}
	
	for (scratch.j=0; scratch.j < scratch.fe_values.get_fe().dofs_per_cell;
		++scratch.j) {
		scratch.local_Rz0[scratch.j] =
			(*dual_Rz_on_t0)[ scratch.local_dof_indices[scratch.j] ];
	}
	
	for (scratch.j=0; scratch.j < scratch.fe_values.get_fe().dofs_per_cell;
		++scratch.j) {
		scratch.local_um[scratch.j] =
			(*dual_um_on_tm)[ scratch.local_dof_indices[scratch.j] ];
	}
	
	for (scratch.j=0; scratch.j < scratch.fe_values.get_fe().dofs_per_cell;
		++scratch.j) {
		scratch.local_up[scratch.j] =
			(*dual_up_on_tm)[ scratch.local_dof_indices[scratch.j] ];
	}
	
	for (scratch.j=0; scratch.j < scratch.fe_values.get_fe().dofs_per_cell;
		++scratch.j) {
		scratch.local_zm[scratch.j] =
			(*dual_z_on_tm)[ scratch.local_dof_indices[scratch.j] ];
	}
	
	for (scratch.j=0; scratch.j < scratch.fe_values.get_fe().dofs_per_cell;
		++scratch.j) {
		scratch.local_Rzm[scratch.j] =
			(*dual_Rz_on_tm)[ scratch.local_dof_indices[scratch.j] ];
	}
	
	// initialize copydata
	copydata.cell = cell;
	copydata.value = 0.;
	
	// assemble cell terms
	for (scratch.q=0; scratch.q < scratch.fe_values.n_quadrature_points; ++scratch.q) {
		scratch.JxW = scratch.fe_values.JxW(scratch.q);
		
		// loop over all basis functions to get the shape values
		for (scratch.j=0; scratch.j < scratch.fe_values.get_fe().dofs_per_cell;
			++scratch.j) {
			scratch.phi[scratch.j] =
				scratch.fe_values.shape_value_component(scratch.j,scratch.q,0);
		}
		
		for (scratch.j=0; scratch.j < scratch.fe_values.get_fe().dofs_per_cell;
			++scratch.j) {
			scratch.grad_phi[scratch.j] =
				scratch.fe_values.shape_grad(scratch.j,scratch.q);
		}
		
		for (scratch.j=0; scratch.j < scratch.fe_values.get_fe().dofs_per_cell;
			++scratch.j) {
			scratch.hessian_phi = scratch.fe_values.shape_hessian(scratch.j,scratch.q);
			
			scratch.laplace_phi[scratch.j] = 0.;
			for (scratch.d=0; scratch.d < dim; ++scratch.d) {
				scratch.laplace_phi[scratch.j] +=
					scratch.hessian_phi[scratch.d][scratch.d];
			}
		}
		
		scratch.value_f = function.f->value(
			scratch.fe_values.quadrature_point(scratch.q), 0
		);
		
		scratch.value_diffusion_epsilon =
			function.diffusion_epsilon->value(
				scratch.fe_values.quadrature_point(scratch.q), 0
			);
		
		scratch.grad_diffusion_epsilon =
			function.diffusion_epsilon->gradient(
				scratch.fe_values.quadrature_point(scratch.q), 0
			);
		
		scratch.val_R_u_kh_j = 0.;
		scratch.val_u_kh_j = 0.;
		scratch.val_z_Rz_j = 0.;
		for (scratch.j=0; scratch.j < scratch.fe_values.get_fe().dofs_per_cell;
			++scratch.j) {
			// - 0 <= here => - density(x_q,t_q) * \partial_t u * 1/tau_n
			scratch.val_R_u_kh_j +=
				scratch.local_u0[scratch.j]
				* (scratch.grad_phi[scratch.j] * scratch.grad_diffusion_epsilon)
				+ scratch.value_diffusion_epsilon
				* scratch.local_u0[scratch.j] * scratch.laplace_phi[scratch.j];
			
			scratch.val_u_kh_j +=
				(scratch.local_up[scratch.j] - scratch.local_um[scratch.j])
				* scratch.phi[scratch.j];
				
			scratch.val_z_Rz_j +=
				(scratch.local_z0[scratch.j] - scratch.local_Rz0[scratch.j])
				* scratch.phi[scratch.j];
		}
		
		// \int_{I_n} ... :
		copydata.value += (
			// R(u_kh):
			( scratch.value_f + scratch.val_R_u_kh_j )
			// z_h - Rz_h:
			* scratch.val_z_Rz_j
			* tau_n
			* scratch.JxW
		);
		
		scratch.val_z_Rz_j = 0.;
		for (scratch.j=0; scratch.j < scratch.fe_values.get_fe().dofs_per_cell;
			++scratch.j) {
			scratch.val_z_Rz_j +=
				(scratch.local_zm[scratch.j] - scratch.local_Rzm[scratch.j])
				* scratch.phi[scratch.j];
		} // for j
		
		// - [ u_kh(t_m) ] * ( z_h(t_m) - I(R(z_h(t_m))) )
		copydata.value += (
			- scratch.val_u_kh_j
			* scratch.val_z_Rz_j
			* scratch.JxW
		);
	} // for q
	
	Assert(
		std::isnan(cell_integrals[copydata.cell]),
		dealii::ExcMessage("ErrorEstimator: you access the same cell at least two times")
	);
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
	
	scratch.fe_values_face.reinit(cell, face_no);
	
	// fetch local dof data ( K^+ / F^+ )
	cell->get_dof_indices(scratch.local_dof_indices);
	
	for (scratch.j=0; scratch.j < scratch.fe_values_face.get_fe().dofs_per_cell;
		++scratch.j) {
		// fetch interpolated u_D ("aka g_h") local dof data into ( K^+ / F^+ )-structure
		scratch.local_u0[scratch.j] =
			(*dual_uD_on_t0)[ scratch.local_dof_indices[scratch.j] ];
	}
	
	for (scratch.j=0; scratch.j < scratch.fe_values_face.get_fe().dofs_per_cell;
		++scratch.j) {
		scratch.local_z0[scratch.j] =
			(*dual_z_on_t0)[ scratch.local_dof_indices[scratch.j] ];
	}
	
	// initialize copydata
	copydata.face = cell->face(face_no);
	copydata.value = 0.;
	
	// assemble face terms
	for (scratch.q=0; scratch.q < scratch.fe_values_face.n_quadrature_points;
		++scratch.q) {
		scratch.JxW = scratch.fe_values_face.JxW(scratch.q);
		scratch.normal_vector = scratch.fe_values_face.normal_vector(scratch.q);
		
		// compute g_h(x_q) and grad z_h(x_q) * n:
		scratch.val_uh=0.;
		scratch.val_grad_zh=0.;
		for (scratch.j=0; scratch.j < scratch.fe_values_face.get_fe().dofs_per_cell;
			++scratch.j) {
			// NOTE: we assemble here "g_h", not u_h; cf. above.
			scratch.val_uh += scratch.local_u0[scratch.j]
				*scratch.fe_values_face.shape_value_component(scratch.j,scratch.q,0);
			
			scratch.val_grad_zh += scratch.local_z0[scratch.j]
				* scratch.fe_values_face.shape_grad(scratch.j,scratch.q)
				* scratch.normal_vector;
		}
		
		// fetch function value(s)
		scratch.value_diffusion_epsilon = function.diffusion_epsilon->value(
			scratch.fe_values_face.quadrature_point(scratch.q), 0
		);
		
		scratch.value_u_D = function.u_D->value(
			scratch.fe_values_face.quadrature_point(scratch.q), 0
		);
		
		// \int_{I_n} ... :
		copydata.value += (
			// finally we use 1/2 of all face values (interior and boundary faces!),
			// thus we need to double the value in the assembly here
			2.0
			// u_D - I^dual(I^primal u_D)
			* (scratch.value_u_D - scratch.val_uh)
			// diffusion_epsilon(x_q) * grad z_h * n
			* scratch.value_diffusion_epsilon * scratch.val_grad_zh 
			* tau_n
			* scratch.JxW
		);
	}
	
	Assert(
		std::isnan(face_integrals[copydata.face]),
		dealii::ExcMessage("ErrorEstimator: you access the same boundary face at least two times")
	);
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
	
	scratch.fe_values_face.reinit(cell, face_no);
	
	scratch.neighbor_fe_values_face.reinit(
		cell->neighbor(face_no),
		cell->neighbor_of_neighbor(face_no)
	);
	
	// fetch local dof data ( K^+ / F^+ )
	cell->get_dof_indices(scratch.local_dof_indices);
	
	for (scratch.j=0; scratch.j < scratch.fe_values_face.get_fe().dofs_per_cell;
		++scratch.j) {
		scratch.local_u0[scratch.j] =
			(*dual_u_on_t0)[ scratch.local_dof_indices[scratch.j] ];
	}
	
	for (scratch.j=0; scratch.j < scratch.fe_values_face.get_fe().dofs_per_cell;
		++scratch.j) {
		scratch.local_z0[scratch.j] =
			(*dual_z_on_t0)[ scratch.local_dof_indices[scratch.j] ];
	}
	
	for (scratch.j=0; scratch.j < scratch.fe_values_face.get_fe().dofs_per_cell;
		++scratch.j) {
		scratch.local_Rz0[scratch.j] =
			(*dual_Rz_on_t0)[ scratch.local_dof_indices[scratch.j] ];
	}
	
	// fetch local dof data ( K^- / F^- )
	cell->neighbor(face_no)->get_dof_indices(scratch.neighbor_local_dof_indices);
	
	for (scratch.j=0;
		scratch.j < scratch.neighbor_fe_values_face.get_fe().dofs_per_cell;
		++scratch.j) {
		scratch.neighbor_local_u0[scratch.j] =
			(*dual_u_on_t0)[ scratch.neighbor_local_dof_indices[scratch.j] ];
	}
	
	// initialize copydata
	copydata.face = cell->face(face_no);
	copydata.value = 0.;
	
	// assemble cell terms
	for (scratch.q=0; scratch.q < scratch.fe_values_face.n_quadrature_points;
		++scratch.q) {
		scratch.JxW = scratch.fe_values_face.JxW(scratch.q);
		scratch.normal_vector = scratch.fe_values_face.normal_vector(scratch.q);
		
		// loop over all basis functions to get the shape values (K^+)
		for (scratch.j=0; scratch.j < scratch.fe_values_face.get_fe().dofs_per_cell;
			++scratch.j) {
			scratch.phi[scratch.j] =
				scratch.fe_values_face.shape_value_component(scratch.j,scratch.q,0);
		}
		
		for (scratch.j=0; scratch.j < scratch.fe_values_face.get_fe().dofs_per_cell;
			++scratch.j) {
			scratch.grad_phi[scratch.j] =
				scratch.fe_values_face.shape_grad(scratch.j,scratch.q);
		}
		
		// loop over all basis functions to get the shape values (K^-)
		for (scratch.j=0;
			scratch.j < scratch.neighbor_fe_values_face.get_fe().dofs_per_cell;
			++scratch.j) {
			scratch.neighbor_grad_phi[scratch.j] =
				scratch.neighbor_fe_values_face.shape_grad(scratch.j,scratch.q);
		}
		
		// fetch function value(s)
		scratch.value_diffusion_epsilon = function.diffusion_epsilon->value(
			scratch.fe_values_face.quadrature_point(scratch.q), 0
		);
		
		Assert(
			(scratch.fe_values_face.get_fe().dofs_per_cell==
			scratch.neighbor_fe_values_face.get_fe().dofs_per_cell),
			dealii::ExcMessage("different fe.p between neighboring cells is not allowed here")
		);
		
		scratch.val_face_jump_grad_u=0.;
		scratch.val_z_Rz_j=0.;
		for (scratch.j=0;
			scratch.j < scratch.fe_values_face.get_fe().dofs_per_cell; ++scratch.j) {
			scratch.val_face_jump_grad_u += (
				scratch.local_u0[scratch.j] * scratch.grad_phi[scratch.j]
				- scratch.neighbor_local_u0[scratch.j]
				* scratch.neighbor_grad_phi[scratch.j]
			) * scratch.normal_vector;
			
			scratch.val_z_Rz_j +=
				(scratch.local_z0[scratch.j] - scratch.local_Rz0[scratch.j])
				* scratch.phi[scratch.j];
		}
		
		// \int_{I_n} ... :
		copydata.value += (
			scratch.value_diffusion_epsilon
			* scratch.val_face_jump_grad_u
			// z_h - Rz_h:
			* scratch.val_z_Rz_j
			* tau_n
			* scratch.JxW
		);
	} // for q
	
	Assert(
		std::isnan(face_integrals[copydata.face]),
		dealii::ExcMessage("ErrorEstimator: you access the same regular face at least two times")
	);
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
	
	for (scratch.subface_no=0; scratch.subface_no < cell->face(face_no)->n_children();
		++scratch.subface_no) {
		
		Assert(
			(cell->neighbor_child_on_subface(face_no, scratch.subface_no)->face(
				cell->neighbor_of_neighbor(face_no) ) ==
			cell->face(face_no)->child(scratch.subface_no)),
			dealii::ExcInternalError()
		);
		
		scratch.fe_values_subface.reinit(cell, face_no, scratch.subface_no);
		
		scratch.neighbor_fe_values_face.reinit(
			cell->neighbor_child_on_subface(face_no, scratch.subface_no),
			cell->neighbor_of_neighbor(face_no)
		);
		
		// fetch local dof data ( K^+ / subface F^+ )
		Assert(cell->active(), dealii::ExcInternalError());
		cell->get_dof_indices(scratch.local_dof_indices);
		
		for (scratch.j=0;
			scratch.j < scratch.fe_values_subface.get_fe().dofs_per_cell;
			++scratch.j) {
			scratch.local_u0[scratch.j] =
				(*dual_u_on_t0)[ scratch.local_dof_indices[scratch.j] ];
		}
		
		for (scratch.j=0;
			scratch.j < scratch.fe_values_subface.get_fe().dofs_per_cell;
			++scratch.j) {
			scratch.local_z0[scratch.j] =
				(*dual_z_on_t0)[ scratch.local_dof_indices[scratch.j] ];
		}
		
		for (scratch.j=0;
			scratch.j < scratch.fe_values_subface.get_fe().dofs_per_cell;
			++scratch.j) {
			scratch.local_Rz0[scratch.j] =
				(*dual_Rz_on_t0)[ scratch.local_dof_indices[scratch.j] ];
		}
		
		// fetch local dof data ( K^- / F^- )
		Assert(
			cell->neighbor_child_on_subface(face_no, scratch.subface_no)->active(),
			dealii::ExcInternalError()
		);
		cell->neighbor_child_on_subface(face_no, scratch.subface_no)->get_dof_indices(
			scratch.neighbor_local_dof_indices
		);
		
		for (scratch.j=0;
			scratch.j < scratch.neighbor_fe_values_face.get_fe().dofs_per_cell;
			++scratch.j) {
			scratch.neighbor_local_u0[scratch.j] =
				(*dual_u_on_t0)[ scratch.neighbor_local_dof_indices[scratch.j] ];
		}
		
		// initialize copydata
		copydata.face = cell->face(face_no)->child(scratch.subface_no);
		copydata.value = 0.;
		
		// assemble cell terms
		for (scratch.q=0; scratch.q < scratch.fe_values_subface.n_quadrature_points;
			++scratch.q) {
			scratch.JxW = scratch.fe_values_subface.JxW(scratch.q);
			scratch.normal_vector = scratch.fe_values_subface.normal_vector(scratch.q);
			
			// loop over all basis functions to get the shape values (K^+ subface)
			for (scratch.j=0;
				scratch.j < scratch.fe_values_subface.get_fe().dofs_per_cell;
				++scratch.j) {
				scratch.phi[scratch.j] =
					scratch.fe_values_subface.shape_value_component(
						scratch.j,scratch.q,0
					);
			}
			
			for (scratch.j=0;
				scratch.j < scratch.fe_values_subface.get_fe().dofs_per_cell;
				++scratch.j) {
				scratch.grad_phi[scratch.j] =
					scratch.fe_values_subface.shape_grad(scratch.j,scratch.q);
			}
			
			// loop over all basis functions to get the shape values (K^-)
			for (scratch.j=0;
				scratch.j < scratch.neighbor_fe_values_face.get_fe().dofs_per_cell;
				++scratch.j) {
				scratch.neighbor_grad_phi[scratch.j] =
					scratch.neighbor_fe_values_face.shape_grad(scratch.j,scratch.q);
			}
			
			// fetch function value(s)
			scratch.value_diffusion_epsilon =
				function.diffusion_epsilon->value(
					scratch.fe_values_subface.quadrature_point(scratch.q), 0
				);
			
			// loop over all basis function combinitions to get the assembly
			Assert(
				(scratch.fe_values_subface.get_fe().dofs_per_cell==
				scratch.neighbor_fe_values_face.get_fe().dofs_per_cell),
				dealii::ExcMessage("different fe.p between neighboring cells is not allowed here")
			);
			
			scratch.val_face_jump_grad_u=0.;
			scratch.val_z_Rz_j=0.;
			for (scratch.j=0;
				scratch.j < scratch.fe_values_subface.get_fe().dofs_per_cell;
				++scratch.j) {
				scratch.val_face_jump_grad_u += (
					scratch.local_u0[scratch.j] * scratch.grad_phi[scratch.j]
					- scratch.neighbor_local_u0[scratch.j]
					* scratch.neighbor_grad_phi[scratch.j]
				) * scratch.normal_vector;
				
				scratch.val_z_Rz_j +=
					(scratch.local_z0[scratch.j] - scratch.local_Rz0[scratch.j])
					* scratch.phi[scratch.j];
			}
			
			// \int_{I_n} ... :
			copydata.value += (
				scratch.value_diffusion_epsilon
				* scratch.val_face_jump_grad_u
				// z_h - Rz_h:
				* scratch.val_z_Rz_j
				* tau_n
				* scratch.JxW
			);
		} // for q
		
		Assert(
			std::isnan(face_integrals[copydata.face]),
			dealii::ExcMessage("ErrorEstimator: you access the same (small) irregular face at least two times")
		);
		face_integrals[copydata.face] = copydata.value;
	}
	
	// compute integral value of big face as sum of the subface integral values
	copydata.face  = cell->face(face_no);
	copydata.value = 0;
	
	for (scratch.subface_no=0; scratch.subface_no < cell->face(face_no)->n_children();
		++scratch.subface_no) {
		copydata.value += face_integrals[cell->face(face_no)->child(scratch.subface_no)];
	}
	
	Assert(
		std::isnan(face_integrals[copydata.face]),
		dealii::ExcMessage("ErrorEstimator: you access the same (big) irregular face at least two times")
	);
	face_integrals[copydata.face] = copydata.value;
}


template<int dim>
void
ErrorEstimator<dim>::copy_local_error(
	const Assembly::CopyData::ErrorEstimates<dim> &) {
	// NOTE: This function is intended to be empty, since this ErrorEstimator
	// stores the assembly uniquely in special map<cell, double> and map<face, double>
	// data structures without summation.
	//
	// The final copy data from the internal data structures into a
	// dealii::Vector<double> eta_K must be done after the WorkStream completion.
}

}}}} // namespace

#include "ErrorEstimator_DWR__cGp_dG0__cGq_cG1.inst.in"
