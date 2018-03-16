/**
 * @file DWR_ErrorEstimator.tpl.cc
 *
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 *
 * @date 2018-03-13, ErrorEstimator class for heat, UK, MPB
 * @date 2017-11-08, ErrorEstimator class, UK, MPB
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
#include <DTM++/base/LogStream.hh>

#include <heat/ErrorEstimator/DWR_ErrorEstimator.tpl.hh>


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

namespace heat {
namespace dwr {

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
	value_epsilon(scratch.value_epsilon),
	grad_epsilon(scratch.grad_epsilon),
	JxW(scratch.JxW) {
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
	fe_values_subface(mapping, fe, quad, dealii::UpdateFlags::update_gradients),
	local_dof_indices(fe.dofs_per_cell),
	phi(fe.dofs_per_cell),
	grad_phi(fe.dofs_per_cell),
	local_u0(fe.dofs_per_cell),
	local_z0(fe.dofs_per_cell),
	local_Rz0(fe.dofs_per_cell),
	// data structures of neighboring face of cell (-)
	neighbor_fe_values_face(mapping, fe, quad, uflags),
	neighbor_local_dof_indices(fe.dofs_per_cell),
	neighbor_phi(fe.dofs_per_cell),
	neighbor_grad_phi(fe.dofs_per_cell),
	neighbor_local_u0(fe.dofs_per_cell),
	neighbor_local_z0(fe.dofs_per_cell),
	neighbor_local_Rz0(fe.dofs_per_cell) {
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
	neighbor_phi(scratch.neighbor_phi),
	neighbor_grad_phi(scratch.neighbor_grad_phi),
	neighbor_local_u0(scratch.neighbor_local_u0),
	neighbor_local_z0(scratch.neighbor_local_z0),
	neighbor_local_Rz0(scratch.neighbor_local_Rz0),
	// other
	value_epsilon(scratch.value_epsilon),
	value_u_D(scratch.value_u_D),
	JxW(scratch.JxW) {
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
	std::shared_ptr< dealii::Function<dim> > _epsilon,
	std::shared_ptr< dealii::Function<dim> > _f,
	std::shared_ptr< dealii::Function<dim> > _u_D,
	std::shared_ptr< dealii::Function<dim> > _u_0,
	std::shared_ptr< heat::Grid_DWR<dim,1> > _grid,
	std::shared_ptr< DTM::types::storage_data_vectors<1> > _u,
	std::shared_ptr< DTM::types::storage_data_vectors<2> > _z,
	std::shared_ptr< DTM::types::storage_data_vectors<1> > _eta
) {
	function.epsilon = _epsilon;
	function.f = _f;
	function.u_D = _u_D;
	function.u_0 = _u_0;
	
	grid = _grid;
	
	primal.storage.u = _u;
	dual.storage.z = _z;
	error_estimator.storage.eta = _eta;
	
	////////////////////////////////////////////////////////////////////////////
	// prepare TMS loop for eta_K on Omega x I_n
	//
	
	////////////////////////////////////////////////////////////////////////////
	// grid: init slab iterator to first space-time slab: Omega x I_1
	//
	
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	Assert(grid->slabs.size(), dealii::ExcNotInitialized());
	auto slab = grid->slabs.begin();
	
	auto u = primal.storage.u->begin();
	auto z = dual.storage.z->begin();
	auto eta = error_estimator.storage.eta->begin();
	
	////////////////////////////////////////////////////////////////////////////
	// do TMS loop
	//
	
	DTM::pout
		<< std::endl
		<< "*******************************************************************"
		<< "*************" << std::endl
		<< "error estimator: assemble eta ..." << std::endl
		<< std::endl;
	
	unsigned int n{1};
	while (slab != grid->slabs.end()) {
		// local time variables
		const double tm = slab->t_m;
		const double t0 = tm + slab->tau_n()/2.;
		
		// set tau_n in error_estimator
		tau_n = slab->tau_n();
		
		DTM::pout
			<< "error estimator: assemble on " << "I_" << n 
// 			<< " = (" << tm << ", " << tn << ") "
			<< std::endl;
		
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
			auto primal_uD_on_t0 = std::make_shared< dealii::Vector<double> > ();
			primal_uD_on_t0->reinit( slab->primal.dof->n_dofs() );
			
			function.u_D->set_time(t0);
			dealii::VectorTools::interpolate(
				*slab->primal.mapping,
				*slab->primal.dof,
				*function.u_D,
				*primal_uD_on_t0
			);
			
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
		(*eta->x[0]) = 0;
		{
			auto cell = slab->dual.dof->begin_active();
			auto endc = slab->dual.dof->end();
			
			for (unsigned int cell_no{0}; cell != endc; ++cell, ++cell_no) {
				(*eta->x[0])[cell_no] = cell_integrals[cell];
				
				for (unsigned int face_no{0};
					face_no < dealii::GeometryInfo<dim>::faces_per_cell; ++face_no) {
					Assert(
						(face_integrals.find(cell->face(face_no)) != face_integrals.end()),
						dealii::ExcMessage("Your face iterator does not exist in your map(face_it,double). \
						Please check if you have assembled this error integral.")
					);
					
					(*eta->x[0])[cell_no] -= 0.5 * face_integrals[cell->face(face_no)];
				}
			}
		}
		
		// check if error_indicators vector has valid entries only
		{
			for (unsigned int cell_no{0}; cell_no < eta->x[0]->size(); ++cell_no) {
				Assert(
					!std::isnan((*eta->x[0])[cell_no]),
					dealii::ExcMessage("Your error indicator has quiet_NaN entries. \
					Please check if you have assembled cell_integrals and face_integrals correctly.")
				);
				(void)cell_no;
			}
		}
		cell_integrals.clear();
		face_integrals.clear();
		
		
		////////////////////////////////////////////////////////////////////////
		// prepare next I_n slab problem:
		//
		
		++n;
		++slab;
		++u; ++z; ++eta;
	}
	
	DTM::pout
		<< "error estimator: assemble eta_K done" << std::endl
		<< "*******************************************************************"
		<< "*************" << std::endl
		<< std::endl;
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
		
		// _t = t0 = 0.5 <=> time dof of dG(0)-Q_G(1) on \hat I = (0,1)
		const double _t{ 1./2. };
		
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
	for (unsigned int face_no{0};
		face_no < dealii::GeometryInfo<dim>::faces_per_cell; ++face_no) {
		////////////////////////////////////////////////////////////////////////
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
	
	// fetch local dof data
	cell->get_dof_indices(scratch.local_dof_indices);
	
	for (unsigned int j{0}; j < scratch.fe_values.get_fe().dofs_per_cell; ++j) {
		scratch.local_u0[j] = (*dual_u_on_t0)[ scratch.local_dof_indices[j] ];
	}
	
	for (unsigned int j{0}; j < scratch.fe_values.get_fe().dofs_per_cell; ++j) {
		scratch.local_z0[j] = (*dual_z_on_t0)[ scratch.local_dof_indices[j] ];
	}
	
	for (unsigned int j{0}; j < scratch.fe_values.get_fe().dofs_per_cell; ++j) {
		scratch.local_Rz0[j] = (*dual_Rz_on_t0)[ scratch.local_dof_indices[j] ];
	}
	
	for (unsigned int j{0}; j < scratch.fe_values.get_fe().dofs_per_cell; ++j) {
		scratch.local_um[j] = (*dual_um_on_tm)[ scratch.local_dof_indices[j] ];
	}
	
	for (unsigned int j{0}; j < scratch.fe_values.get_fe().dofs_per_cell; ++j) {
		scratch.local_up[j] = (*dual_up_on_tm)[ scratch.local_dof_indices[j] ];
	}
	
	for (unsigned int j{0}; j < scratch.fe_values.get_fe().dofs_per_cell; ++j) {
		scratch.local_zm[j] = (*dual_z_on_tm)[ scratch.local_dof_indices[j] ];
	}
	
	for (unsigned int j{0}; j < scratch.fe_values.get_fe().dofs_per_cell; ++j) {
		scratch.local_Rzm[j] = (*dual_Rz_on_tm)[ scratch.local_dof_indices[j] ];
	}
	
	// initialize local matrix with zeros
	copydata.value=0.;
	
	// assemble cell terms
	for (unsigned int q{0}; q < scratch.fe_values.n_quadrature_points; ++q) {
		scratch.JxW = scratch.fe_values.JxW(q);
		
		// loop over all basis functions to get the shape values
		for (unsigned int k{0}; k < scratch.fe_values.get_fe().dofs_per_cell; ++k) {
			scratch.phi[k] =
				scratch.fe_values.shape_value_component(k,q,0);
		}
		
		for (unsigned int k{0}; k < scratch.fe_values.get_fe().dofs_per_cell; ++k) {
			scratch.grad_phi[k] =
				scratch.fe_values.shape_grad(k,q);
		}
		
		for (unsigned int k{0}; k < scratch.fe_values.get_fe().dofs_per_cell; ++k) {
			scratch.hessian_phi = scratch.fe_values.shape_hessian(k,q);
			
			scratch.laplace_phi[k] = 0.;
			for (unsigned int d{0}; d < dim; ++d) {
				scratch.laplace_phi[k] += scratch.hessian_phi[d][d];
			}
		}
		
		scratch.value_f = function.f->value(scratch.fe_values.quadrature_point(q), 0);
		
		scratch.value_epsilon =
			function.epsilon->value(scratch.fe_values.quadrature_point(q), 0);
		
		scratch.grad_epsilon =
			function.epsilon->gradient(scratch.fe_values.quadrature_point(q), 0);
		
		// loop over all basis function combinitions to get the assembly
		for (unsigned int j{0}; j < scratch.fe_values.get_fe().dofs_per_cell; ++j) {
			// \int_{I_n} ... :
			copydata.value += (
				// R(u_kh):
				(	scratch.value_f
					// - 0 <= here => - density(x_q,t_q) * \partial_t u * 1/tau_n
					+ scratch.local_u0[j] * (scratch.grad_phi[j] * scratch.grad_epsilon)
					+ scratch.value_epsilon * scratch.local_u0[j] * scratch.laplace_phi[j]
				)
				// z_h - Rz_h:
				* (scratch.local_z0[j] - scratch.local_Rz0[j]) * scratch.phi[j]
				* tau_n
				* scratch.JxW
			);
			
			// - [ u_kh(t_m) ] * ( z_h(t_m) - I(R(z_h(t_m))) )
			copydata.value += (
				- (scratch.local_up[j] - scratch.local_um[j]) * scratch.phi[j]
				* (scratch.local_zm[j] - scratch.local_Rzm[j]) * scratch.phi[j]
				* scratch.JxW
			);
		} // for j
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
	
// 	scratch.fe_values_face.reinit(cell, face_no);
// 	
// 	function.BoundaryValues->value_list(scratch.fe_values_face.get_quadrature_points(),
// 							   scratch.boundary_values);
// 	
// 	scratch.fe_values_face.get_function_gradients(
// 	*dual.z,
// 	scratch.dual_solution_gradients
// 	);
// 	
// 	scratch.fe_values_face.get_function_values(
// 		g_interpolated,
// 		scratch.g_h
// 	);
// 	
// 	for (unsigned int q=0;q<scratch.fe_values_face.n_quadrature_points; ++q) {
// 		scratch.inhom_dirichlet_difference[q] = (scratch.boundary_values[q] - scratch.g_h[q]);
// 	}
// 	
// 	copydata.face  = cell->face(face_no);
// 	copydata.value = 0;
// 	
// 	for (unsigned int q=0; q < scratch.fe_values_face.n_quadrature_points; ++q) { // (g-g_h, epsilon*grad(z_h)*n)_dOmega
// 		copydata.value +=
// 			2.*(													//multiplied with 2, because within the estimate() function
// 			(scratch.inhom_dirichlet_difference[q] *						//the whole faces contribution will be subtracted  by the factor
// 			(scratch.fe_values_face.normal_vector(q)*
// 			(function.epsilon->value(scratch.fe_values_face.quadrature_point(q), 0)*
// 			scratch.dual_solution_gradients[q])))*
// 			scratch.fe_values_face.JxW(q));
// 	}

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
	
	for (unsigned int j{0}; j < scratch.fe_values_face.get_fe().dofs_per_cell; ++j) {
		scratch.local_u0[j] =
			(*dual_u_on_t0)[ scratch.local_dof_indices[j] ];
	}
	
	for (unsigned int j{0}; j < scratch.fe_values_face.get_fe().dofs_per_cell; ++j) {
		scratch.local_z0[j] =
			(*dual_z_on_t0)[ scratch.local_dof_indices[j] ];
	}
	
	for (unsigned int j{0}; j < scratch.fe_values_face.get_fe().dofs_per_cell; ++j) {
		scratch.local_Rz0[j] =
			(*dual_Rz_on_t0)[ scratch.local_dof_indices[j] ];
	}
	
	// fetch local dof data ( K^- / F^- )
	cell->neighbor(face_no)->get_dof_indices(scratch.neighbor_local_dof_indices);
	
	for (unsigned int j{0};
		j < scratch.neighbor_fe_values_face.get_fe().dofs_per_cell; ++j) {
		scratch.neighbor_local_u0[j] =
			(*dual_u_on_t0)[ scratch.neighbor_local_dof_indices[j] ];
	}
	
	for (unsigned int j{0};
		j < scratch.neighbor_fe_values_face.get_fe().dofs_per_cell; ++j) {
		scratch.neighbor_local_z0[j] =
			(*dual_z_on_t0)[ scratch.neighbor_local_dof_indices[j] ];
	}
	
	for (unsigned int j{0};
		j < scratch.neighbor_fe_values_face.get_fe().dofs_per_cell; ++j) {
		scratch.neighbor_local_Rz0[j] =
			(*dual_Rz_on_t0)[ scratch.neighbor_local_dof_indices[j] ];
	}
	
	
	
	
	
	
// 	// TODO:
// 	scratch.fe_values_face.get_function_gradients(
// 		*dual.u,
// 		scratch.cell_grads
// 	);
// 	
// 	
// 	// TODO:
// 	scratch.neighbor_fe_values_face.get_function_gradients(
// 		*dual.u,
// 		scratch.neighbor_grads
// 	);
// 	
// 	
// 	// TODO:
// 	for (unsigned int q=0; q < scratch.fe_values_face.n_quadrature_points; ++q) {
// 		scratch.jump_residuals[q] = (
// 			(function.epsilon->value(scratch.fe_values_face.quadrature_point(q), 0)*
// 			(scratch.cell_grads[q] - scratch.neighbor_grads[q])) *
// 			scratch.fe_values_face.normal_vector(q)
// 		);
// 	}
// 	
// 	// TODO:
// 	scratch.fe_values_face.get_function_values(
// 		dual_weights,
// 		scratch.dual_weights
// 	);
// 	
// 	
// 	// TODO:
// 	copydata.face  = cell->face(face_no);
// 	copydata.value = 0;
// 	
// 	for (unsigned int q=0; q < scratch.fe_values_face.n_quadrature_points; ++q) {
// 		copydata.value += (
// 			scratch.jump_residuals[q] * scratch.dual_weights[q] *
// 			scratch.fe_values_face.JxW(q)
// 		);
// 	}
// 	
	
	
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
	
// 	Assert(
// 		(cell->neighbor(face_no).state() == dealii::IteratorState::valid),
// 		dealii::ExcInternalError()
// 	);
// 	
// 	Assert(
// 		(cell->neighbor(face_no)->has_children()),
// 		dealii::ExcInternalError()
// 	);
// 	
// 	for (unsigned int subface_no=0; subface_no < cell->face(face_no)->n_children();
// 		++subface_no) {
// 		Assert(
// 			(cell->neighbor_child_on_subface(face_no,subface_no)->face(
// 				cell->neighbor_of_neighbor(face_no) ) ==
// 			cell->face(face_no)->child(subface_no)),
// 			dealii::ExcInternalError()
// 		);
// 		
// 		scratch.fe_values_subface.reinit(cell, face_no, subface_no);
// 		
// 		scratch.neighbor_fe_values_face.reinit(
// 			cell->neighbor_child_on_subface(face_no, subface_no),
// 			cell->neighbor_of_neighbor(face_no)
// 		);
// 		
// 		scratch.fe_values_subface.get_function_gradients(
// 			*dual.u,
// 			scratch.cell_grads
// 		);
// 		
// 		scratch.neighbor_fe_values_face.get_function_gradients(
// 			*dual.u,
// 			scratch.neighbor_grads
// 		);
// 		
// 		for (unsigned int q=0; q < scratch.fe_values_face.n_quadrature_points; ++q) {
// 			scratch.jump_residuals[q] = (
// 				(function.epsilon->value(scratch.fe_values_face.quadrature_point(q), 0)*
// 				(scratch.neighbor_grads[q] - scratch.cell_grads[q])) *
// 				scratch.neighbor_fe_values_face.normal_vector(q)
// 			);
// 		}
// 		
// 		scratch.neighbor_fe_values_face.get_function_values(
// 			dual_weights,
// 			scratch.dual_weights
// 		);
// 		
// 		copydata.face  = cell->face(face_no)->child(subface_no);
// 		copydata.value = 0;
// 		
// 		for (unsigned int q=0; q < scratch.fe_values_face.n_quadrature_points; ++q) {
// 			copydata.value += (
// 				scratch.jump_residuals[q] * scratch.dual_weights[q] *
// 				scratch.neighbor_fe_values_face.JxW(q)
// 			);
// 		}
// 		
// 		face_integrals[copydata.face] = copydata.value;
// 	}
// 	
// 	// compute integral value of big face as sum of the subface integral values
// 	copydata.face  = cell->face(face_no);
// 	copydata.value = 0;
// 	
// 	for (unsigned int subface_no=0; subface_no < cell->face(face_no)->n_children();
// 		++subface_no) {
// 		copydata.value += face_integrals[cell->face(face_no)->child(subface_no)];
// 	}
	
	Assert(
		std::isnan(face_integrals[cell->face(face_no)]),
		dealii::ExcMessage("ErrorEstimator: you access the same irregular face at least two times")
	);
	face_integrals[cell->face(face_no)] = copydata.value;
}


template<int dim>
void
ErrorEstimator<dim>::copy_local_error(
	const Assembly::CopyData::ErrorEstimates<dim> &) {
}


}} // namespace

#include "DWR_ErrorEstimator.inst.in"
