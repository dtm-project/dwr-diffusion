/**
 * @file Heat_DWR__cGp_dG0__cGq_cG1.tpl.cc
 * 
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * 
 * @date 2018-03-08, primal problem, UK
 * @date 2018-03-06, new implementation, UK
 * @date 2017-08-01, Heat/DWR, MPB, UK
 *
 * @brief Heat/DWR Problem
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

#include <heat/Heat_DWR__cGp_dG0__cGq_cG1.tpl.hh>

#include <heat/assembler/L2_MassAssembly.tpl.hh>

#include <heat/assembler/L2_LaplaceAssembly.tpl.hh>

#include <heat/assembler/L2_ForceConstrainedAssembly.tpl.hh>

template <int dim>
using ForceAssembler = heat::Assemble::L2::ForceConstrained::Assembler<dim>;

#include <heat/types/boundary_id.hh>

#include <heat/Force/Forces.hh>

// DEAL.II includes

// #include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

// // C++ includes
// #include <fstream>
// #include <vector>
// #include <algorithm>
// #include <list>
// #include <iterator>
// 
// #include <iomanip>

namespace heat {

template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
set_input_parameters(
	std::shared_ptr< dealii::ParameterHandler > parameter_handler) {
	Assert(parameter_handler.use_count(), dealii::ExcNotInitialized());
	
	parameter_set = std::make_shared< heat::dwr::ParameterSet > (
		parameter_handler
	);
}

template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
set_grid(std::shared_ptr< Grid_DWR<dim,1> > _grid) {
	Assert(_grid.use_count(), dealii::ExcNotInitialized());
	grid = _grid;
}

template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
run() {
	// check
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	
	init_functions();
	
	init_grid();
	
	// DWR loop:
	grid->set_boundary_indicators();
	grid->distribute();
	
	reinit_storage();
	
	// primal problem:
	primal_do_forward_TMS();
}


////////////////////////////////////////////////////////////////////////////////
// internal functions
//

template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
init_functions() {
	// TODO: read those from parameter input file
	function.u_D = std::make_shared< dealii::ZeroFunction<dim> > (1);
	function.u_0 = std::make_shared< dealii::ZeroFunction<dim> > (1);
	
// 	function.f = std::make_shared< dealii::ConstantFunction<dim> > (M_PI);
	function.f = std::make_shared< heat::force::Test0<dim> > ();
	
	function.epsilon = std::make_shared< dealii::ConstantFunction<dim> > (1.0);
	
	function.density = std::make_shared< dealii::ConstantFunction<dim> > (1.0);
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
init_grid() {
	////////////////////////////////////////////////////////////////////////////
	// initialize slabs of grid
	//
	
	Assert((parameter_set->fe.p), dealii::ExcInvalidState());
	Assert((parameter_set->fe.p < parameter_set->fe.q), dealii::ExcInvalidState());
	
	Assert((parameter_set->t0 >= 0), dealii::ExcInvalidState());
	Assert((parameter_set->t0 < parameter_set->T), dealii::ExcInvalidState());
	Assert((parameter_set->tau_n > 0), dealii::ExcInvalidState());
	
	grid->initialize_slabs(
		parameter_set->fe.p,
		parameter_set->fe.q,
		parameter_set->t0,
		parameter_set->T,
		parameter_set->tau_n
	);
	
	grid->generate();
	
	grid->refine_global(
		parameter_set->global_refinement
	);
	
	DTM::pout
		<< "grid: number of slabs = " << grid->slabs.size()
		<< std::endl;
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
reinit_storage() {
	////////////////////////////////////////////////////////////////////////////
	// init storage containers for vector data
	//
	
	////////////////////////////////////////////////////////////////////////////
	// get number of time steps
	//
	const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};
	
	////////////////////////////////////////////////////////////////////////////
	// primal space: time dG(0) method ( here: dG(0)-Q_{Gauss(1)} )
	//
	
	////////////////////////////////////////////////////////////////////////////
	// primal dof vectors um (on primal solution space)
	//
	primal.storage.um = std::make_shared< DTM::types::storage_data_vectors<1> > ();
	primal.storage.um->resize(N);
	
	{
		auto slab = grid->slabs.begin();
		for (auto &element : *primal.storage.um) {
			// create shared_ptr to Vector<double>
			element.x[0] = std::make_shared< dealii::Vector<double> > ();
			
			// init. Vector<double> with n_dofs components
			Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
			Assert(slab->primal.dof.use_count(), dealii::ExcNotInitialized());
			Assert(
				slab->primal.dof->n_dofs(),
				dealii::ExcMessage("Error: slab->primal.dof->n_dofs() == 0")
			);
			
			element.x[0]->reinit(
				slab->primal.dof->n_dofs()
			);
			
			++slab;
		}
	}
	
	////////////////////////////////////////////////////////////////////////////
	// primal solution dof vectors u (on primal solution space)
	//
	primal.storage.u = std::make_shared< DTM::types::storage_data_vectors<1> > ();
	primal.storage.u->resize(N);
	
	{
		auto slab = grid->slabs.begin();
		for (auto &element : *primal.storage.u) {
			for (unsigned int j{0}; j < element.x.size(); ++j) {
				// create shared_ptr to Vector<double>
				element.x[j] = std::make_shared< dealii::Vector<double> > ();
				
				// init. Vector<double> with n_dofs components
				Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
				Assert(slab->primal.dof.use_count(), dealii::ExcNotInitialized());
				Assert(
					slab->primal.dof->n_dofs(),
					dealii::ExcMessage("Error: slab->primal.dof->n_dofs() == 0")
				);
				
				element.x[j]->reinit(
					slab->primal.dof->n_dofs()
				);
			}
			++slab;
		}
	}
	
	////////////////////////////////////////////////////////////////////////////
	// primal dof vectors un (on primal solution space)
	//
	primal.storage.un = std::make_shared< DTM::types::storage_data_vectors<1> > ();
	primal.storage.un->resize(N);
	
	{
		auto slab = grid->slabs.begin();
		for (auto &element : *primal.storage.un) {
			// create shared_ptr to Vector<double>
			element.x[0] = std::make_shared< dealii::Vector<double> > ();
			
			// init. Vector<double> with n_dofs components
			Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
			Assert(slab->primal.dof.use_count(), dealii::ExcNotInitialized());
			Assert(
				slab->primal.dof->n_dofs(),
				dealii::ExcMessage("Error: slab->primal.dof->n_dofs() == 0")
			);
			
			element.x[0]->reinit(
				slab->primal.dof->n_dofs()
			);
			
			++slab;
		}
	}
	
// 	////////////////////////////////////////////////////////////////////////////
// 	// dual space: time cG(1) method ( here: cG(1)-Q_{Gauss-Lobatto(2)} )
// 	//
// 	
// 	////////////////////////////////////////////////////////////////////////////
// 	// primal solution dof vectors u (on dual solution space)
// 	//
// 	dual.storage.u = std::make_shared< DTM::types::storage_data_vectors<2> > ();
// 	dual.storage.u->resize(N);
// 	
// 	{
// 		auto slab = grid->slabs.begin();
// 		for (auto &element : *dual.storage.u) {
// 			for (unsigned int j{0}; j < element.x.size(); ++j) {
// 				// create shared_ptr to Vector<double>
// 				element.x[j] = std::make_shared< dealii::Vector<double> > ();
// 				
// 				// init. Vector<double> with n_dofs components
// 				Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
// 				Assert(slab->dual.dof.use_count(), dealii::ExcNotInitialized());
// 				Assert(
// 					slab->dual.dof->n_dofs(),
// 					dealii::ExcMessage("Error: slab->dual.dof->n_dofs() == 0")
// 				);
// 				
// 				element.x[j]->reinit(
// 					slab->dual.dof->n_dofs()
// 				);
// 			}
// 			++slab;
// 		}
// 	}
	
// 	///////////////////////////////////////////////////////
// 	// dual solution dof vectors z (on dual solution space)
// 	dual.storage.z = std::make_shared< DTM::types::storage_data_vectors<1> > ();
// 	dual.storage.z->resize(N+1);
// 	for (auto &element : *dual.storage.z) {
// 	for (unsigned int j{0}; j < element.x.size(); ++j) {
// 		element.x[j] = std::make_shared< dealii::Vector<double> > ();
// 	}}
	
// 	/////////////////////////////////////////////////
// 	// auxiliary vectors eta (on dual solution space)
// 	dual.storage.eta = std::make_shared< DTM::types::storage_data_vectors<1> > ();
// 	dual.storage.eta->resize(N);
// 	for (auto &element : *dual.storage.eta) {
// 	for (unsigned int j{0}; j < element.x.size(); ++j) {
// 		element.x[j] = std::make_shared< dealii::Vector<double> > ();
// 	}}
}


////////////////////////////////////////////////////////////////////////////////
// primal problem
//

template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
primal_assemble_system(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab
) {
	// ASSEMBLY MASS MATRIX ////////////////////////////////////////////////////
	primal.M = std::make_shared< dealii::SparseMatrix<double> > ();
	primal.M->reinit(*slab->primal.sp);
	
	*primal.M = 0;
	{
		heat::Assemble::L2::Mass::
		Assembler<dim> assemble_mass(
			primal.M,
			slab->primal.dof,
			slab->primal.fe,
			slab->primal.mapping,
			slab->primal.constraints
		);
		
		Assert(function.density.use_count(), dealii::ExcNotInitialized());
		assemble_mass.set_density(function.density);
		
		DTM::pout << "dwr-heat: assemble mass matrix...";
		assemble_mass.assemble();
		DTM::pout << " (done)" << std::endl;
	}
	
	// ASSEMBLY STIFFNESS MATRIX ///////////////////////////////////////////////
	primal.A = std::make_shared< dealii::SparseMatrix<double> > ();
	primal.A->reinit(*slab->primal.sp);
	
	*primal.A = 0;
	{
		heat::Assemble::L2::Laplace::
		Assembler<dim> assemble_stiffness_cell_terms (
			primal.A,
			slab->primal.dof,
			slab->primal.fe,
			slab->primal.mapping,
			slab->primal.constraints
		);
		
		Assert(function.epsilon.use_count(), dealii::ExcNotInitialized());
		assemble_stiffness_cell_terms.set_epsilon_function(function.epsilon);
		
		DTM::pout << "dwr-heat: assemble cell stiffness matrix...";
		assemble_stiffness_cell_terms.assemble();
		DTM::pout << " (done)" << std::endl;
	}
	
	// construct system matrix K = M + tau A
	DTM::pout << "dwr-heat: construct system matrix K = M + tau A...";
	
	primal.K = std::make_shared< dealii::SparseMatrix<double> > ();
	primal.K->reinit(*slab->primal.sp);
	
	*primal.K = 0;
	primal.K->add(slab->tau_n(), *primal.A);
	primal.K->add(1.0, *primal.M);
	
	DTM::pout << " (done)" << std::endl;
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
primal_assemble_rhs(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &um,
	const double t0
) {
	primal.f0 = std::make_shared< dealii::Vector<double> > ();
	primal.f0->reinit( slab->primal.dof->n_dofs() );
	
	auto assemble_f0 = std::make_shared< ForceAssembler<dim> > (
		primal.f0,
		slab->primal.dof,
		slab->primal.fe,
		slab->primal.mapping,
		slab->primal.constraints
	);
	
	Assert(function.f.use_count(), dealii::ExcNotInitialized());
	assemble_f0->set_function(function.f);
	
	DTM::pout << "dwr-heat: assemble force f0...";
	*primal.f0 = 0;
	assemble_f0->assemble(
		t0,
		0,   // n_q_points: 0 -> p+1 in auto mode
		true // auto mode
	);
	DTM::pout << " (done)" << std::endl;
	
	// construct vector b = M um + tau_n f0
	DTM::pout << "dwr-heat: construct linear system rhs vector...";
	
	primal.b = std::make_shared< dealii::Vector<double> > ();
	primal.b->reinit( slab->primal.dof->n_dofs() );
	
	Assert(primal.M.use_count(), dealii::ExcNotInitialized());
	primal.M->vmult(*primal.b, *um->x[0]);
	
	primal.b->add(slab->tau_n(), *primal.f0);
	
	DTM::pout << " (done)" << std::endl;
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
primal_solve(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab
) {
	////////////////////////////////////////////////////////////////////////////
	// apply Dirichlet boundary values
	//
	
	DTM::pout << "dwr-heat: dealii::MatrixTools::apply_boundary_values...";
	std::map<dealii::types::global_dof_index, double> boundary_values;
	
	Assert(function.u_D.use_count(), dealii::ExcNotInitialized());
	dealii::VectorTools::interpolate_boundary_values(
		*slab->primal.dof,
		static_cast< dealii::types::boundary_id > (
			heat::types::boundary_id::Dirichlet
		),
		*function.u_D,
		boundary_values
	);
	
	dealii::MatrixTools::apply_boundary_values(
		boundary_values,
		*primal.K,
		*primal.iterator.u->x[0],
		*primal.b
	);
	
	DTM::pout << " (done)" << std::endl;
	
	////////////////////////////////////////////////////////////////////////////
	// solve linear system directly
	//
	
	DTM::pout << "dwr-heat: setup direct lss and solve...";
	
	dealii::SparseDirectUMFPACK iA;
	iA.initialize(*primal.K);
	iA.vmult(*primal.iterator.u->x[0], *primal.b);
	
	DTM::pout << " (done)" << std::endl;
	
	////////////////////////////////////////////////////////////////////////////
	// distribute hanging node constraints on solution
	//
	
	DTM::pout << "dwr-heat: primal.constraints->distribute...";
	slab->primal.constraints->distribute(
		*primal.iterator.u->x[0]
	);
	DTM::pout << " (done)" << std::endl;
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
primal_do_forward_TMS() {
	////////////////////////////////////////////////////////////////////////////
	// prepare TMS loop
	//
	
	////////////////////////////////////////////////////////////////////////////
	// grid: init slab iterator to first space-time slab: Omega x I_1
	//
	
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	Assert(grid->slabs.size(), dealii::ExcNotInitialized());
	auto slab = grid->slabs.begin();
	
	////////////////////////////////////////////////////////////////////////////
	// storage: init iterators to storage_data_vectors
	//          corresponding to first space-time slab: Omega x I_1
	//
	
	Assert(primal.storage.um.use_count(), dealii::ExcNotInitialized());
	Assert(primal.storage.um->size(), dealii::ExcNotInitialized());
	primal.iterator.um = primal.storage.um->begin();
	
	Assert(primal.storage.u.use_count(), dealii::ExcNotInitialized());
	Assert(primal.storage.u->size(), dealii::ExcNotInitialized());
	primal.iterator.u = primal.storage.u->begin();
	
	Assert(primal.storage.un.use_count(), dealii::ExcNotInitialized());
	Assert(primal.storage.un->size(), dealii::ExcNotInitialized());
	primal.iterator.un = primal.storage.un->begin();
	
	// shortcut references
	auto &um = primal.iterator.um;
	auto &u  = primal.iterator.u;
	auto &un = primal.iterator.un;
	
	////////////////////////////////////////////////////////////////////////////
	// interpolate (or project) initial value(s)
	//
	
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	Assert(function.u_0.use_count(), dealii::ExcNotInitialized());
	function.u_0->set_time(slab->t_m);
	
	// primal grid
	Assert(slab->primal.mapping.use_count(), dealii::ExcNotInitialized());
	Assert(slab->primal.dof.use_count(), dealii::ExcNotInitialized());
	Assert(um->x[0]->size(), dealii::ExcNotInitialized());
	
	dealii::VectorTools::interpolate(
		*slab->primal.mapping,
		*slab->primal.dof,
		*function.u_0,
		*um->x[0]
	);
	
	// output "solution" at initial time t0
	primal_init_data_output(slab);
	primal_do_data_output(slab,slab->t_m);
	
	////////////////////////////////////////////////////////////////////////////
	// do TMS loop
	//
	
	DTM::pout
		<< std::endl
		<< "*******************************************************************"
		<< "*************" << std::endl
		<< "primal: solving forward TMS problem..." << std::endl
		<< std::endl;
	
	unsigned int n{1};
	while (slab != grid->slabs.end()) {
		// local time variables: \f$ t0 \in I_n = (t_m, t_n) \f$
		const double tm = slab->t_m;
		const double t0 = tm + slab->tau_n()/2.;
		const double tn = slab->t_n;
		
		DTM::pout
			<< "primal: solving problem on "
			<< "I_" << n << " = (" << tm << ", " << tn << ") "
			<< std::endl;
		
		if (slab != grid->slabs.begin()) {
			// for n > 1 interpolate between two (different) spatial meshes
			// the solution u(t_n)|_{I_{n-1}}  to  u(t_m)|_{I_n}
			dealii::VectorTools::interpolate_to_different_mesh(
				// solution on I_{n-1}:
				*std::prev(slab)->primal.dof,
				*std::prev(un)->x[0],
				// solution on I_n:
				*slab->primal.dof,
				*slab->primal.constraints,
				*um->x[0]
			);
		}
		
		// assemble
		primal_assemble_system(slab);
		primal_assemble_rhs(slab,um,t0);
		
		// apply boundary values and solve for u0
		primal_solve(slab);
		
		////////////////////////////////////////////////////////////////////////
		// do postprocessing on the solution
		//
		
		// evaluate solution u(t_n)
		double zeta0 = 1.0; // zeta0( t_n ) = 1.0 for dG(0)
		*un->x[0] = 0;
		un->x[0]->add(zeta0, *u->x[0]);
		
		// output solution at t_n
		primal_do_data_output(slab,tn);
		
		////////////////////////////////////////////////////////////////////////
		// prepare next I_n slab problem:
		//
		
		++n;
		++slab;
		
		++um;
		++u;
		++un;
		
		DTM::pout << std::endl;
	}
	
	DTM::pout
		<< "primal: forward TMS problem done" << std::endl
		<< "*******************************************************************"
		<< "*************" << std::endl
		<< std::endl;
}



template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
primal_init_data_output(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab
) {
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
	// TODO:
	DTM::pout
		<< "primal solution data output: patches = "
		<< parameter_set->fe.p // auto mode = cG in space: take p patches per K
// 		<< parameter_set->data_output.patches
		<< std::endl;
	
	std::vector<std::string> data_field_names;
	data_field_names.push_back("u");
	
	std::vector< dealii::DataComponentInterpretation::DataComponentInterpretation > dci_field;
	dci_field.push_back(dealii::DataComponentInterpretation::component_is_scalar);
	
	Assert((slab != grid->slabs.end()), dealii::ExcInvalidState());
	Assert(slab->primal.dof.use_count(), dealii::ExcNotInitialized());
	primal.data_output.set_DoF_data(
		slab->primal.dof
	);
	
	primal.data_output.set_data_field_names(data_field_names);
	primal.data_output.set_data_component_interpretation_field(dci_field);
	
	// TODO:
	primal.data_output.set_data_output_patches(
		parameter_set->fe.p // auto mode = cG in space: take p patches per K
// 		parameter_set->data_output.patches
	);
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
primal_do_data_output(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const double &t_n) {
	
	primal.data_output.set_DoF_data(
		slab->primal.dof
	);
	
	// TODO: construct solution at t_n
	// NOTE: dG(0) is constant in time, thus we can simply output u->x[0]
	
	//const double zeta0 = 1.0;
	//u_trigger.add(zeta0, *u->x[0])
	
	// TODO
	primal.data_output.write_data(
		"primal",
		primal.iterator.um->x[0],
		t_n
	);
}




////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// 	for (unsigned int n{0}; n <= ((data.T-data.t0)/data.tau_n); ++n) {
// 		if (n == 0) {
// // 			//NOTE: done:
// // 			primal_compute_initial_condition();
// // 			
// // 			//NOTE (UK): do this in dual problem:
// // 			// Store initial condition u_0, interpolated in dual FE room (dual.u) 
// // 			// in the first element of list dual.storage.u
// // 			dual.storage.u->front().x->reinit(grid->slabs.front().dual.dof->n_dofs());
// // 			*(dual.storage.u->front().x) = *(dual.u);
// // 			
// 		}
// 		else if (n == 1) {
// // 			// Compute u_1 on same grid as u_0.
// // 			++In_uth; //increase iterator of list dual.storage.u
// // 			++In_uthprimal; //increase iterator of list primal.storage.u
// // 			// Set iterators
// // 			primal.iterator.slab = Inth_primal;
// // 			primal.iterator.slab_previous = Inth_primal_prev; //for this timestep primal.iterator.slab=primal.iterator.slab_previous holds
// 			
// // 			//NOTE: do this in dual problem!
// // 			// interpolate current solution to dual fe-room
// // 			interpolate_primal_to_dual();
// // 			// Save current interpolated solution in list dual.storage.u
// // 			In_uth->x->reinit(primal.iterator.slab->dual.dof->n_dofs());
// // 			*(In_uth->x) = *(dual.u);
// 		}
// 		else {
// 			// Compute u_2,...,u_N
// // 			++In_uth; //increase iterator of list dual.storage.u
// // 
// // 			// NOTE: only in dual problem
// // 			
// // 			// interpolate current solution to dual fe-room
// // 			interpolate_primal_to_dual();
// // 
// // 			// Save current interpolated solution (dual.u) in list dual.storage.u
// // 			In_uth->x->reinit(primal.iterator.slab->dual.dof->n_dofs());
// // 			*(In_uth->x) = *(dual.u);
// 		}
// 
// 	} // end primal-loop n
// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// old


// template<int dim>
// void
// Heat_DWR__cGp_dG0__cGq_cG1<dim>::
// init(const unsigned int global_refinement) {

// 	
// 	////////////////////////////////////////////////////////////////////////////
// 	// init error estimator
// 	//
// 	error_estimator.DWR = std::make_shared<Heat::DWR::ErrorEstimator<dim> > ();
// 	
// 	error_estimator.DWR->set_objects(
// 		grid,
// 		function.epsilon,
// 		function.BoundaryValues,
// 		function.BoundaryValues_dual, // TODO
// 		function.f,
// 		function.f // TODO
// 	);
// }

// template<int dim>
// void
// Heat_DWR__cGp_dG0__cGq_cG1<dim>::
// dual_init_data_output() {
// 	////////////////////////////////////////////////////////////////////////////
// 	// INIT DATA OUTPUT
// 	//
// 	Assert(dual.iterator.slab->dual.dof.use_count(), dealii::ExcNotInitialized());
// 	
// 	DTM::pout << "Heat DWR: dual solution   data output: patches = " << dual.data_output_patches << std::endl;
// 	
// 	std::vector<std::string> data_field_names;
// // 	data_field_names.push_back("u");
// 	data_field_names.push_back("z");
// 	
// 	std::vector< dealii::DataComponentInterpretation::DataComponentInterpretation > dci_field;
// // 	dci_field.push_back(dealii::DataComponentInterpretation::component_is_scalar);
// 	dci_field.push_back(dealii::DataComponentInterpretation::component_is_scalar);
// 	
// // 	dual.data_output.set_DoF_data(
// // 		dual.iterator.slab->dual.dof
// // 	);
// 	dual.data_output.set_DoF_data(
// 		primal.iterator.slab->dual.dof
// 	);
// 	
// 	dual.data_output.set_data_field_names(data_field_names);
// 	dual.data_output.set_data_component_interpretation_field(dci_field);
// 	dual.data_output.set_data_output_patches(dual.data_output_patches);
// }


// template<int dim>
// void
// Heat_DWR__cGp_dG0__cGq_cG1<dim>::
// dual_reinit() {
// 	// now create vectors and matricies for dual problem
// 	Assert(dual.iterator.slab->dual.dof.use_count(), dealii::ExcNotInitialized());
// 	Assert(dual.iterator.slab->dual.sp.use_count(), dealii::ExcNotInitialized());
// 	
// 	dual.M.reinit(*(dual.iterator.slab->dual.sp));
// 	dual.A.reinit(*(dual.iterator.slab->dual.sp));
// 	dual.system_matrix.reinit(*(dual.iterator.slab->dual.sp));
// 	
// 	dual.z = std::make_shared< dealii::Vector<double> > ();
// 	dual.z->reinit(dual.iterator.slab->dual.dof->n_dofs());
// 	dual.z_old_interpolated = std::make_shared< dealii::Vector<double> > ();
// 	dual.z_old_interpolated->reinit(dual.iterator.slab->dual.dof->n_dofs());
// 	
// 	dual.Je.reinit(dual.iterator.slab->dual.dof->n_dofs());
// 	dual.system_rhs.reinit(dual.iterator.slab->dual.dof->n_dofs());
// 	
// 	//TEST // TODO
// 	dual.Je_old_interpolated.reinit(dual.iterator.slab->dual.dof->n_dofs());
// }


// template<int dim>
// void
// Heat_DWR__cGp_dG0__cGq_cG1<dim>::
// primal_compute_initial_condition() {
// 
// 	dual.u = std::make_shared< dealii::Vector<double> > ();
// 	dual.u->reinit(grid->slabs.front().dual.dof->n_dofs());
// 	
// 	dealii::FETools::interpolate(
// 		*(grid->slabs.front().primal.dof),
// 		*(primal.slab.u_old),
// 		*(grid->slabs.front().dual.dof),
// 		*(grid->slabs.front().dual.constraints),
// 		*(dual.u)
// 	);
// 	grid->slabs.front().dual.constraints->distribute(*(dual.u));
// }


// template<int dim>
// void
// Heat_DWR__cGp_dG0__cGq_cG1<dim>::
// interpolate_primal_to_dual() {
// 	dealii::FETools::interpolate(
// 		*(primal.iterator.slab->primal.dof),
// 		*(primal.slab.u),
// 		*(primal.iterator.slab->dual.dof),
// 		*(primal.iterator.slab->dual.constraints),
// 		*(dual.u)
// 	);
// }


////////////////////////////////////////////////////////////////////////////////
// dual problem

// template<int dim>
// void
// Heat_DWR__cGp_dG0__cGq_cG1<dim>::
// dual_compute_initial_condition() {
// 	
// 	switch (dual.Je_type) {
// 		case Heat::types::error_functional::forbidden:
// 			AssertThrow(false, dealii::ExcMessage("You need to initialise dual.Je_type."));
// 			break;
// 			
// 		case Heat::types::error_functional::L2_final:
// 			dual_compute_initial_condition_L2final();
// 			// Initialize rhs_vector dual.Je_old 
// 			dual.Je_old.reinit(dual.iterator.slab->dual.dof->n_dofs());
// 			break;
// 			
// 		case Heat::types::error_functional::L2_global:
// 			// Compute ones the global space time L2 Error:
// 			compute_global_STL2_error();
// 			//
// 			dual_compute_initial_condition_L2global();
// 			// Initialize rhs_vectors dual.Je_old and dual.Je
// 			dual.Je_old.reinit(dual.iterator.slab->dual.dof->n_dofs());
// 			dual.Je.reinit(dual.iterator.slab->dual.dof->n_dofs());
// 			// Compute dual.Je at timepoint t_N = 0.5 (within dual_assemble_je())
// 			// and store it in dual.Je_old.
// 			dual_assemble_Je_L2global();
// 			dual.Je_old = dual.Je;
// 			break;
// 			
// 		case Heat::types::error_functional::mean_final:
// 			dual_compute_initial_condition_mean_final();
// 			// Initialize rhs_vector dual.Je_old 
// 			dual.Je_old.reinit(dual.iterator.slab->dual.dof->n_dofs());
// 			break;
// 			
// 		case Heat::types::error_functional::mean_global:
// 			dual_compute_initial_condition_mean_global();
// 			// Initialize rhs_vectors dual.Je_old and dual.Je
// 			dual.Je_old.reinit(dual.iterator.slab->dual.dof->n_dofs());
// 			dual.Je.reinit(dual.iterator.slab->dual.dof->n_dofs());
// 			// Compute dual.Je at timepoint t_N = 0.5 (within dual_assemble_je())
// 			// and store it in dual.Je_old.
// 			dual_assemble_Je_mean_global();
// 			dual.Je_old = dual.Je;
// 			break;
// 			
// 		case Heat::types::error_functional::point:
// 			dual_compute_initial_condition_point_final();
// 			// Initialize rhs_vectors dual.Je_old
// 			dual.Je_old.reinit(dual.iterator.slab->dual.dof->n_dofs());
// 			break;
// 			
// 		default:
// 			AssertThrow(false, dealii::ExcMessage("Your dual.Je_type is unknown, please check your inputs."));
// 	} // end switch(dual.Je_type)
// }


// template<int dim>
// void
// Heat_DWR__cGp_dG0__cGq_cG1<dim>::
// dual_compute_initial_condition_L2global() {
// 	Assert(grid->slabs.back().dual.dof.use_count(), dealii::ExcNotInitialized());
// ////////////////////////////////////////////////////////////////////////////////
// 	// Initial condition for global L2Error-functional
// ////////////////////////////////////////////////////////////////////////////////
// 	
// 	dual.z_old = std::make_shared< dealii::Vector<double> > ();
// 	dual.z_old->reinit(grid->slabs.back().dual.dof->n_dofs());
// }

// template<int dim>
// void
// Heat_DWR__cGp_dG0__cGq_cG1<dim>::
// dual_assemble_rhs_at_t_Nminus1() {
// 	switch (dual.Je_type) {
// 		case Heat::types::error_functional::forbidden:
// 			AssertThrow(false, dealii::ExcMessage("You need to initialise dual.Je_type."));
// 			break;
// 		
// 		case Heat::types::error_functional::L2_final:
// 			dual_assemble_Je_L2final();
// 			// Compute special RHS on last Interval I_N 
// 			dual.system_rhs = 0;
// 			(dual.system_rhs).add(1.,dual.Je);
// 			// End of computation of RHS
// 			break;
// 			
// 		case Heat::types::error_functional::L2_global:
// 			dual_assemble_Je_L2global();
// 			dual_interpolate_to_next_grid();
// 			dual_assemble_rhs_L2global();
// 			break;
// 			
// 		case Heat::types::error_functional::mean_final:
// 			dual_assemble_Je_mean_final();
// 			// Compute special RHS on last Interval I_N 
// 			dual.system_rhs = 0;
// 			(dual.system_rhs).add(1.,dual.Je);
// 			// End of computation of RHS
// 			break;
// 			
// 		case Heat::types::error_functional::mean_global:
// 			dual_assemble_Je_mean_global();
// 			dual_interpolate_to_next_grid();
// 			dual_assemble_rhs_mean_global();
// 			break;
// 			
// 		case Heat::types::error_functional::point:
// 			dual_assemble_Je_point_final();
// 			// Compute special RHS on last Interval I_N 
// 			dual.system_rhs = 0;
// 			(dual.system_rhs).add(1.,dual.Je);
// 			// End of computation of RHS
// 			break;
// 			
// 		default:
// 			AssertThrow(false, dealii::ExcMessage("Your dual.Je_type is unknown, please check your inputs."));
// 	} // end switch(dual.Je-type)
// }


// template<int dim>
// void
// Heat_DWR__cGp_dG0__cGq_cG1<dim>::
// dual_assemble_rhs() {
// 	switch (dual.Je_type) {
// 		case Heat::types::error_functional::forbidden:
// 			AssertThrow(false, dealii::ExcMessage("You need to initialise dual.Je_type."));
// 			break;
// 		
// 		case Heat::types::error_functional::L2_final:
// 			dual_interpolate_to_next_grid();
// 			dual_assemble_rhs_L2final();
// 			break;
// 			
// 		case Heat::types::error_functional::L2_global:
// 			dual_assemble_Je_L2global();
// 			dual_interpolate_to_next_grid();
// 			dual_assemble_rhs_L2global();
// 			break;
// 			
// 		case Heat::types::error_functional::mean_final:
// 			dual_interpolate_to_next_grid();
// 			dual_assemble_rhs_mean_final();
// 			break;
// 			
// 		case Heat::types::error_functional::mean_global:
// 			dual_assemble_Je_mean_global();
// 			dual_interpolate_to_next_grid();
// 			dual_assemble_rhs_mean_global();
// 			break;
// 			
// 		case Heat::types::error_functional::point:
// 			dual_interpolate_to_next_grid();
// 			dual_assemble_rhs_point_final();
// 			break;
// 			
// 		default:
// 			AssertThrow(false, dealii::ExcMessage("Your dual.Je_type is unknown, please check your inputs."));
// 	}
// }


// ////////////////////////////////////////////////////////////////////////////////
// //////////  L2Error_global-Evaluation (J(phi)=1/(L2-Norm(e))*(e,phi)) //////////////////
// ////////////////////////////////////////////////////////////////////////////////
// 
// template<int dim>
// void
// Heat_DWR__cGp_dG0__cGq_cG1<dim>::
// dual_assemble_Je_L2global() {
// 	Assert(grid.use_count(), dealii::ExcNotInitialized());
// 	Assert(dual.iterator.slab->dual.fe.use_count(), dealii::ExcNotInitialized());
// 	Assert(dual.iterator.slab->dual.mapping.use_count(), dealii::ExcNotInitialized());
// 	Assert(dual.iterator.slab->dual.constraints.use_count(), dealii::ExcNotInitialized());
// 	
// 	dual.Je = 0;
// 	
//  	dealii::QGauss<dim> quad (dual.iterator.slab->dual.fe->tensor_degree()+2);
// 	
// 	// Setup a FE_Values object.
// 	dealii::FEValues<dim> fe_values(
// 		*(dual.iterator.slab->dual.mapping),
// 		*(dual.iterator.slab->dual.fe),
// 		quad,
// 		dealii::update_values |
// 		dealii::update_gradients |
// 		dealii::update_quadrature_points |
// 		dealii::update_JxW_values);
// 	
// 	dealii::Vector<double> local_dual_Je (
// 		dual.iterator.slab->dual.fe->dofs_per_cell
// 	);
// 	
// 	// stationärer Fall
// 	std::vector<double> exact_solution_values(quad.size());
// 	std::vector<double> u_h_values(quad.size());
// 	
// 	// Setup a small vector, to store the global dof indices.
// 	// NOTE: We are using a C++ standart template vector, not a deal.II
// 	// "Vector".
// 	// The FiniteElement object "fe" will tell us, how much dofs has each cell.
// 	std::vector< dealii::types::global_dof_index > local_dof_indices(
// 		dual.iterator.slab->dual.fe->dofs_per_cell
// 	);
// 	
// 	// Now we do the real work, looping over all cells to compute the cell
// 	// assemblys. For this we need an iterator (kind of a pointer), which allows
// 	// us, to iterate over each cell easily.
// 	// We initialise it with the first active cell of our triangulation.
// 	auto cell = dual.iterator.slab->dual.dof->begin_active();
// 	auto endc = dual.iterator.slab->dual.dof->end();
// 	
// 	for ( ; cell != endc; ++cell) {
// 		// First we have to compute the values of the gradients and
// 		// the JxW values.
// 		// The reinit of the fe_values object on the current cell will do this.
// 		fe_values.reinit(cell);
// 		
// 		// Initialise the full matrix for the cell assembly with 0.
// 		local_dual_Je = 0;
// 		
// 	// stationärer Fall	
// 		// Set up the exact solution vector
// 		function.BoundaryValues_dual->value_list(fe_values.get_quadrature_points(),
// 								   exact_solution_values);
// 		// Set up the computed solution vector
// 		fe_values.get_function_values(*(rit_In_uback->x),
// 									  u_h_values);
// 		
// 		// Now loop over all shape function combinations and quadrature points
// 		// to get the assembly.
// 		for (unsigned int i(0); i < dual.iterator.slab->dual.fe->dofs_per_cell; ++i)
// 		for (unsigned int q(0); q < quad.size(); ++q) {
// 			local_dual_Je(i) +=
// 				(1./L2Error_global) *
// 				(((exact_solution_values[q]-u_h_values[q]) *
// 				fe_values.shape_value(i,q))* 
// 				fe_values.JxW(q));
// 		}
// 		
// 		// Store the global indices into the vector local_dof_indices.
// 		// The cell object will give us the information.
// 		Assert(
// 			(local_dof_indices.size() == dual.iterator.slab->dual.fe->dofs_per_cell),
// 			dealii::ExcNotInitialized()
// 		);
// 		cell->get_dof_indices(local_dof_indices);
// 		
// 		// Copy the local assembly to the global matrix.
// 		// We use the constraint object, to set all constraints with that step.
// 		Assert(dual.iterator.slab->dual.constraints.use_count(), dealii::ExcNotInitialized());
// 		
// 		dual.iterator.slab->dual.constraints->distribute_local_to_global(
// 			local_dual_Je, local_dof_indices, dual.Je
// 		);
// 		
// 	}
// 	
// }


// template<int dim>
// void
// Heat_DWR__cGp_dG0__cGq_cG1<dim>::
// dual_interpolate_to_next_grid() {
// 	dealii::VectorTools::interpolate_to_different_mesh(
// 		*(dual.iterator.slab_previous->dual.dof),
// 		*(dual.z_old),
// 		*(dual.iterator.slab->dual.dof),
// 		*(dual.iterator.slab->dual.constraints),
// 		*(dual.z_old_interpolated)
// 	);
// 	
// 	// Only needed if within dual_assemble_Je has to be sth computed because of 
// 	// the used error functional, for exmpl L^2error at final timepoint dual.Je 
// 	// is always 0.
// 	dealii::VectorTools::interpolate_to_different_mesh(
// 		*(dual.iterator.slab_previous->dual.dof),
// 		dual.Je_old,
// 		*(dual.iterator.slab->dual.dof),
// 		*(dual.iterator.slab->dual.constraints),
// 		dual.Je_old_interpolated
// 	);
// }


// template<int dim>
// void
// Heat_DWR__cGp_dG0__cGq_cG1<dim>::
// dual_assemble_rhs_L2global() {
// 	dual.system_rhs = 0;
// 	(dual.A).vmult(dual.system_rhs,*(dual.z_old_interpolated));
// 	dual.system_rhs *= (-((data.tau_n)/2.));
// 	(dual.M).vmult_add(dual.system_rhs,*(dual.z_old_interpolated));
// 	(dual.system_rhs).add(((data.tau_n)/2.),dual.Je_old_interpolated);
// 	(dual.system_rhs).add(((data.tau_n)/2.),dual.Je);
// }


// template<int dim>
// void
// Heat_DWR__cGp_dG0__cGq_cG1<dim>::
// dual_solve() {
// 	////////////////////////////////////////////////////////////////////////////
// 	// apply Dirichlet boundary values
// 	std::map<dealii::types::global_dof_index, double> boundary_values;
// 	dealii::VectorTools::interpolate_boundary_values(
// 		*(dual.iterator.slab->dual.dof),
// 		static_cast< dealii::types::boundary_id > (
// 			Heat::types::boundary_id::Dirichlet
// 		),
// 		dealii::ZeroFunction<dim> (),
// 		boundary_values
// 	);
// 	
// 	dealii::MatrixTools::apply_boundary_values(
// 		boundary_values,
// 		dual.system_matrix,
// 		*(dual.z),
// 		dual.system_rhs
// 	);
// 	
// 	////////////////////////////////////////////////////////////////////////////
// 	// solve linear system
// 	dealii::SparseDirectUMFPACK iA;
// 	iA.initialize(dual.system_matrix);
// 	iA.vmult(*(dual.z), dual.system_rhs);
// 	
// 	////////////////////////////////////////////////////////////////////////////
// 	// distribute hanging node constraints on solution
// 	dual.iterator.slab->dual.constraints->distribute(*(dual.z));
// }

// template<int dim>
// void
// Heat_DWR__cGp_dG0__cGq_cG1<dim>::
// compute_Ieff() {
// 	switch (dual.Je_type) {
// 		case Heat::types::error_functional::forbidden:
// 			AssertThrow(false, dealii::ExcMessage("You need to initialise dual.Je_type."));
// 			break;
// 		
// 		case Heat::types::error_functional::L2_final:
// 			compute_Ieff_L2final();
// 			break;
// 			
// 		case Heat::types::error_functional::L2_global:
// 			compute_Ieff_L2global();
// 			break;
// 			
// 		case Heat::types::error_functional::mean_final:
// 			compute_Ieff_mean_final();
// 			break;
// 			
// 		case Heat::types::error_functional::mean_global:
// 			compute_Ieff_mean_global();
// 			break;
// 			
// 		case Heat::types::error_functional::point:
// 			compute_Ieff_point_final();
// 			break;
// 		
// 		default:
// 			AssertThrow(false, dealii::ExcMessage("Your dual.Je_type is unknown, please check your inputs."));
// 	}	
// }


// template<int dim>
// void
// Heat_DWR__cGp_dG0__cGq_cG1<dim>::
// compute_Ieff_L2global() {
// 	// Compute eta_K^n within fct. estimate() of class DWR_ErrorEstimator ans store
// 	// the local contributions of one time-interval I_n within the vector erroro_indicators
// 	// and all these vectors within the list dual.storage.eta.
// 	error_estimator.DWR->estimate(
// 		dual.storage.u,
// 		dual.storage.z,
// 		dual.storage.eta
// 	);
// 
// 	////////////////////////////////////////////////////////////////////////////
// 	////// Computaion of I_eff  ////////////////////////////////////////////////
// 	////////////////////////////////////////////////////////////////////////////
// 	
// 	// Sum up the local contributions of all cells within one time-interval of the 
// 	// local vector error_indicators (sum_(K in T_h)n_K^n). Then sum up all these
// 	// added values and store it within the double eta (=sum_(n=1)^N (sum_(K in T_h)n_K^n))
// 	auto In_eta_test(dual.storage.eta->begin());
// 	auto endIn_eta_test(dual.storage.eta->end());
// 	// Variable for estimated error eta
// 	double eta;
// 	eta = 0;
// 	for (; In_eta_test != endIn_eta_test; ++In_eta_test) {
// // 		for (unsigned int i=0; i < In_etat_test->x->size(); ++i) {
// // 			(*(In_etat_test->x))[i] = std::fabs((*(In_etat_test->x))[i]);
// // 		}
// 
// 		std::cout << "Groesse eta = " << In_eta_test->x->size() << std::endl;
// 		std::cout << "Summe eta_K = "<< std::accumulate (In_eta_test->x->begin(),
//                                 In_eta_test->x->end(), 0.) << std::endl;
// 		// local sum for each time-interval I_n
// 		double eta_k;
// 		eta_k = 0;
// 		eta_k = std::accumulate (In_eta_test->x->begin(),
//                                 In_eta_test->x->end(), 0.);
// 		// Add the local contribution of one time-interval I_n to the global eta.
// 		eta += eta_k;
// 	} // end of loop In_eta_test
// 	
// // 	// Computation of the exact error J(e) = J(u)-J(u_kh):
// 	// In now done within the function compute_global_STL2_error() (see below)
// 	
// // 	// Variable for effectivity index I_eff
// 	double I_eff;
// 	I_eff = 0.;
// 	// Computaion of effectivity index:
// 	I_eff = eta/L2Error_global;
// 	// Output of estimated error eta, exact error J(e) and effectivity index I_eff:
// 	std::cout << "eta = " << eta << std::endl;
// 	std::cout << "gl Fehler = " << L2Error_global << std::endl;
// 	std::cout << "I_eff = " << eta/L2Error_global << std::endl;
// 	
// 	// Add value of I_eff to table
// 	primal.convergence_table.add_value("J(e)",L2Error_global);
// 	primal.convergence_table.add_value("Eta",eta);
// 	primal.convergence_table.add_value("I_eff", I_eff);
// 
// }


// template<int dim>
// void
// Heat_DWR__cGp_dG0__cGq_cG1<dim>::
// compute_global_STL2_error() {
// 	auto In_u_error(primal.storage.u->begin());
// 	auto endIn_u_error(primal.storage.u->end());
// 	auto In_error(grid->slabs.begin());
// 	auto endIn_error(grid->slabs.end());
// 	// Variable for local L2Error
// 	double L2Error_local;
// 	L2Error_local = 0;
// // 	// Vector for global errors on each cell K (E_K in dealii notation)
// // 	dealii::Vector<double> global_diff (primal.iterator.slab->tria->n_active_cells());
// 	// Loop over all time-intervals I_n (n=1,...,N(=number of grids))
// 	for (unsigned int n{0};n <= grid->slabs.size(); ++n,++In_u_error) {
// 		
// 		if (n == grid->slabs.size()) {
// 			++In_error;
// 			primal.iterator.slab = In_error;
// 			dealii::Vector<double> difference_per_cell (primal.iterator.slab->tria->n_active_cells());
// 			const dealii::QTrapez<1> q_trapez;
// 			const dealii::QIterated<dim> q_iterated (q_trapez,20);
// 		
// 			__sync_synchronize();
// 			volatile const double ttt{n*data.tau_n};
// 			function.BoundaryValues_dual->set_time(ttt);
// 			__sync_synchronize();
// 			
// 			dealii::VectorTools::integrate_difference (*(primal.iterator.slab->primal.mapping),
// 													*(primal.iterator.slab->primal.dof),
// 													*(In_u_error->x),
// 													*(function.BoundaryValues_dual),
// 													difference_per_cell,
// 													q_iterated,//dealii::QGauss<dim>(4),//q_iterated, //dealii::QGauss<dim>(6),// q_iterated (alternativ)
// 													dealii::VectorTools::L2_norm);
// 			
// 			double L2_error_error_end = difference_per_cell.l2_norm();
// 			L2_error_error_end *= L2_error_error_end;
// 			L2_error_error_end *= (data.tau_n/2.);
// 			L2Error_local += L2_error_error_end;
// 		} // end if (n == grid->slabs.size())
// 		else if (n == 0) {
// 			primal.iterator.slab = In_error;
// 			dealii::Vector<double> difference_per_cell (primal.iterator.slab->tria->n_active_cells());
// 			const dealii::QTrapez<1> q_trapez;
// 			const dealii::QIterated<dim> q_iterated (q_trapez,20);
// 			
// 			__sync_synchronize();
// 			volatile const double ttt{n*data.tau_n};
// 			function.BoundaryValues_dual->set_time(ttt);
// 			__sync_synchronize();
// 			
// 			dealii::VectorTools::integrate_difference (*(primal.iterator.slab->primal.mapping),
// 													*(primal.iterator.slab->primal.dof),
// 													*(In_u_error->x),
// 													*(function.BoundaryValues_dual),
// 													difference_per_cell,
// 													q_iterated,//dealii::QGauss<dim>(4),//q_iterated, //dealii::QGauss<dim>(6),// q_iterated (alternativ)
// 													dealii::VectorTools::L2_norm);
// 			
// 			double L2_error_error_end = difference_per_cell.l2_norm();
// 			L2_error_error_end *= L2_error_error_end;
// 			L2_error_error_end *= (data.tau_n/2.);
// 			L2Error_local += L2_error_error_end;
// 		}
// 		else if (n == 1) {
// 			primal.iterator.slab = In_error;
// 			dealii::Vector<double> difference_per_cell (primal.iterator.slab->tria->n_active_cells());
// 			const dealii::QTrapez<1> q_trapez;
// 			const dealii::QIterated<dim> q_iterated (q_trapez,20);
// 			
// 			__sync_synchronize();
// 			volatile const double ttt{n*data.tau_n};
// 			function.BoundaryValues_dual->set_time(ttt);
// 			__sync_synchronize();
// 			
// 			dealii::VectorTools::integrate_difference (*(primal.iterator.slab->primal.mapping),
// 													*(primal.iterator.slab->primal.dof),
// 													*(In_u_error->x),
// 													*(function.BoundaryValues_dual),
// 													difference_per_cell,
// 													q_iterated,//dealii::QGauss<dim>(4),//q_iterated, //dealii::QGauss<dim>(6),// q_iterated (alternativ)
// 													dealii::VectorTools::L2_norm);
// 			
// 			double L2_error_error_end = difference_per_cell.l2_norm();
// 			L2_error_error_end *= L2_error_error_end;
// 			L2_error_error_end *= (data.tau_n);
// 			L2Error_local += L2_error_error_end;
// 		}
// 		else {
// 		++In_error;
// 		primal.iterator.slab = In_error;
// 		dealii::Vector<double> difference_per_cell (primal.iterator.slab->tria->n_active_cells());
// 		const dealii::QTrapez<1> q_trapez;
// 		const dealii::QIterated<dim> q_iterated (q_trapez,20);
// 		
// 		__sync_synchronize();
// 		volatile const double ttt{n*data.tau_n};
// 		function.BoundaryValues_dual->set_time(ttt);
// 		__sync_synchronize();
// 		
// 		dealii::VectorTools::integrate_difference (*(primal.iterator.slab->primal.mapping),
// 													*(primal.iterator.slab->primal.dof),
// 													*(In_u_error->x),
// 													*(function.BoundaryValues_dual),
// 													difference_per_cell,
// 													q_iterated,//dealii::QGauss<dim>(4),//q_iterated, //dealii::QGauss<dim>(6),// q_iterated (alternativ)
// 													dealii::VectorTools::L2_norm);
// 		
// 		double L2_error_error = difference_per_cell.l2_norm();
// 		L2_error_error *= L2_error_error;
// 		L2_error_error *= data.tau_n;
// 		L2Error_local += L2_error_error;
// 		
// 		}//end else
// 
// 	} // end of loop n over all time-intervals I_n
// 	L2Error_global = std::sqrt(L2Error_local);
// 	
// 	////////////////////////////////////////////////////////////////////////////
// 	/////////////// Rechteckregel //////////////////////////////////////////////
// 	////////////////////////////////////////////////////////////////////////////
// 	
// // 	auto In_u_error(dual.storage.u->begin());
// // 	auto endIn_u_error(dual.storage.u->end());
// // 	auto In_error(grid->slabs.begin());
// // 	auto endIn_error(grid->slabs.end());
// // 	Variable for local L2Error
// // 	double L2Error_local;
// // 	L2Error_local = 0;
// // 	Vector for global errors on each cell K (E_K in dealii notation)
// // 	dealii::Vector<double> global_diff (primal.iterator.slab->tria->n_active_cells());
// // 	Loop over all time-intervals I_n (n=1,...,N(=number of grids))
// // 	for (unsigned int n{1};n <= grid->slabs.size(); ++n,++In_error) {
// // 		
// // 			++In_u_error;
// // 			primal.iterator.slab = In_error;
// // 			dealii::Vector<double> difference_per_cell (primal.iterator.slab->tria->n_active_cells());
// // 			const dealii::QTrapez<1> q_trapez;
// // 			const dealii::QIterated<dim> q_iterated (q_trapez,10);
// // 		std::cout << "tau_n ist = " << data.tau_n << std::endl;
// // 			function.BoundaryValues->set_time(n*data.tau_n);
// // 			dealii::VectorTools::integrate_difference (*(primal.iterator.slab->dual.mapping),
// // 													*(primal.iterator.slab->dual.dof),
// // 													*(In_u_error->x),
// // 													*(function.BoundaryValues),
// // 													difference_per_cell,
// // 													q_iterated, //dealii::QGauss<dim>(6),// q_iterated (alternativ)
// // 													dealii::VectorTools::L2_norm);
// // 			
// // 			double L2_error_error_end = difference_per_cell.l2_norm();
// // 			L2_error_error_end *= L2_error_error_end;
// // 			L2_error_error_end *= (data.tau_n);
// // 			L2Error_local += L2_error_error_end;
// // 
// // 
// // 	} // end of loop n over all time-intervals I_n
// // 	L2Error_global = std::sqrt(L2Error_local);
// }


// template<int dim>
// void
// Heat_DWR__cGp_dG0__cGq_cG1<dim>::
// refine_grids_dwr() {
// 	
// 	// Build up a loop over all grids:
// 	auto Inth(grid->slabs.begin());
// 	auto endInth(grid->slabs.end());
// 	auto In_etath(dual.storage.eta->begin());
// 	
// 	for (; Inth != endInth; ++Inth) {
// 	
// 		for (unsigned int i=0; i < In_etath->x->size(); ++i) {
// 			(*(In_etath->x))[i] = std::fabs((*(In_etath->x))[i]);
// 		}
// 		
// 		////////////////////////////////////////////////////////////////////////////
// 		/// Schwegeler mesh-refinement strategy
// 		const double theta = 1.2; 										//theta aus (0.25,5).
// 		double eta_max = 0.;
// 		for (unsigned int i=0; i < In_etath->x->size(); ++i) {		//gives back the max. entry of error_indicators.
// 			if((*(In_etath->x))[i] > eta_max) {
// 				eta_max = (*(In_etath->x))[i];
// 			}
// 		}
// 		std::cout << "   eta_max = " << eta_max << std::endl;
// 		double eta_ks = std::accumulate (In_etath->x->begin(),
// 										 In_etath->x->end(), 0.);
// 		double Zellen = Inth->tria->n_active_cells();					//Number of active cells.
// 
// 		double mu = theta*(eta_ks/Zellen);								//mu.
// 		std::cout << "   mu = " << mu << std::endl;
// 		if(mu > eta_max) {mu=mu/2.;}									//while mu > eta_max: mu:=mu/2.
// 		int Anz_ZuVerfZellen = 0;
// 		for (unsigned int i=0; i < In_etath->x->size(); ++i) {		//determines how many cells schould be refined
// 			if((*(In_etath->x))[i] > mu) {							//(those cells, whose related error_indicators entry
// 				Anz_ZuVerfZellen = Anz_ZuVerfZellen + 1.;				//is > than mu).
// 			}
// 		}
// 		std::cout << "Anz_ZuVerfZellen=" << Anz_ZuVerfZellen << std::endl;
// 		double topfractioncells = Anz_ZuVerfZellen/Zellen; 				//percentage of to be refined cells.
// 		std::cout << "topfractioncells=" << topfractioncells << std::endl;
// 		dealii::GridRefinement::refine_and_coarsen_fixed_number (*(Inth->tria),	//mark cells, which schould be refined
// 																*(In_etath->x),
// 																topfractioncells, 0.0);
// 		
// 		Inth->tria->refine_global();
// 	
// 	++In_etath;
// 	} // end for-loop Inth (over all grids)
// }


// template<int dim>
// void
// Heat_DWR__cGp_dG0__cGq_cG1<dim>::
// solve_dual_problem() {
// 	auto Inth_dual(grid->slabs.rbegin());
// 	auto endIn_dual(grid->slabs.rend());
// 	auto Inth_dual_prev(grid->slabs.rbegin());
// 	auto endIn_dual_prev(grid->slabs.rend());
// 	auto In_zth(dual.storage.z->rbegin());
// 	auto endIn_zth(dual.storage.z->rend());
// 	auto In_uth_test(dual.storage.u->rbegin());
// 	auto endIn_uth_test(dual.storage.u->rend());
// 	
// 	for (unsigned int n = ((data.T-data.t0)/data.tau_n); n >= 0 ; --n) {
// 		if (n == ((data.T-data.t0)/data.tau_n)) {
// 			// Compute "initial condition" z_N and store it in dual.z_old and
// 			// afterwards store this in the last element of list dual.storage.z (recognize
// 			// that the list-iterator of dual.storage.z ist starting at the last element (dual.storage.z->rbegin())
// 			
// 			// Set iterator for primal solution u_kh (needed only for global L2-Error)
// 			rit_In_uback = In_uth_test;
// 			// Output of time_step 1 at time-point t_N
// 			data.dual_time = n*data.tau_n;
// 			++data.dual_timestep_number;
// 			std::cout << "Time step " << data.dual_timestep_number << " at t = "
// 						<< data.dual_time << std::endl;
// 			// Set iterators
// 			dual.iterator.slab = Inth_dual;
// 			dual.iterator.slab_previous = Inth_dual_prev;
// 			std::cout << "Zellen it = " << dual.iterator.slab->tria->n_active_cells() << std::endl;
// 			std::cout << "Zellen it_prev = " << dual.iterator.slab_previous->tria->n_active_cells() << std::endl;
// 			// Initialize rhs_vectors dual.Je_old and dual.Je
// 			dual.Je_old.reinit(dual.iterator.slab->dual.dof->n_dofs());
// 			// Set time to t_N and compute z_N (at timepoint t_N = 0.5)
// 			__sync_synchronize();
// 			dual_set_time(data.T);
// 			__sync_synchronize();
// 			
// 			dual_compute_initial_condition();
// 
// 			// Store initial condition z_N (dual.z_old) in the last element of list dual.storage.z
// 			dual.storage.z->back().x->reinit(grid->slabs.back().dual.dof->n_dofs());
// 			*(dual.storage.z->back().x) = *(dual.z_old);
// 			
// 		}
// 		else if (n == ((data.T-data.t0)/data.tau_n)-1) {
// 			// Compute z_N-1 at time point t_N-1
// 			
// 			// Output of time_step N+1 at time-point t_0
// 			data.dual_time = n*data.tau_n;
// 			++data.dual_timestep_number;
// 			std::cout << "Time step " << data.dual_timestep_number << " at t = "
// 						<< data.dual_time << std::endl;
// 			// Increase and set iterator for primal solution u_kh (needed only for global L2-Error)
// 			++In_uth_test;
// 			rit_In_uback = In_uth_test;
// 
// 			++In_zth; // "increase" iterator of list dual.storage.z. (running backward from last to first element)
// 			++Inth_dual; // "increase" iterator of list In (grids).
// 			// Set iterators
// 			dual.iterator.slab = Inth_dual;
// 			dual.iterator.slab_previous = Inth_dual_prev;
// 			
// 			std::cout << "Zellen it = " << dual.iterator.slab->tria->n_active_cells() << std::endl;
// 			std::cout << "Zellen it_prev = " << dual.iterator.slab_previous->tria->n_active_cells() << std::endl;
// 			
// 			dual_reinit();
// 			dual_assemble_system();
// 			__sync_synchronize();
// 			volatile const double ttt{n*data.tau_n};
// 			dual_set_time(ttt);
// 			__sync_synchronize();
// 			
// 			dual_assemble_rhs_at_t_Nminus1();
// 
// 			dual_solve();
// 
// 			// Save current solution (dual.z) in list dual.storage.z;
// 			In_zth->x->reinit(dual.iterator.slab->dual.dof->n_dofs());
// 			*(In_zth->x) = *(dual.z);
// 			
// 			// Save current solution (dual.z) and rhs_value (dual.Je) for 
// 			// next time step in dual.z_old and dual.Je_old
// 			// (needed within dual_assemble_rhs() of next time step) 
// 			dual.z_old = std::make_shared< dealii::Vector<double> > ();
// 			dual.z_old->reinit(dual.iterator.slab->dual.dof->n_dofs());
// 			*(dual.z_old) = *(dual.z);
// 			dual.Je_old.reinit(dual.iterator.slab->dual.dof->n_dofs());
// 			dual.Je_old = dual.Je;
// 			
// 			++Inth_dual_prev;
// 		}
// 		else if (n == 0) {
// 			// Compute z_0 at time point t_0
// 
// 			// Output of time_step N+1 at time-point t_0
// 			data.dual_time = n*data.tau_n;
// 			++data.dual_timestep_number;
// 			std::cout << "Time step " << data.dual_timestep_number << " at t = "
// 						<< data.dual_time << std::endl;
// 			// Increase and set iterator for primal solution u_kh (needed only for global L2-Error)
// 			++In_uth_test;
// 			rit_In_uback = In_uth_test;
// 			++In_zth; // increase iterator of list dual.storage.z, points now on first element of list In-Z
// 			// Set iterators
// 			dual.iterator.slab = Inth_dual;
// 			dual.iterator.slab_previous = Inth_dual_prev;
// 			
// 			std::cout << "Zellen it = " << dual.iterator.slab->tria->n_active_cells() << std::endl;
// 			std::cout << "Zellen it_prev = " << dual.iterator.slab_previous->tria->n_active_cells() << std::endl;
// 			
// 			dual_reinit();
// 			dual_assemble_system();
// 			__sync_synchronize();
// 			volatile const double ttt{n*data.tau_n};
// 			dual_set_time(ttt);
// 			__sync_synchronize();
// 			
// 			dual_assemble_rhs();
// 			
// 			dual_solve();
// 			
// 			// Store z_0 (dual.z) in the first element of list dual.storage.z.
// 			In_zth->x->reinit(dual.iterator.slab->dual.dof->n_dofs());
// 			*(In_zth->x) = *(dual.z);
// 			
// 			break;
// 		}
// 		else {
// 			// Compute z_N-2,...,z_1
// 			
// 			// Output of time_steps 2,...,N at time-points t_N-1,...,t_1
// 			data.dual_time = n*data.tau_n;
// 			++data.dual_timestep_number;
// 			std::cout << "Time step " << data.dual_timestep_number << " at t = "
// 						<< data.dual_time << std::endl;
// 			// Increase and set iterator for primal solution u_kh (needed only for global L2-Error)
// 			++In_uth_test;
// 			rit_In_uback = In_uth_test;
// 
// 			++In_zth; // "increase" iterator of list dual.storage.z. (running backward from last to first element)
// 			++Inth_dual; // "increase" iterator of list In (grids).
// 			// Set iterators
// 			dual.iterator.slab = Inth_dual;
// 			dual.iterator.slab_previous = Inth_dual_prev;
// 			
// 			std::cout << "Zellen it = " << dual.iterator.slab->tria->n_active_cells() << std::endl;
// 			std::cout << "Zellen it_prev = " << dual.iterator.slab_previous->tria->n_active_cells() << std::endl;
// 			
// 			dual_reinit();
// 			dual_assemble_system();
// 			
// 			__sync_synchronize();
// 			volatile const double ttt{n*data.tau_n};
// 			dual_set_time(ttt);
// 			__sync_synchronize();
// 			dual_assemble_rhs();
// 			
// 			dual_solve();
// 			
// 			// Save current solution (dual.z) in list dual.storage.z
// 			In_zth->x->reinit(dual.iterator.slab->dual.dof->n_dofs());
// 			*(In_zth->x) = *(dual.z);
// 			
// 			// Save current solution (dual.z) and rhs_value (dual.Je) for 
// 			// next time step in dual.z_old and dual.Je_old
// 			// (needed within dual_assemble_rhs() of next time step) 
// 			dual.z_old = std::make_shared< dealii::Vector<double> > ();
// 			dual.z_old->reinit(dual.iterator.slab->dual.dof->n_dofs());
// 			*(dual.z_old) = *(dual.z);
// 			dual.Je_old.reinit(dual.iterator.slab->dual.dof->n_dofs());
// 			dual.Je_old = dual.Je;
// 			
// 			++Inth_dual_prev;
// 		}
// 	} //end for-loop n dual
// }


// template<int dim>
// void
// Heat_DWR__cGp_dG0__cGq_cG1<dim>::
// run() {
// 	
// ////////////////////////////////////////////////////////////////////////////////
// /////////////// DWR-loop ///////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////
// 
// 	// Start DWR-loop cycle =^ one DWR-step
// 	const int endcycle = 2;
// 	for (unsigned int cycle = 0; cycle < endcycle /*true*/; ++cycle) {
// 		std::cout << "======================" << std::endl;
// 		std::cout << "Refinement Cycle: " << cycle << std::endl;
// 		std::cout << "======================" << std::endl;
// 		
// 		// Set time and running variables new at the beginning of every DWR-step
// 		data.primal_time = 0.;
// 		data.dual_time = data.T;
// 		data.primal_timestep_number = 0;
// 		data.dual_timestep_number = 0;
// 		
// 		// Prepare grids and dofs
// 		grid->set_boundary_indicators();
// 		grid->distribute();
// 		
// 		////////////////////////////////////////////////////////////////////////
// 		/////////////// begin primal ///////////////////////////////////////////
// 		////////////////////////////////////////////////////////////////////////
// 		
// 		// solve complete forward TMS
// 		solve_primal_problem();
// 		
// 		// For convergence results
// 		primal_process_solution(data.T, cycle);
// 
// 		////////////////////////////////////////////////////////////////////////
// 		///////////////////////// end primal ///////////////////////////////////
// 		////////////////////////////////////////////////////////////////////////
// 
// 
// 		////////////////////////////////////////////////////////////////////////
// 		///////////////////////// begin dual ///////////////////////////////////
// 		////////////////////////////////////////////////////////////////////////
// 		
// 		solve_dual_problem();
// 		
// 		////////////////////////////////////////////////////////////////////////
// 		///////////////////////// end dual /////////////////////////////////////
// 		////////////////////////////////////////////////////////////////////////
// 		
// 		// Compute I_eff:
// 		std::cout << "dual.storage.eta hat Groesse = " << dual.storage.eta->size() << std::endl;
// 	
// 		compute_Ieff();
// 
// 		// Solution Output
// 		if (cycle == endcycle-1) {
// 			primal_and_dual_solution_output();
// 		} // end if Soltion Output (cycle == endcycle-1)
// 		
// 		// Refine Grids
// 		refine_grids_dwr();
// 	
// 	} // end DWR-loop
// 	
// 	////////////////////////////////////////////////////////////////////////////
// 	//////////////// end DWR-loop "cycle" //////////////////////////////////////
// 	////////////////////////////////////////////////////////////////////////////


} // namespace

#include "Heat_DWR__cGp_dG0__cGq_cG1.inst.in"
