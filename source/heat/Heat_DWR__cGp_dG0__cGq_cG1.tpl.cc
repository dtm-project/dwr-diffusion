/**
 * @file Heat_DWR__cGp_dG0__cGq_cG1.tpl.cc
 *
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 *
 * @date 2018-07-19, finialised dwr-loop, UK
 * @date 2018-03-08, primal problem, UK
 * @date 2018-03-06, new implementation, UK
 * @date 2018-03-05, UK
 * @date 2017-08-01, Heat/DWR, MPB, UK
 *
 * @brief Heat/DWR Problem with primal solver: cG(p)-dG(0) and dual solver: cG(q)-cG(1)
 */

/*  Copyright (C) 2012-2018 by Uwe Koecher and contributors                   */
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
#include <heat/Heat_DWR__cGp_dG0__cGq_cG1.tpl.hh>

#include <heat/grid/Grid_DWR_Selector.tpl.hh>

#include <heat/Density/Density_Selector.tpl.hh>
#include <heat/Diffusion/Diffusion_Selector.tpl.hh>
#include <heat/Force/Force_Selector.tpl.hh>

#include <heat/InitialValue/InitialValue_Selector.tpl.hh>
#include <heat/DirichletBoundary/DirichletBoundary_Selector.tpl.hh>

#include <heat/ExactSolution/ExactSolution_Selector.tpl.hh>

#include <heat/types/boundary_id.hh>

#include <heat/assembler/L2_MassAssembly.tpl.hh>
#include <heat/assembler/L2_LaplaceAssembly.tpl.hh>

#include <heat/assembler/L2_ForceConstrainedAssembly.tpl.hh>
template <int dim>
using ForceAssembler = heat::Assemble::L2::ForceConstrained::Assembler<dim>;

#include <heat/assembler/L2_Je_global_L2L2_Assembly.tpl.hh>

// DEAL.II includes
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

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
run() {
	// check
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
	
	// check primal time discretisation
	if ((parameter_set->fe.primal.time_type.compare("dG") == 0) &&
		(parameter_set->fe.primal.r == 0)) {
		DTM::pout
			<< "primal time discretisation = dG("
			<< parameter_set->fe.primal.r
			<< ")-Q_"
			<< parameter_set->fe.primal.time_type_support_points
			<< std::endl;
	}
	else {
		AssertThrow(
			false,
			dealii::ExcMessage(
				"primal time discretisation unknown"
			)
		);
	}
	
	// check dual time discretisation
	if ((parameter_set->fe.dual.time_type.compare("cG") == 0) &&
		(parameter_set->fe.dual.s == 1)) {
		DTM::pout
			<< "dual time discretisation = cG("
			<< parameter_set->fe.dual.s
			<< ")-Q_"
			<< parameter_set->fe.dual.time_type_support_points
			<< std::endl;
	}
	else {
		AssertThrow(
			false,
			dealii::ExcMessage(
				"dual time discretisation unknown"
			)
		);
	}
	
	// determine setw value for dwr loop number of data output filename
	setw_value_dwr_loops = static_cast<unsigned int>(
		std::floor(std::log10(parameter_set->dwr.loops))+1
	);
	
	init_grid();
	init_functions();
	
	// DWR loop:
	DTM::pout
		<< std::endl
		<< "*******************************************************************"
		<< "*************" << std::endl
		<< "dwr loops = " << parameter_set->dwr.loops << std::endl
		<< std::endl;
	
	for (unsigned int dwr_loop{0}; dwr_loop < parameter_set->dwr.loops; ++dwr_loop) {
		DTM::pout
			<< "***************************************************************"
			<< "*****************" << std::endl
			<< "dwr loop = " << dwr_loop+1 << std::endl;
		
		convergence_table.add_value("DWR-loop", dwr_loop+1);
		
		grid->set_boundary_indicators();
		grid->distribute();
		
		// primal problem:
		primal_reinit_storage();
		primal_init_data_output();
		primal_do_forward_TMS(dwr_loop);
		
		// dual problem
		dual_reinit_storage();
		dual_init_data_output();
		dual_do_backward_TMS(dwr_loop);
		
		// error estimation
		eta_reinit_storage();
		compute_error_indicators();
		compute_effectivity_index();
		
		// do space-time mesh refinements and coarsenings only if we have
		// another dwr-loop
		if ((dwr_loop+1) < parameter_set->dwr.loops)
			refine_and_coarsen_space_time_grid();
	}
	
	write_convergence_table_to_tex_file();
}


////////////////////////////////////////////////////////////////////////////////
// protected member functions (internal use only)
//

template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
init_grid() {
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
	
	////////////////////////////////////////////////////////////////////////////
	// init grid from input parameter file spec.
	//
	{
		heat::grid::Selector<dim> selector;
		selector.create_grid(
			parameter_set->Grid_Class,
			parameter_set->Grid_Class_Options,
			parameter_set->TriaGenerator,
			parameter_set->TriaGenerator_Options,
			grid
		);
		
		Assert(grid.use_count(), dealii::ExcNotInitialized());
	}
	
	////////////////////////////////////////////////////////////////////////////
	// initialize slabs of grid
	//
	
	Assert((parameter_set->fe.primal.p), dealii::ExcInvalidState());
	Assert(
		(parameter_set->fe.primal.p < parameter_set->fe.dual.q),
		dealii::ExcInvalidState()
	);
	
	Assert((parameter_set->t0 >= 0), dealii::ExcInvalidState());
	Assert((parameter_set->t0 < parameter_set->T), dealii::ExcInvalidState());
	Assert((parameter_set->tau_n > 0), dealii::ExcInvalidState());
	
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	grid->initialize_slabs(
		parameter_set->fe.primal.p,
		parameter_set->fe.dual.q,
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
init_functions() {
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
	// density function density:
	{
		heat::diffusion::Selector<dim> selector;
		selector.create_function(
			parameter_set->density_function,
			parameter_set->density_options,
			function.density
		);
		
		Assert(function.density.use_count(), dealii::ExcNotInitialized());
	}
	
	// diffusion function epsilon:
	{
		heat::diffusion::Selector<dim> selector;
		selector.create_function(
			parameter_set->epsilon_function,
			parameter_set->epsilon_options,
			function.epsilon
		);
		
		Assert(function.epsilon.use_count(), dealii::ExcNotInitialized());
	}
	
	// force function f:
	{
		heat::force::Selector<dim> selector;
		selector.create_function(
			parameter_set->force_function,
			parameter_set->force_options,
			function.f
		);
		
		Assert(function.f.use_count(), dealii::ExcNotInitialized());
	}
	
	// initial value function u_0:
	{
		heat::initial_value::Selector<dim> selector;
		selector.create_function(
			parameter_set->initial_value_u0_function,
			parameter_set->initial_value_u0_options,
			function.u_0
		);
		
		Assert(function.u_0.use_count(), dealii::ExcNotInitialized());
	}
	
	// dirichlet boundary function u_D:
	{
		heat::dirichlet_boundary::Selector<dim> selector;
		selector.create_function(
			parameter_set->dirichlet_boundary_u_D_function,
			parameter_set->dirichlet_boundary_u_D_options,
			function.u_D
		);
		
		Assert(function.u_D.use_count(), dealii::ExcNotInitialized());
	}
	
	// exact solution function u_E (if any)
	{
		heat::exact_solution::Selector<dim> selector;
		selector.create_function(
			parameter_set->exact_solution_function,
			parameter_set->exact_solution_options,
			function.u_E
		);
		
		Assert(function.u_E.use_count(), dealii::ExcNotInitialized());
	}
}


////////////////////////////////////////////////////////////////////////////////
// primal problem
//

template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
primal_reinit_storage() {
	////////////////////////////////////////////////////////////////////////////
	// init storage containers for vector data:
	// NOTE: * primal space: time dG(0) method (having 1 independent solution)
	//       * primal solution dof vectors: u
	//
	
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	// get number of time steps N
	const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};
	
	primal.storage.u = std::make_shared< DTM::types::storage_data_vectors<1> > ();
	primal.storage.u->resize(N);
	
	{
		auto slab = grid->slabs.begin();
		for (auto &element : *primal.storage.u) {
			for (unsigned int j{0}; j < element.x.size(); ++j) {
				element.x[j] = std::make_shared< dealii::Vector<double> > ();
				
				Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
				Assert(slab->primal.dof.use_count(), dealii::ExcNotInitialized());
				Assert(
					slab->primal.dof->n_dofs(),
					dealii::ExcMessage("Error: slab->primal.dof->n_dofs() == 0")
				);
				
				// initialise dealii::Vector<double> with n_dofs components:
				element.x[j]->reinit(
					slab->primal.dof->n_dofs()
				);
			}
			++slab;
		}
	}
}


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
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
primal_assemble_rhs(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
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
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
primal_solve_slab_problem(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &u,
	const double t0
) {
	////////////////////////////////////////////////////////////////////////////
	// construct system matrix K = M + tau A
	//
	
	DTM::pout << "dwr-heat: construct system matrix K = M + tau A...";
	
	primal.K = std::make_shared< dealii::SparseMatrix<double> > ();
	primal.K->reinit(*slab->primal.sp);
	
	*primal.K = 0;
	primal.K->add(slab->tau_n(), *primal.A);
	primal.K->add(1.0, *primal.M);
	
	DTM::pout << " (done)" << std::endl;
	
	////////////////////////////////////////////////////////////////////////////
	// construct system right hand side vector b = M um + tau_n f0
	//
	
	DTM::pout << "dwr-heat: construct linear system rhs vector...";
	
	primal.b = std::make_shared< dealii::Vector<double> > ();
	primal.b->reinit( slab->primal.dof->n_dofs() );
	
	Assert(primal.M.use_count(), dealii::ExcNotInitialized());
	Assert(primal.um.use_count(), dealii::ExcNotInitialized());
	primal.M->vmult(*primal.b, *primal.um);
	
	primal.b->add(slab->tau_n(), *primal.f0);
	
	DTM::pout << " (done)" << std::endl;
	
	////////////////////////////////////////////////////////////////////////////
	// apply Dirichlet boundary values
	//
	
	DTM::pout << "dwr-heat: dealii::MatrixTools::apply_boundary_values...";
	std::map<dealii::types::global_dof_index, double> boundary_values;
	
	Assert(function.u_D.use_count(), dealii::ExcNotInitialized());
	function.u_D->set_time(t0);
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
		*u->x[0],
		*primal.b
	);
	
	DTM::pout << " (done)" << std::endl;
	
	////////////////////////////////////////////////////////////////////////////
	// condense hanging nodes in system matrix, if any
	//
	
	DTM::pout << "dwr-heat: slab->primal.constraints->condense(*primal.K)...";
	slab->primal.constraints->condense(*primal.K);
	
	DTM::pout << " (done)" << std::endl;
	
	////////////////////////////////////////////////////////////////////////////
	// solve linear system with direct solver
	//
	
	DTM::pout << "dwr-heat: setup direct lss and solve...";
	
	dealii::SparseDirectUMFPACK iA;
	iA.initialize(*primal.K);
	iA.vmult(*u->x[0], *primal.b);
	
	DTM::pout << " (done)" << std::endl;
	
	////////////////////////////////////////////////////////////////////////////
	// distribute constraints on solution
	//
	
	DTM::pout << "dwr-heat: primal.constraints->distribute...";
	slab->primal.constraints->distribute(
		*u->x[0]
	);
	DTM::pout << " (done)" << std::endl;
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
primal_do_forward_TMS(
	const unsigned int dwr_loop) {
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
	
	Assert(primal.storage.u.use_count(), dealii::ExcNotInitialized());
	Assert(primal.storage.u->size(), dealii::ExcNotInitialized());
	auto u = primal.storage.u->begin();
	
	////////////////////////////////////////////////////////////////////////////
	// interpolate (or project) initial value(s)
	//
	
	primal.um = std::make_shared< dealii::Vector<double> >();
	primal.um->reinit( slab->primal.dof->n_dofs() );
	
	Assert(function.u_0.use_count(), dealii::ExcNotInitialized());
	function.u_0->set_time(slab->t_m);
	
	Assert((slab != grid->slabs.end()), dealii::ExcInternalError());
	Assert(slab->primal.mapping.use_count(), dealii::ExcNotInitialized());
	Assert(slab->primal.dof.use_count(), dealii::ExcNotInitialized());
	Assert(primal.um.use_count(), dealii::ExcNotInitialized());
	
	dealii::VectorTools::interpolate(
		*slab->primal.mapping,
		*slab->primal.dof,
		*function.u_0,
		*primal.um
	);
	// call hanging nodes to make the result continuous again (Note: after the 
	// first dwr-loop the initial grid could have hanging nodes)
	slab->primal.constraints->distribute(*primal.um);
	
	// output "initial value solution" at initial time t0
	*u->x[0] = *primal.um;
	primal_do_data_output(slab,u,dwr_loop,true);
	
	// init error computations (for global L2(L2) goal functional)
	primal_init_error_computations();
	
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
		
		double t0{0};
		if (parameter_set->fe.primal.time_type_support_points.compare("Gauss-Radau")==0) {
			t0 = slab->t_n;
		}
		else if (parameter_set->fe.primal.time_type_support_points.compare("Gauss")==0) {
			t0 = tm + slab->tau_n()/2.;
		}
		
		const double tn = slab->t_n;
		
		DTM::pout
			<< "primal: solving problem on "
			<< "I_" << n << " = (" << tm << ", " << tn << ") "
			<< std::endl;
		
		if (slab != grid->slabs.begin()) {
			Assert(primal.un.use_count(), dealii::ExcNotInitialized());
			primal.um = std::make_shared< dealii::Vector<double> >();
			primal.um->reinit( slab->primal.dof->n_dofs() );
			
			// for n > 1 interpolate between two (different) spatial meshes
			// the solution u(t_n)|_{I_{n-1}}  to  u(t_m)|_{I_n}
			dealii::VectorTools::interpolate_to_different_mesh(
				// solution on I_{n-1}:
				*std::prev(slab)->primal.dof,
				*primal.un,
				// solution on I_n:
				*slab->primal.dof,
				*slab->primal.constraints,
				*primal.um
			);
		}
		
		// assemble slab problem
		primal_assemble_system(slab);
		primal_assemble_rhs(slab,t0);
		
		// solve slab problem (i.e. apply boundary values and solve for u0)
		primal_solve_slab_problem(slab,u,t0);
		
		////////////////////////////////////////////////////////////////////////
		// do postprocessing on the solution
		//
		
		// do error computations ( for global L2(L2) goal )
		primal_do_error_L2(slab,u);
		
		// evaluate solution u(t_n)
		primal.un = std::make_shared< dealii::Vector<double> >();
		primal.un->reinit( slab->primal.dof->n_dofs() );
		double zeta0 = 1.; // zeta0( t_n ) = 1. for dG(0)
		*primal.un = 0;
		primal.un->add(zeta0, *u->x[0]);
		
		// output solution at t_n
		primal_do_data_output(slab,u,dwr_loop,false);
		
		////////////////////////////////////////////////////////////////////////
		// prepare next I_n slab problem:
		//
		
		++n;
		++slab;
		++u;
		
		////////////////////////////////////////////////////////////////////////
		// allow garbage collector to clean up memory
		//
		
		primal.M = nullptr;
		primal.A = nullptr;
		primal.f0 = nullptr;
		
		primal.K = nullptr;
		primal.b = nullptr;
		
		DTM::pout << std::endl;
	}
	
	DTM::pout
		<< "primal: forward TMS problem done" << std::endl
		<< "*******************************************************************"
		<< "*************" << std::endl
		<< std::endl;
	
	////////////////////////////////////////////////////////////////////////////
	// allow garbage collector to clean up memory
	//
	
	primal.um = nullptr;
	primal.un = nullptr;
	
	////////////////////////////////////////////////////////////////////////////
	// finish error computation ( for global L2(L2) goal functional )
	//
	
	primal_finish_error_computations();
	DTM::pout
		<< "*******************************************************************"
		<< "*************" << std::endl
		<< "primal: || u - u_kh ||_L2(L2) = " << primal_L2_L2_error_u
		<< std::endl
		<< std::endl;
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
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


////////////////////////////////////////////////////////////////////////////////
// primal: L2(L2) error computation
//

template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
primal_init_error_computations() {
	primal_L2_L2_error_u = 0;
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
primal_do_error_L2(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &u) {
	////////////////////////////////////////////////////////////////////////////
	// compute L^2 in time error
	//
	
	// prepare L2(Omega) norm
	double norm_sqr{-1};
	
	dealii::QGauss<dim> quad_cell(
		slab->primal.fe->tensor_degree()+1
	);
	
	dealii::Vector<double> difference_per_cell(
		slab->primal.dof->n_dofs()
	);
	
	// prepare L2 in time norm
	double zeta0;
	double _t;
	
	// create quadrature for time integration of L2 in time integral on slab
	// m - number of Gauss points to be evaluated for \int_{I_n} ||err||^2_L2 dt
	[[maybe_unused]]unsigned int m(1);
	dealii::QGauss<1> quad_int_In(m);
// 	dealii::QGaussLobatto<1> quad_int_In(m);
// 	dealii::QGaussChebyshev<1> quad_int_In(m);
// 	dealii::QSimpson<1> qiter;
// // 	dealii::QMidpoint<1> qiter;
// // 	dealii::QTrapez<1> qiter;
// 	dealii::QIterated<1> quad_int_In(qiter, 4);
	
	std::vector< dealii::Point<1> > tq(quad_int_In.get_points());
	std::vector< double > w(quad_int_In.get_weights());
	
	// L_2 norm
	for (unsigned int q(0); q < quad_int_In.size(); ++q) {
		_t = tq[q][0];
		
		function.u_E->set_time(_t * slab->tau_n() + slab->t_m);
		
		zeta0 = 1.;
		
		////////////////////////////////////////////////////////////////////////
		dealii::Vector<double> u_trigger;
		u_trigger.reinit(
			slab->primal.dof->n_dofs()
		);
		
		// evalute space-time solution
		u_trigger.equ(zeta0, *u->x[0]);
		
		////////////////////////////////////////////////////////////////////////
		// u:
		difference_per_cell = 0;
		
		dealii::VectorTools::integrate_difference(
			*slab->primal.dof,
			u_trigger,
			*function.u_E,
			difference_per_cell,
			quad_cell,
			dealii::VectorTools::L2_norm
		);
		
		norm_sqr = difference_per_cell.norm_sqr();
		
		primal_L2_L2_error_u += w[q] * norm_sqr * slab->tau_n();
	}
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
primal_finish_error_computations() {
	primal_L2_L2_error_u = std::sqrt(primal_L2_L2_error_u);
	DTM::pout << "primal_L2_L2_error_u = " << primal_L2_L2_error_u << std::endl;
}


////////////////////////////////////////////////////////////////////////////////
// primal data output
//

template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
primal_init_data_output() {
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
	DTM::pout
		<< "primal solution data output: patches = "
		<< parameter_set->data_output.primal.patches
		<< std::endl;
	
	std::vector<std::string> data_field_names;
	data_field_names.push_back("u");
	
	std::vector< dealii::DataComponentInterpretation::DataComponentInterpretation > dci_field;
	dci_field.push_back(dealii::DataComponentInterpretation::component_is_scalar);
	
	primal.data_output = std::make_shared< DTM::DataOutput<dim> >();
	primal.data_output->set_data_field_names(data_field_names);
	primal.data_output->set_data_component_interpretation_field(dci_field);
	
	primal.data_output->set_data_output_patches(
		parameter_set->data_output.primal.patches
	);
	
	// set up which dwr loop(s) are allowed to make data output:
	primal.data_output_dwr_loop = -3; // -3 => not initialized
	if ( !parameter_set->data_output.primal.dwr_loop.compare("none") ) {
		primal.data_output_dwr_loop = -2;
	}
	else if ( !parameter_set->data_output.primal.dwr_loop.compare("all") ) {
		primal.data_output_dwr_loop = -1;
	}
	else if ( !parameter_set->data_output.primal.dwr_loop.compare("last") ) {
		Assert(parameter_set->dwr.loops > 0, dealii::ExcInternalError());
		primal.data_output_dwr_loop = parameter_set->dwr.loops-1;
	}
	else {
		primal.data_output_dwr_loop = std::stoi(parameter_set->data_output.primal.dwr_loop);
	}
	
	DTM::pout
		<< "primal solution data output: dwr loop = "
		<< primal.data_output_dwr_loop
		<< std::endl;
	
	// check if we use a fixed trigger interval, or, do output once on a I_n
	if ( !parameter_set->data_output.primal.trigger_type.compare("fixed") ) {
		primal.data_output_trigger_type_fixed = true;
	}
	else {
		primal.data_output_trigger_type_fixed = false;
	}
	
	// only for fixed
	primal.data_output_trigger = parameter_set->data_output.primal.trigger;
	
	if (primal.data_output_trigger_type_fixed) {
		DTM::pout
			<< "primal solution data output: using fixed mode with trigger = "
			<< primal.data_output_trigger
			<< std::endl;
	}
	else {
		DTM::pout
			<< "primal solution data output: using I_n mode (trigger adapts to I_n automatically)"
			<< std::endl;
	}
	
	primal.data_output_time_value = parameter_set->t0;
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
primal_do_data_output(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &u,
	const unsigned int dwr_loop,
	const bool dG_initial_value) {
	
	if (!( (primal.data_output_dwr_loop == -1) ||
		(primal.data_output_dwr_loop == static_cast<int>(dwr_loop)) ))
		return;
	
	if (primal.data_output_trigger <= 0) return;
	
	// adapt trigger value for I_n output mode
	if (!primal.data_output_trigger_type_fixed) {
		if (std::next(slab) != grid->slabs.end()) {
			primal.data_output_trigger = std::next(slab)->tau_n();
		}
		else {
			primal.data_output_trigger = slab->tau_n();
		}
	}
	
	primal.data_output->set_DoF_data(
		slab->primal.dof
	);
	
	auto u_trigger = std::make_shared< dealii::Vector<double> > ();
	u_trigger->reinit(
		slab->primal.dof->n_dofs()
	);
	
	std::ostringstream filename;
	filename
		<< "solution-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop+1;
	
	double &t{primal.data_output_time_value};
	
	if (dG_initial_value) {
		// NOTE: for dG-in-time discretisations the initial value function
		//       does not belong to the set of dof's. Thus, we need a special
		//       implementation here to output "primal.um".
		
		u_trigger->equ(1., *u->x[0]); // NOTE: this must be primal.um!
		
		primal.data_output->write_data(
			filename.str(),
			u_trigger,
			t
		);
		
		t += primal.data_output_trigger;
	}
	else {
		for ( ; t <= slab->t_n; t += primal.data_output_trigger) {
			[[maybe_unused]] const double _t{ (t - slab->t_m) / slab->tau_n() };
			
			const double zeta0{1.};
			
			// evalute space-time solution
			u_trigger->equ(zeta0, *u->x[0]);
			
			primal.data_output->write_data(
				filename.str(),
				u_trigger,
				t
			);
		}
	}
	
	// check if data for t=T was written
	if (std::next(slab) == grid->slabs.end()) {
	if (primal.data_output_trigger_type_fixed) {
		if ( t > slab->t_n ) {
			// overshoot of time variable; manually set to t = T and do data output
			t = slab->t_n;
			
			[[maybe_unused]] const double _t{ (t - slab->t_m) / slab->tau_n() };
			
			const double zeta0{1.};
			
			// evalute space-time solution
			u_trigger->equ(zeta0, *u->x[0]);
			
			primal.data_output->write_data(
				filename.str(),
				u_trigger,
				t
			);
		}
	}}
}


////////////////////////////////////////////////////////////////////////////////
// dual problem
//

template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
dual_reinit_storage() {
	////////////////////////////////////////////////////////////////////////////
	// init storage containers for vector data:
	// NOTE: * dual space: time cG(1) method (having 2 independent solutions)
	//       * dual solution dof vectors: z
	//
	
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	// get number of time steps N
	const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};
	
	dual.storage.z = std::make_shared< DTM::types::storage_data_vectors<2> > ();
	dual.storage.z->resize(N);
	
	{
		auto slab = grid->slabs.begin();
		for (auto &element : *dual.storage.z) {
			for (unsigned int j{0}; j < element.x.size(); ++j) {
				element.x[j] = std::make_shared< dealii::Vector<double> > ();
				
				Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
				Assert(slab->dual.dof.use_count(), dealii::ExcNotInitialized());
				Assert(
					slab->dual.dof->n_dofs(),
					dealii::ExcMessage("Error: slab->dual.dof->n_dofs() == 0")
				);
				
				// initialise dealii::Vector<double> with n_dofs components:
				element.x[j]->reinit(
					slab->dual.dof->n_dofs()
				);
			}
			++slab;
		}
	}
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
dual_assemble_system(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab
) {
	// ASSEMBLY MASS MATRIX ////////////////////////////////////////////////////
	dual.M = std::make_shared< dealii::SparseMatrix<double> > ();
	dual.M->reinit(*slab->dual.sp);
	
	*dual.M = 0;
	{
		heat::Assemble::L2::Mass::
		Assembler<dim> assemble_mass(
			dual.M,
			slab->dual.dof,
			slab->dual.fe,
			slab->dual.mapping,
			slab->dual.constraints
		);
		
		Assert(function.density.use_count(), dealii::ExcNotInitialized());
		assemble_mass.set_density(function.density);
		
		DTM::pout << "dwr-heat: assemble mass matrix...";
		assemble_mass.assemble();
		DTM::pout << " (done)" << std::endl;
	}
	
	// ASSEMBLY STIFFNESS MATRIX ///////////////////////////////////////////////
	dual.A = std::make_shared< dealii::SparseMatrix<double> > ();
	dual.A->reinit(*slab->dual.sp);
	
	*dual.A = 0;
	{
		heat::Assemble::L2::Laplace::
		Assembler<dim> assemble_stiffness_cell_terms (
			dual.A,
			slab->dual.dof,
			slab->dual.fe,
			slab->dual.mapping,
			slab->dual.constraints
		);
		
		Assert(function.epsilon.use_count(), dealii::ExcNotInitialized());
		assemble_stiffness_cell_terms.set_epsilon_function(function.epsilon);
		
		DTM::pout << "dwr-heat: assemble cell stiffness matrix...";
		assemble_stiffness_cell_terms.assemble();
		DTM::pout << " (done)" << std::endl;
	}
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
dual_assemble_rhs(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &u,
	const unsigned int &n,
	const double &t0,
	const double &t1
) {
	////////////////////////////////////////////////////////////////////////////
	// NOTE: this is only for a global L2(L2) goal functional
	//
	
	Assert(function.u_E.use_count(), dealii::ExcNotInitialized());
	
	// init assembler:
	auto assemble_Je = std::make_shared<
		heat::Assemble::L2::Je_global_L2L2::Assembler<dim> > (
		slab->dual.dof,
		slab->dual.fe,
		slab->dual.mapping,
		slab->dual.constraints
	);
	
	////////////////////////////////////////////////////////////////////////////
	// NOTE: forward problem is dG(0) (constant in time and JUMPS in t_m = t0),
	//       therefore we need the reconstruction on dual cG(1)-Q_GL(2) here.
	//
	// NOTE: u_h(t0) = u_h(t0)|_{I_{n-1}}
	//
	
	// interpolate primal solution u_h(t0) to dual solution space
	dual.u0 = std::make_shared< dealii::Vector<double> > ();
	dual.u0->reinit( slab->dual.dof->n_dofs() );
	
	auto u0_on_primal = std::make_shared< dealii::Vector<double> > ();
	u0_on_primal->reinit( slab->primal.dof->n_dofs() );
	*u0_on_primal = 0;
	
	// determine u0_on_primal depending on dual time type support points
	if (parameter_set->fe.dual.time_type_support_points.compare("Gauss-Lobatto")==0) {
		if ( n > 1 ) {
			// n > 1:
			//   get u_h(t0) from:    Omega_h^primal x I_{n-1} (t_{n-1})
			//   (1) interpolated to: Omega_h^primal x I_{n} (t_m) => u0_on_primal
			//   (2) interpolated to: Omega_h^dual x I_{n} (t_m)   => dual.u0
			
			// (1) interpolate_to_different_mesh (in primal):
			//     - needs the same fe: dof1.get_fe() = dof2.get_fe()
			//     - allow different triangulations: dof1.get_tria() != dof2.get_tria()
			
			dealii::VectorTools::interpolate_to_different_mesh(
				// solution on I_{n-1}:
				*std::prev(slab)->primal.dof,
				*std::prev(u)->x[0],
				// solution on I_n:
				*slab->primal.dof,
				*slab->primal.constraints,
				*u0_on_primal
			);
			
			// (2) interpolate primal -> dual:
			//     - needs the same tria: dof1.get_tria() == dof2.get_tria()
			//     - allow different FE-spaces: dof1.get_fe() != dof2.get_fe()
			dealii::FETools::interpolate(
				// primal solution
				*slab->primal.dof,
				*u0_on_primal,
				// dual solution
				*slab->dual.dof,
				*slab->dual.constraints,
				*dual.u0
			);
			
			u0_on_primal = nullptr;
		}
		else {
			// n == 1: interpolate initial value function u_0
			// firstly to primal space and finally from that into the dual space
			// to keep the original error of the jump u^m(t_0) of the primal problem
			// NOTE: dual.u0(t_0) corresponds to primal.u^-(t_0)
			
			auto primal_um_on_t0 = std::make_shared< dealii::Vector<double> > ();
			primal_um_on_t0->reinit( slab->primal.dof->n_dofs() );
			
			function.u_0->set_time(t0);
			
			dealii::VectorTools::interpolate(
				*slab->primal.mapping,
				*slab->primal.dof,
				*function.u_0,
				*primal_um_on_t0
			);
			// call hanging nodes to make the result continuous again (Note: after the 
			// first dwr-loop the initial grid could have hanging nodes)
			slab->primal.constraints->distribute(*primal_um_on_t0);
			
			dealii::FETools::interpolate(
				// primal solution
				*slab->primal.dof,
				*primal_um_on_t0,
				// dual solution
				*slab->dual.dof,
				*slab->dual.constraints,
				*dual.u0
			);
			
			primal_um_on_t0 = nullptr;
		}
	}
	else if (parameter_set->fe.dual.time_type_support_points.compare("Gauss")==0) {
		//   get u_h(t0) from:    Omega_h^primal x I_{n}            => u->x[0]
		//   (1) (1) evaluate primal solution u->x[0] at t0 (u(t0)) => u0_on_primal
		//   (2) interpolated to: Omega_h^dual x I_{n} (t0)   => dual.u0
		
		// (1) evaluate primal solution u->x[0] at t0 (u(t0))
		primal_get_u_t_on_slab(slab, u, t0, u0_on_primal);
		
		// (2) interpolate primal -> dual:
		//     - needs the same tria: dof1.get_tria() == dof2.get_tria()
		//     - allow different FE-spaces: dof1.get_fe() != dof2.get_fe()
		dealii::FETools::interpolate(
			// primal solution
			*slab->primal.dof,
			*u0_on_primal,
			// dual solution
			*slab->dual.dof,
			*slab->dual.constraints,
			*dual.u0
		);
		
		u0_on_primal = nullptr;
	}
	
	// init vector and run assemble J(v)(e) = (v,e)
	DTM::pout << "dwr-heat: assemble Je0...";
	dual.Je0 = std::make_shared< dealii::Vector<double> > ();
	dual.Je0->reinit( slab->dual.dof->n_dofs() );
	
	*dual.Je0 = 0;
	assemble_Je->assemble(
		dual.Je0,
		t0,
		function.u_E,
		dual.u0,
		0,   // n_q_points: 0 -> q+1 in auto mode
		true // auto mode
	);
	dual.u0 = nullptr;
	
	*dual.Je0 *= 1./primal_L2_L2_error_u;
	DTM::pout << " (done)" << std::endl;
	
	if (parameter_set->fe.dual.time_type_support_points.compare("Gauss-Lobatto")==0) {
		///////////////////////////////
		// u_h(t1) = u_h(t1)|_{I_{n}}
		//
		
		// interpolate primal solution u_h(t1) to dual solution space
		dual.u1 = std::make_shared< dealii::Vector<double> > ();
		dual.u1->reinit( slab->dual.dof->n_dofs() );
		
		dealii::FETools::interpolate(
			*slab->primal.dof,
			*u->x[0],
			*slab->dual.dof,
			*slab->dual.constraints,
			*dual.u1
		);
		
		// init vector and run assemble J(v)(e) = ((v,e))
		DTM::pout << "dwr-heat: assemble Je1...";
		dual.Je1 = std::make_shared< dealii::Vector<double> > ();
		dual.Je1->reinit( slab->dual.dof->n_dofs() );
		
		*dual.Je1 = 0;
		assemble_Je->assemble(
			dual.Je1,
			t1,
			function.u_E,
			dual.u1,
			0,   // n_q_points: 0 -> q+1 in auto mode
			true // auto mode
		);
		dual.u1 = nullptr;
		
		*dual.Je1 *= 1./primal_L2_L2_error_u;
		DTM::pout << " (done)" << std::endl;
	}
	////////////////////////////////////////////////////////////////////////
	// NOTE: for parameter_set->fe.dual.time_type_support_points.compare("Gauss")
	//       we have \f$ \beta_{1,1} = 0 \f$ and therefore we do not need
	//       to assemble anything for Je^1!
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
dual_solve_slab_problem(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<2>::iterator &z
) {
	////////////////////////////////////////////////////////////////////////////
	// construct system matrix K
	//
	
	dual.K = std::make_shared< dealii::SparseMatrix<double> > ();
	dual.K->reinit(*slab->dual.sp);
	
	*dual.K = 0;
	
	if (parameter_set->fe.dual.time_type_support_points.compare("Gauss-Lobatto")==0) {
		// construct cG(1)-Q_GL(2) system matrix K = M + tau/2 A
		DTM::pout << "dwr-heat: construct system matrix K = M + tau/2 A...";
		dual.K->add(slab->tau_n()/2., *dual.A);
		dual.K->add(1., *dual.M);
	}
	else if (parameter_set->fe.dual.time_type_support_points.compare("Gauss")==0) {
		// construct cG(1)-Q_G(2) system matrix K = 2 M + tau A
		DTM::pout << "dwr-heat: construct system matrix K = 2 M + tau A...";
		dual.K->add(slab->tau_n(),*dual.A);
		dual.K->add(2., *dual.M);
	}
	else {
		AssertThrow(false, dealii::ExcNotImplemented());
	}
	
	DTM::pout << " (done)" << std::endl;
	
	////////////////////////////////////////////////////////////////////////////
	// construct system right hand side vector b
	//
	
	if (parameter_set->fe.dual.time_type_support_points.compare("Gauss-Lobatto")==0) {
		////////////////////////////////////////////////////////////////////////
		// construct vector b = tau_n/2. * ( Je^0 + Je^1 ) + (M - tau_n/2 A) z^1
		//
		
		DTM::pout
			<< "dwr-heat: construct linear system rhs vector "
			<< "b = (M - tau/2 A) z^1 + tau/2 * ( Je^0 + Je^1 ) ...";
		
		dual.b = std::make_shared< dealii::Vector<double> > ();
		dual.b->reinit( slab->dual.dof->n_dofs() );
		
		dual.A->vmult(*dual.b, *z->x[1]);
		*dual.b *= -slab->tau_n()/2.;
		
		dual.M->vmult_add(*dual.b, *z->x[1]);
		
		dual.b->add(slab->tau_n()/2., *dual.Je0 );
		dual.b->add(slab->tau_n()/2., *dual.Je1 );
		
		DTM::pout << " (done)" << std::endl;
	}
	else if (parameter_set->fe.dual.time_type_support_points.compare("Gauss")==0) {
		////////////////////////////////////////////////////////////////////////
		// construct vector b = (tau_n * Je^0) + (0 * Je^1)  + (M - 0 * A) z^1
		//
		
		DTM::pout
			<< "dwr-heat: construct linear system rhs vector "
			<< "b = (M - 0* A) z^1 + (tau * Je^0) + (0 * Je^1) ...";
		
		dual.b = std::make_shared< dealii::Vector<double> > ();
		dual.b->reinit( slab->dual.dof->n_dofs() );
		
		// add 2 M z^1
		dual.M->vmult(*dual.b, *z->x[1]);
		*dual.b *= 2.0;
		dual.b->add(slab->tau_n(), *dual.Je0 );
		
		DTM::pout << " (done)" << std::endl;
	}
	else {
		AssertThrow(false, dealii::ExcNotImplemented());
	}
	
	////////////////////////////////////////////////////////////////////////////
	// apply homog. Dirichlet boundary and homg. Neumann boundary condition on
	// respective parts of the boundary
	//
	
	DTM::pout << "dwr-heat: dealii::MatrixTools::apply_boundary_values...";
	std::map<dealii::types::global_dof_index, double> boundary_values;
	
	dealii::VectorTools::interpolate_boundary_values(
		*slab->dual.dof,
		static_cast< dealii::types::boundary_id > (
			heat::types::boundary_id::Dirichlet
		),
		dealii::ZeroFunction<dim>(1),
		boundary_values
	);
	
	dealii::MatrixTools::apply_boundary_values(
		boundary_values,
		*dual.K,
		*z->x[0],
		*dual.b
	);
	
	DTM::pout << " (done)" << std::endl;
	
	////////////////////////////////////////////////////////////////////////////
	// condense hanging nodes in system matrix, if any
	//
	
	DTM::pout << "dwr-heat: slab->dual.constraints->condense(*dual.K)...";
	slab->dual.constraints->condense(*dual.K);
	
	DTM::pout << " (done)" << std::endl;
	
	////////////////////////////////////////////////////////////////////////////
	// solve linear system directly
	//
	
	DTM::pout << "dwr-heat: setup direct lss and solve...";
	
	dealii::SparseDirectUMFPACK iA;
	iA.initialize(*dual.K);
	iA.vmult(*z->x[0], *dual.b);
	
	DTM::pout << " (done)" << std::endl;
	
	////////////////////////////////////////////////////////////////////////////
	// distribute hanging nodes constraints on solution
	//
	
	DTM::pout << "dwr-heat: dual.constraints->distribute...";
	slab->dual.constraints->distribute(
		*z->x[0]
	);
	DTM::pout << " (done)" << std::endl;
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
dual_do_backward_TMS(
	const unsigned int dwr_loop) {
	////////////////////////////////////////////////////////////////////////////
	// prepare TMS loop
	//
	
	////////////////////////////////////////////////////////////////////////////
	// grid: init slab iterator to last space-time slab: Omega x I_N
	//
	
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	Assert(grid->slabs.size(), dealii::ExcNotInitialized());
	auto slab = std::prev(grid->slabs.end());
	
	////////////////////////////////////////////////////////////////////////////
	// storage: init iterators to storage_data_vectors
	//          corresponding to last space-time slab: Omega x I_N
	//
	
	Assert(dual.storage.z.use_count(), dealii::ExcNotInitialized());
	Assert(dual.storage.z->size(), dealii::ExcNotInitialized());
	auto z = std::prev(dual.storage.z->end());
	
	Assert(primal.storage.u.use_count(), dealii::ExcNotInitialized());
	Assert(primal.storage.u->size(), dealii::ExcNotInitialized());
	auto u = std::prev(primal.storage.u->end());
	
	////////////////////////////////////////////////////////////////////////////
	// final condition z_kh(T)
	//
	
	// NOTE: for goal functional || u - u_kh ||_L2(L2) -> z(T) = 0
	*z->x[1] = 0;
	
	////////////////////////////////////////////////////////////////////////////
	// do TMS loop
	//
	
	DTM::pout
		<< std::endl
		<< "*******************************************************************"
		<< "*************" << std::endl
		<< "dual: solving backward TMS problem..." << std::endl
		<< std::endl;
	
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};
	unsigned int n{N};
	
	while (n) {
		// local time variables: \f$ t0, t1 \in I_n = (t_m, t_n) \f$
		const double tm = slab->t_m;
		double t0{0};
		
		if (parameter_set->fe.dual.time_type_support_points.compare("Gauss-Lobatto")==0) {
			t0 = slab->t_m;
		}
		else if (parameter_set->fe.dual.time_type_support_points.compare("Gauss")==0) {
			t0 = tm + slab->tau_n()/2.;
		}
		
		const double t1 = slab->t_n;
		const double tn = slab->t_n;
		
		DTM::pout
			<< "dual: solving problem on "
			<< "I_" << n << " = (" << tm << ", " << tn << ") "
			<< std::endl;
		
		if (n < N) {
			// for 0 < n < N interpolate between two (different) spatial meshes
			// the solution z(t_m)|_{I_{n+1}}  to  z(t_n)|_{I_n}
			
			if (parameter_set->fe.dual.time_type_support_points.compare("Gauss-Lobatto")==0) {
				dealii::VectorTools::interpolate_to_different_mesh(
					// solution on I_{n+1}:
					*std::next(slab)->dual.dof,
					*std::next(z)->x[0],
					// solution on I_n:
					*slab->dual.dof,
					*slab->dual.constraints,
					*z->x[1]
				);
			}
			else if (parameter_set->fe.dual.time_type_support_points.compare("Gauss")==0) {
				dealii::VectorTools::interpolate_to_different_mesh(
					// solution on I_{n+1}:
					*std::next(slab)->dual.dof,
					*dual.zm,
					// solution on I_n:
					*slab->dual.dof,
					*slab->dual.constraints,
					*z->x[1]
				);
				dual.zm = nullptr;
			}
		} // end if (n < N)
		
		// assemble slab problem
		dual_assemble_system(slab);
		dual_assemble_rhs(slab,u,n,t0,t1);
		
		// solve slab problem (i.e. apply boundary values and solve for z0)
		dual_solve_slab_problem(slab,z);
		
		////////////////////////////////////////////////////////////////////////
		// do postprocessing on the solution
		//
		dual_do_data_output(slab,z,dwr_loop);
		
		////////////////////////////////////////////////////////////////////////
		// prepare next I_n slab problem:
		//
		
		if (parameter_set->fe.dual.time_type_support_points.compare("Gauss")==0) {
			// Evaluate dual solution z(tm) (at left time-point) and save as dual.zm,
			// which is used to be interpolated to the "next" grid at tm.
			dual.zm = std::make_shared< dealii::Vector<double> > ();
			dual.zm->reinit( slab->dual.dof->n_dofs() );
			*dual.zm = 0.;
			dual_get_z_t_on_slab(slab, z, tm, dual.zm);
		}
		
		--n;
		--slab;
		
		--u;
		--z;
		
		////////////////////////////////////////////////////////////////////////
		// allow garbage collector to clean up memory
		//
		
		dual.M = nullptr;
		dual.A = nullptr;
		
		dual.Je0 = nullptr;
		dual.Je1 = nullptr;
		
		dual.K = nullptr;
		dual.b = nullptr;
		
		DTM::pout << std::endl;
	}
	
	DTM::pout
		<< "dual: forward TMS problem done" << std::endl
		<< "*******************************************************************"
		<< "*************" << std::endl
		<< std::endl;
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
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
	double xi0{0.};
	double xi1{0.};
	if (parameter_set->fe.dual.time_type_support_points.compare("Gauss-Lobatto")==0) {
		xi0 = 1.-_t;
		xi1 = _t;
	}
	else if (parameter_set->fe.dual.time_type_support_points.compare("Gauss")==0) {
		xi0 = -2.*_t+2.;
		xi1 = 2.*_t-1.;
	}
	
	z_result->equ(xi0, *z->x[0]);
	z_result->add(xi1, *z->x[1]);
}


////////////////////////////////////////////////////////////////////////////////
// dual data output
//

template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
dual_init_data_output() {
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
	DTM::pout
		<< "dual solution data output: patches = "
		<< parameter_set->data_output.dual.patches
		<< std::endl;
	
	std::vector<std::string> data_field_names;
	data_field_names.push_back("z");
	
	std::vector< dealii::DataComponentInterpretation::DataComponentInterpretation > dci_field;
	dci_field.push_back(dealii::DataComponentInterpretation::component_is_scalar);
	
	dual.data_output = std::make_shared< DTM::DataOutput<dim> >();
	dual.data_output->set_data_field_names(data_field_names);
	dual.data_output->set_data_component_interpretation_field(dci_field);
	
	dual.data_output->set_data_output_patches(
		parameter_set->data_output.dual.patches
	);
	
	dual.data_output_dwr_loop = -3; // -2 => not initialized
	// set up which dwr loop(s) are allowed to make data output:
	if ( !parameter_set->data_output.dual.dwr_loop.compare("none") ) {
		dual.data_output_dwr_loop = -2;
	}
	else if ( !parameter_set->data_output.dual.dwr_loop.compare("all") ) {
		dual.data_output_dwr_loop = -1;
	}
	else if ( !parameter_set->data_output.dual.dwr_loop.compare("last") ) {
		Assert(parameter_set->dwr.loops > 0, dealii::ExcInternalError());
		dual.data_output_dwr_loop = parameter_set->dwr.loops-1;
	}
	else {
		dual.data_output_dwr_loop = std::stoi(parameter_set->data_output.dual.dwr_loop);
	}
	
	DTM::pout
		<< "dual solution data output: dwr loop = "
		<< dual.data_output_dwr_loop
		<< std::endl;
	
	// check if we use a fixed trigger interval, or, do output once on a I_n
	if ( !parameter_set->data_output.dual.trigger_type.compare("fixed") ) {
		dual.data_output_trigger_type_fixed = true;
	}
	else {
		dual.data_output_trigger_type_fixed = false;
	}
	
	// only for fixed
	dual.data_output_trigger = parameter_set->data_output.dual.trigger;
	
	if (dual.data_output_trigger_type_fixed) {
		DTM::pout
			<< "dual solution data output: using fixed mode with trigger = "
			<< dual.data_output_trigger
			<< std::endl;
	}
	else {
		DTM::pout
			<< "dual solution data output: using I_n mode (trigger adapts to I_n automatically)"
			<< std::endl;
	}
	
	dual.data_output_time_value = parameter_set->T;
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
dual_do_data_output(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<2>::iterator &z,
	const unsigned int dwr_loop) {
	
	if (!( (dual.data_output_dwr_loop == -1) ||
		(dual.data_output_dwr_loop == static_cast<int>(dwr_loop)) ))
		return;
	
	if (dual.data_output_trigger <= 0) return;
	
	// adapt trigger value for I_n output mode
	if (!dual.data_output_trigger_type_fixed) {
		if (slab != grid->slabs.begin()) {
			dual.data_output_trigger = std::prev(slab)->tau_n();
		}
		else {
			dual.data_output_trigger = slab->tau_n();
		}
	}
	
	dual.data_output->set_DoF_data(
		slab->dual.dof
	);
	
	auto z_trigger = std::make_shared< dealii::Vector<double> > ();
	z_trigger->reinit(
		slab->dual.dof->n_dofs()
	);
	
	std::ostringstream filename;
	filename
		<< "dual-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop+1;
	
	double &t{dual.data_output_time_value};
	
	for ( ; t >= slab->t_m; t -= dual.data_output_trigger) {
		const double _t{ (t - slab->t_m) / slab->tau_n() };
		
		const double xi0{ 1. - _t };
		const double xi1{ _t };
		
		// evalute space-time solution
		z_trigger->equ(xi0, *z->x[0]);
		z_trigger->add(xi1, *z->x[1]);
		
		dual.data_output->write_data(
			filename.str(),
			z_trigger,
			t
		);
	}
	
	// check if data for t=0 (t_0) was written
	if (slab == grid->slabs.begin()) {
	if (dual.data_output_trigger_type_fixed) {
		if ( t < slab->t_m ) {
			// undershoot of time variable; manually set t = 0 and do data output
			t = slab->t_m;
			
			const double _t{ (t - slab->t_m) / slab->tau_n() };
		
			const double xi0{ 1. - _t };
			const double xi1{ _t };
			
			// evalute space-time solution
			z_trigger->equ(xi0, *z->x[0]);
			z_trigger->add(xi1, *z->x[1]);
			
			dual.data_output->write_data(
				filename.str(),
				z_trigger,
				t
			);
		}
	}}
}


////////////////////////////////////////////////////////////////////////////////
// error estimation
//

template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
eta_reinit_storage() {
	////////////////////////////////////////////////////////////////////////////
	// init storage containers for vector data
	//
	
	////////////////////////////////////////////////////////////////////////////
	// get number of time steps
	//
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};
	
	////////////////////////////////////////////////////////////////////////////
	// error indicators vector eta = sum of eta_K
	//
	
	error_estimator.storage.eta = std::make_shared< DTM::types::storage_data_vectors<1> > ();
	error_estimator.storage.eta->resize(N);
	
	{
		auto slab = grid->slabs.begin();
		for (auto &element : *error_estimator.storage.eta) {
			for (unsigned int j{0}; j < element.x.size(); ++j) {
				// create shared_ptr to Vector<double>
				element.x[j] = std::make_shared< dealii::Vector<double> > ();
				
				// init. Vector<double> with n_dofs components
				Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
				Assert(slab->tria.use_count(), dealii::ExcNotInitialized());
				Assert(
					slab->tria->n_global_active_cells(),
					dealii::ExcMessage("Error: slab->tria->n_global_active_cells() == 0")
				);
				
				element.x[j]->reinit(
					slab->tria->n_global_active_cells()
				);
			}
			++slab;
		}
	}
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
compute_error_indicators() {
	error_estimator.dwr =
		std::make_shared< heat::dwr::cGp_dG0::cGq_cG1::ErrorEstimator<dim> > ();
	
	error_estimator.dwr->estimate(
		function.density,
		function.epsilon,
		function.f,
		function.u_D,
		function.u_0,
		grid,
		parameter_set,
		primal.storage.u,
		dual.storage.z,
		error_estimator.storage.eta
	);
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
compute_effectivity_index() {
	double eta{0.};
	for ( const auto &eta_it : *error_estimator.storage.eta ) {
		eta += std::accumulate(eta_it.x[0]->begin(), eta_it.x[0]->end(), 0.);
	}
	
	DTM::pout << "eta = " << eta << std::endl;
	DTM::pout << "primal_L2_L2_error_u = " << primal_L2_L2_error_u << std::endl;
	
	const double I_eff{std::abs(eta/primal_L2_L2_error_u)};
	DTM::pout << "I_eff = " << I_eff << std::endl;
	
	unsigned int K_max{0};
	unsigned int slabs_size{0};
	slabs_size = grid->slabs.size();
	auto slab{grid->slabs.begin()};
	auto ends{grid->slabs.end()};
	for (; slab != ends; ++slab) {
		K_max = (K_max > slab->tria->n_global_active_cells()) ? K_max : slab->tria->n_global_active_cells();
	}
	// convergence_table
	convergence_table.add_value("N_max", slabs_size);
	convergence_table.add_value("K_max", K_max);
	convergence_table.add_value("primal_L2_L2_error_u", primal_L2_L2_error_u);
	convergence_table.add_value("eta", eta);
	convergence_table.add_value("I_eff", I_eff);
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
refine_and_coarsen_space_time_grid() {
	// set eta_K = | eta_K |
	for (auto &eta_In : *error_estimator.storage.eta) {
		for (auto &eta_K : *eta_In.x[0] ) {
			eta_K = std::abs(eta_K);
			Assert(eta_K >= 0., dealii::ExcInternalError());
		}
		
#ifdef DEBUG
		for (auto &eta_K : *eta_In.x[0] ) {
			Assert(eta_K >= 0., dealii::ExcInternalError());
		}
#endif
	}
	
	Assert(
		error_estimator.storage.eta->size()==grid->slabs.size(),
		dealii::ExcInternalError()
	);
	
	const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};
	std::vector<double> eta(N);
	
	// 1st loop: compute eta^n on I_n for n=1..N
	{
		auto eta_it{error_estimator.storage.eta->begin()};
		for (unsigned n{0}; n < N; ++n, ++eta_it) {
			Assert(
				(eta_it != error_estimator.storage.eta->end()),
				dealii::ExcInternalError()
			);
			
			eta[n] = std::accumulate(
				eta_it->x[0]->begin(),
				eta_it->x[0]->end(),
				0.
			);
		}
	}
	
	// 2nd loop: mark for time refinement
	{
		if (parameter_set->dwr.refine_and_coarsen.time.strategy.compare(
			"global") == 0) {
			// global refinement in time (marks all I_n for refinement)
			auto slab{grid->slabs.begin()};
			auto ends{grid->slabs.end()};
			for (unsigned int n{0} ; slab != ends; ++slab, ++n) {
				Assert((n < N), dealii::ExcInternalError());
				slab->set_refine_in_time_flag();
			}
		}
		else if (parameter_set->dwr.refine_and_coarsen.time.strategy.compare(
			"fixed_fraction") == 0) {
			Assert(
				((parameter_set->dwr.refine_and_coarsen.time.top_fraction >= 0.) &&
				(parameter_set->dwr.refine_and_coarsen.time.top_fraction <= 1.)),
				dealii::ExcMessage(
					"parameter_set->dwr.refine_and_coarsen.time.top_fraction "
					"must be in [0,1]"
				)
			);
			
			if (parameter_set->dwr.refine_and_coarsen.time.top_fraction > 0.) {
				std::vector<double> eta_sorted(eta);
				std::sort(eta_sorted.begin(), eta_sorted.end());
				
				// check if index for eta_criterium_for_mark_time_refinement is valid
				Assert(
					( static_cast<int>(N)
					- static_cast<int>(std::floor(static_cast<double>(N)
						* parameter_set->dwr.refine_and_coarsen.time.top_fraction)) ) >= 0,
					dealii::ExcInternalError()
				);
				
				unsigned int index_for_mark_time_refinement {
					static_cast<unsigned int> (
						static_cast<int>(N)
						- static_cast<int>(std::floor(
							static_cast<double>(N)
							* parameter_set->dwr.refine_and_coarsen.time.top_fraction
						))
					)
				};
				
				const auto eta_criterium_for_mark_time_refinement{
					eta_sorted[ index_for_mark_time_refinement < N ?
						index_for_mark_time_refinement : N-1 ]
				};
				
				auto slab{grid->slabs.begin()};
				auto ends{grid->slabs.end()};
				for (unsigned int n{0} ; slab != ends; ++slab, ++n) {
					Assert((n < N), dealii::ExcInternalError());
					
					if (eta[n] >= eta_criterium_for_mark_time_refinement) {
						slab->set_refine_in_time_flag();
					}
				}
			}
		}
		else {
			AssertThrow(
				false,
				dealii::ExcMessage(
					"parameter_set->dwr.refine_and_coarsen.time.strategy unknown"
				)
			);
		}
	}
	
	// 3rd loop execute_coarsening_and_refinement
	{
		unsigned int K_max{0};
		auto slab{grid->slabs.begin()};
		auto ends{grid->slabs.end()};
		auto eta_it{error_estimator.storage.eta->begin()};
		for (unsigned int n{0} ; slab != ends; ++slab, ++eta_it, ++n) {
			Assert((n < N), dealii::ExcInternalError());
			
			Assert(
				(eta_it != error_estimator.storage.eta->end()),
				dealii::ExcInternalError()
			);
			
			DTM::pout << "\tn = " << n << std::endl;
			
			const auto n_active_cells_on_slab{slab->tria->n_global_active_cells()};
			DTM::pout << "\t#K = " << n_active_cells_on_slab << std::endl;
			K_max = (K_max > n_active_cells_on_slab) ? K_max : n_active_cells_on_slab;
			
			if (parameter_set->dwr.refine_and_coarsen.space.strategy.compare("global") == 0) {
				// global refinement in space
				slab->tria->refine_global(1);
			}
			else if (parameter_set->dwr.refine_and_coarsen.space.strategy.compare("fixed_fraction") == 0) {
				// mark for refinement with fixed fraction
				// (similar but not identical to Hartmann Ex. Sec. 1.4.2)
				const double top_fraction{ slab->refine_in_time ?
					parameter_set->dwr.refine_and_coarsen.space.top_fraction1 :
					parameter_set->dwr.refine_and_coarsen.space.top_fraction2
				};
				dealii::GridRefinement::refine_and_coarsen_fixed_fraction(
					*slab->tria,
					*eta_it->x[0],
					top_fraction,
					parameter_set->dwr.refine_and_coarsen.space.bottom_fraction,
					slab->tria->n_global_active_cells() *
						parameter_set->dwr.refine_and_coarsen.space.max_growth_factor_n_active_cells
				);
				
				// execute refinement in space under the conditions of mesh smoothing
				slab->tria->execute_coarsening_and_refinement();
			}
			else if (parameter_set->dwr.refine_and_coarsen.space.strategy.compare("Schwegler") == 0) {
				// mark for refinement with strategy from K. Schwegler PhD thesis
				
				Assert(
					((parameter_set->dwr.refine_and_coarsen.space.theta1 > 1.)
					&& (parameter_set->dwr.refine_and_coarsen.space.theta1 < 5.)),
					dealii::ExcMessage("theta1 must be in (1,5)")
				);
				
				Assert(
					((parameter_set->dwr.refine_and_coarsen.space.theta2 > 1.)
					&& (parameter_set->dwr.refine_and_coarsen.space.theta2 < 5.)),
					dealii::ExcMessage("theta2 must be in (1,5)")
				);
				
				Assert(
					(parameter_set->dwr.refine_and_coarsen.space.theta2 >=
					parameter_set->dwr.refine_and_coarsen.space.theta1),
					dealii::ExcMessage("(theta2 >= theta1)")
				);
				
				const double theta{ slab->refine_in_time ?
					parameter_set->dwr.refine_and_coarsen.space.theta1 :
					parameter_set->dwr.refine_and_coarsen.space.theta2
				};
				DTM::pout << "\ttheta = " << theta << std::endl;
				
				double mu{theta * eta[n] / n_active_cells_on_slab};
				DTM::pout << "\tmu = " << mu << std::endl;
				
				const auto eta_max{
					*std::max_element(eta_it->x[0]->begin(), eta_it->x[0]->end())
				};
				DTM::pout << "\teta_max = " << eta_max << std::endl;
				
				while (mu > eta_max) {
					mu /= 2.;
				}
				DTM::pout << "\tmu = " << mu << std::endl;
				
				auto cell{slab->tria->begin_active()};
				auto endc{slab->tria->end()};
				for ( ; cell != endc; ++cell) {
					if ( (*eta_it->x[0])[ cell->index() ] > mu ) {
						cell->set_refine_flag(
							dealii::RefinementCase<dim>::isotropic_refinement
						);
					}
				}
				
				// execute refinement in space under the conditions of mesh smoothing
				slab->tria->execute_coarsening_and_refinement();
			}
			else {
				AssertThrow(
					false,
					dealii::ExcMessage(
						"parameter_set->dwr.refine_and_coarsen.space.strategy unknown"
					)
				);
			}
			
			// refine in time
			if (slab->refine_in_time) {
				grid->refine_slab_in_time(slab);
				slab->refine_in_time = false;
			}
		}
		
		DTM::pout << "\t#Kmax (before refinement) = " << K_max << std::endl;
	}
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
write_convergence_table_to_tex_file() {
// 	convergence_table.set_precision("DWR-loop", 5);
	convergence_table.set_precision("primal_L2_L2_error_u", 5);
	convergence_table.set_precision("eta", 5);
	convergence_table.set_precision("I_eff", 3);
	
	convergence_table.set_scientific("primal_L2_L2_error_u", true);
	convergence_table.set_scientific("eta", true);
	
	std::cout << std::endl;
	convergence_table.write_text(std::cout);
	
	// Set tex captions and formation of respective columns
	convergence_table.set_tex_caption("DWR-loop","DWR-loop");
	convergence_table.set_tex_caption("N_max","$N_{\\text{max}}$");
	convergence_table.set_tex_caption("K_max","$K_{\\text{max}}$");
	convergence_table.set_tex_caption("primal_L2_L2_error_u","$\\|e\\|_{(0,T)\\times\\Omega}$");
	convergence_table.set_tex_caption("eta","$\\eta$");
	convergence_table.set_tex_caption("I_eff","I$_{\\text{eff}}$");
	convergence_table.set_tex_format("DWR-loop","c");
	convergence_table.set_tex_format("N_max","r");
	convergence_table.set_tex_format("K_max","r");
	convergence_table.set_tex_format("primal_L2_L2_error_u","c");
	convergence_table.set_tex_format("eta","c");
	convergence_table.set_tex_format("I_eff","c");
	
	std::vector<std::string> new_order;
	new_order.push_back("DWR-loop");
	new_order.push_back("N_max");
	new_order.push_back("K_max");
	new_order.push_back("primal_L2_L2_error_u");
	new_order.push_back("eta");
	new_order.push_back("I_eff");
	convergence_table.set_column_order (new_order);
		
	convergence_table.evaluate_convergence_rates("primal_L2_L2_error_u",dealii::ConvergenceTable::reduction_rate);
	convergence_table.evaluate_convergence_rates("primal_L2_L2_error_u",dealii::ConvergenceTable::reduction_rate_log2);

	
	// write TeX/LaTeX file of the convergence table with deal.II
	{
		std::string filename = "convergence-table.tex";
		std::ofstream out(filename.c_str());
		convergence_table.write_tex(out);
	}
	
	// read/write TeX/LaTeX file to make pdflatex *.tex working
	{
		std::ifstream in("convergence-table.tex");
		
		std::string filename = "my-convergence-table.tex";
		std::ofstream out(filename.c_str());
		
		std::string line;
		std::getline(in, line);
		out << line << std::endl;
		out << "\\usepackage{amsmath}" << std::endl;
		
		for ( ; std::getline(in, line) ; )
			out << line << std::endl;
		
		out.close();
	}
}

} // namespace

#include "Heat_DWR__cGp_dG0__cGq_cG1.inst.in"
