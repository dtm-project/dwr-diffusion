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

#include <heat/types/boundary_id.hh>
#include <heat/Force/Forces.hh>
#include <heat/ExactSolution/ExactSolutions.hh>

#include <heat/Heat_DWR__cGp_dG0__cGq_cG1.tpl.hh>

#include <heat/assembler/L2_MassAssembly.tpl.hh>
#include <heat/assembler/L2_LaplaceAssembly.tpl.hh>

#include <heat/assembler/L2_ForceConstrainedAssembly.tpl.hh>
template <int dim>
using ForceAssembler = heat::Assemble::L2::ForceConstrained::Assembler<dim>;

#include <heat/assembler/L2_Je_global_L2L2_Assembly.tpl.hh>

// DEAL.II includes
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/fe/fe_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

// C++ includes
#include <algorithm>

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
	dwr_loops = 6; // TODO: input parameter
	for (unsigned int dwr_loop{0}; dwr_loop < dwr_loops; ++dwr_loop) {
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
		
		// if another dwr loop do:
		refine_and_coarsen_space_time_grid();
	}
}


////////////////////////////////////////////////////////////////////////////////
// internal functions
//

template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
init_functions() {
	// TODO: read those from parameter input file
	
	// Hartmann Sec. 1.4.2 Test problem:
	const double a{50.};
	function.u_D = std::make_shared< heat::ExactSolution::Hartmann142<dim> > (a);
	function.u_0 = std::make_shared< heat::ExactSolution::Hartmann142<dim> > (a);
	
	function.epsilon = std::make_shared< dealii::Functions::ConstantFunction<dim> > (1.0);
	function.density = std::make_shared< dealii::Functions::ConstantFunction<dim> > (1.0);
	function.f = std::make_shared< heat::force::Hartmann142<dim> > (a,function.epsilon);
	
	// exact solution (if any)
	function.u_E = std::make_shared< heat::ExactSolution::Hartmann142<dim> > (a);
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


////////////////////////////////////////////////////////////////////////////////
// primal problem
//

template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
primal_reinit_storage() {
	////////////////////////////////////////////////////////////////////////////
	// init storage containers for vector data
	//
	
	////////////////////////////////////////////////////////////////////////////
	// get number of time steps
	//
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};
	
	////////////////////////////////////////////////////////////////////////////
	// primal space: time dG(0) method ( here: dG(0)-Q_{Gauss(1)} )
	//
	
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
	
	// construct vector b = M um + tau_n f0
	DTM::pout << "dwr-heat: construct linear system rhs vector...";
	
	primal.b = std::make_shared< dealii::Vector<double> > ();
	primal.b->reinit( slab->primal.dof->n_dofs() );
	
	Assert(primal.M.use_count(), dealii::ExcNotInitialized());
	Assert(primal.um.use_count(), dealii::ExcNotInitialized());
	primal.M->vmult(*primal.b, *primal.um);
	
	primal.b->add(slab->tau_n(), *primal.f0);
	
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
	// solve linear system directly
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
	
	// output "initial value solution" at initial time t0
	primal_do_data_output(slab,primal.um,slab->t_m,dwr_loop);
	
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
		const double t0 = tm + slab->tau_n()/2.;
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
		double zeta0 = 1.0; // zeta0( t_n ) = 1.0 for dG(0)
		*primal.un = 0;
		primal.un->add(zeta0, *u->x[0]);
		
		// output solution at t_n
		primal_do_data_output(slab,primal.un,tn,dwr_loop);
		
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
		slab->primal.fe->tensor_degree()+2
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
		primal.data_output_dwr_loop = dwr_loops;
	}
	else {
		primal.data_output_dwr_loop = std::stoi(parameter_set->data_output.primal.dwr_loop);
	}
	
	DTM::pout
		<< "primal solution data output: dwr loop = "
		<< primal.data_output_dwr_loop
		<< std::endl;
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
primal_do_data_output(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
	std::shared_ptr< dealii::Vector<double> > u_trigger,
	const double &t_trigger,
	const unsigned int dwr_loop) {
	
	if (!( (primal.data_output_dwr_loop == -1) ||
		(primal.data_output_dwr_loop == static_cast<int>(dwr_loop)) ))
		return;
	
	primal.data_output->set_DoF_data(
		slab->primal.dof
	);
	
	// TODO: construct solution at t_n
	// NOTE: dG(0) is constant in time, thus we can simply output u->x[0]
	
	//const double zeta0 = 1.0;
	//u_trigger.add(zeta0, *u->x[0])
	
	// TODO
	
	std::ostringstream filename;
	filename << "solution-dwr_loop-" << dwr_loop;
	
	primal.data_output->write_data(
		filename.str(),
		u_trigger,
		t_trigger
	);
}


////////////////////////////////////////////////////////////////////////////////
// dual problem
//

template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
dual_reinit_storage() {
	////////////////////////////////////////////////////////////////////////////
	// init storage containers for vector data
	//
	
	////////////////////////////////////////////////////////////////////////////
	// get number of time steps
	//
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};
	
	////////////////////////////////////////////////////////////////////////////
	// dual space: time cG(1) method ( here: cG(1)-Q_{Gauss-Lobatto(2)} )
	//
	
	////////////////////////////////////////////////////////////////////////////
	// dual solution dof vectors z (on dual solution space)
	//
	
	dual.storage.z = std::make_shared< DTM::types::storage_data_vectors<2> > ();
	dual.storage.z->resize(N);
	
	{
		auto slab = grid->slabs.begin();
		for (auto &element : *dual.storage.z) {
			for (unsigned int j{0}; j < element.x.size(); ++j) {
				// create shared_ptr to Vector<double>
				element.x[j] = std::make_shared< dealii::Vector<double> > ();
				
				// init. Vector<double> with n_dofs components
				Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
				Assert(slab->dual.dof.use_count(), dealii::ExcNotInitialized());
				Assert(
					slab->dual.dof->n_dofs(),
					dealii::ExcMessage("Error: slab->dual.dof->n_dofs() == 0")
				);
				
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
	
	// construct cG(1)-Q_GL(2) system matrix K = M + tau/2 A
	DTM::pout << "dwr-heat: construct system matrix K = M + tau/2 A...";
	
	dual.K = std::make_shared< dealii::SparseMatrix<double> > ();
	dual.K->reinit(*slab->dual.sp);
	
	*dual.K = 0;
	dual.K->add(slab->tau_n()/2., *dual.A);
	dual.K->add(1.0, *dual.M);
	
	DTM::pout << " (done)" << std::endl;
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
dual_assemble_rhs(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &u,
	const typename DTM::types::storage_data_vectors<2>::iterator &z,
	const unsigned int &n,
	const double &t0,
	const double &t1
) {
	////////////////////////////////////////////////////////////////////////////
	// TODO: this is only for global L2(L2) goal functional
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
	// NOTE: forward problem is dG(0) (constant in time and JUMPS in t_m = t0)
	// NOTE: we need the reconstruction on dual cG(1)-Q_GL(2) here
	//
	
	///////////////////////////////
	// u_h(t0) = u_h(t0)|_{I_{n-1}}
	//
	
	// interpolate primal solution u_h(t0) to dual solution space
	dual.u0 = std::make_shared< dealii::Vector<double> > ();
	dual.u0->reinit( slab->dual.dof->n_dofs() );
	
	if ( n > 1 ) {
		// n > 1:
		//   get u_h(t0) from:    Omega_h^primal x I_{n-1} (t_{n-1})
		//   (1) interpolated to: Omega_h^primal x I_{n} (t_m) => u0_on_primal
		//   (2) interpolated to: Omega_h^dual x I_{n} (t_m)   => dual.u0
		
		// (1) interpolate_to_different_mesh (in primal):
		//     - needs the same fe: dof1.get_fe() = dof2.get_fe()
		//     - allow different triangulations: dof1.get_tria() != dof2.get_tria()
		auto u0_on_primal = std::make_shared< dealii::Vector<double> > ();
		u0_on_primal->reinit( slab->primal.dof->n_dofs() );
		
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
	
	////////////////////////////////////////////////////////////////////////////
	// construct vector b = tau_n( Je^0 + Je^1 ) + (M - tau_n/2 A) z^1
	//
	
	dual.b = std::make_shared< dealii::Vector<double> > ();
	dual.b->reinit( slab->dual.dof->n_dofs() );
	
	dual.A->vmult(*dual.b, *z->x[1]);
	*dual.b *= -slab->tau_n()/2.;
	
	dual.M->vmult_add(*dual.b, *z->x[1]);
	
	dual.b->add(slab->tau_n()/2., *dual.Je0 );
	dual.b->add(slab->tau_n()/2., *dual.Je1 );
	
	DTM::pout << " (done)" << std::endl;
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
dual_solve_slab_problem(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<2>::iterator &z
) {
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
	
	DTM::pout << "dwr-heat: primal.constraints->distribute...";
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
	
	// output "final value solution" at final time T
	dual_do_data_output(slab,z->x[1],slab->t_n,dwr_loop);
	
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
		const double t0 = slab->t_m;
		const double t1 = slab->t_n;
		const double tn = slab->t_n;
		
		DTM::pout
			<< "dual: solving problem on "
			<< "I_" << n << " = (" << tm << ", " << tn << ") "
			<< std::endl;
		
		if (n < N) {
			// for 0 < n < N interpolate between two (different) spatial meshes
			// the solution z(t_m)|_{I_{n+1}}  to  z(t_n)|_{I_n}
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
		
		// assemble slab problem
		dual_assemble_system(slab);
		dual_assemble_rhs(slab,u,z,n,t0,t1);
		
		// solve slab problem (i.e. apply boundary values and solve for z0)
		dual_solve_slab_problem(slab,z);
		
		////////////////////////////////////////////////////////////////////////
		// do postprocessing on the solution
		//
		
		// output "final value solution" at final time "t_m=t0"
		dual_do_data_output(slab,z->x[0],t0,dwr_loop);
		
		////////////////////////////////////////////////////////////////////////
		// prepare next I_n slab problem:
		//
		
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
	const double xi0{ 1. - _t };
	const double xi1{ _t };
	
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
		dual.data_output_dwr_loop = dwr_loops;
	}
	else {
		dual.data_output_dwr_loop = std::stoi(parameter_set->data_output.dual.dwr_loop);
	}
	
	DTM::pout
		<< "dual solution data output: dwr loop = "
		<< dual.data_output_dwr_loop
		<< std::endl;
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
dual_do_data_output(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
	std::shared_ptr< dealii::Vector<double> > z_trigger,
	const double &t_trigger,
	const unsigned int dwr_loop) {
	
	if (!( (dual.data_output_dwr_loop == -1) ||
		(dual.data_output_dwr_loop == static_cast<int>(dwr_loop)) ))
		return;
	
	dual.data_output->set_DoF_data(
		slab->dual.dof
	);
	
	// TODO: construct solution at t_trigger
	
	//const double xi0 = ...; (xi0(t_trigger))
	//const double xi1 = ...; (xi1(t_trigger))
	//u_trigger.add(xi0, *z->x[0])
	//u_trigger.add(xi1, *z->x[1])
	
	std::ostringstream filename;
	filename << "dual-dwr_loop-" << dwr_loop;
	
	dual.data_output->write_data(
		filename.str(),
		z_trigger,
		t_trigger
	);
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
					slab->tria->n_active_cells(),
					dealii::ExcMessage("Error: slab->tria->n_active_cells() == 0")
				);
				
				element.x[j]->reinit(
					slab->tria->n_active_cells()
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
		function.epsilon,
		function.f,
		function.u_D,
		function.u_0,
		grid,
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
	
	////////////////////////////////////////////////////////////////////////////
	// Schwegeler space-time refinement strategy
	//
	const double theta_t{0.5}; // TODO: read from parameter input
	
// 	// TODO: read in from parameter input
// 	const double theta1 = 1.2;
// 	const double theta2 = 1.2;
// // 	const double theta2 = std::max(theta1*2., 4.999);
// 	
	Assert(
		((theta_t >= 0.) && (theta_t <= 1.)),
		dealii::ExcMessage("theta_t must be in [0,1]")
	);
	
// 	Assert(
// 		((theta1 > 1.) && (theta1 < 5.)),
// 		dealii::ExcMessage("theta1 must be in (1,5)")
// 	);
// 	
// 	Assert(
// 		((theta2 > 1.) && (theta2 < 5.)),
// 		dealii::ExcMessage("theta2 must be in (1,5)")
// 	);
// 	
// 	Assert(
// 		(theta2 >= theta1),
// 		dealii::ExcMessage("(theta2 >= theta1)")
// 	);
	
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
		std::vector<double> eta_sorted(eta);
		std::sort(eta_sorted.begin(), eta_sorted.end());
		
		// check if index for eta_criterium_for_mark_time_refinement is valid
		Assert(
			( static_cast<int>(N)
			- static_cast<int>(std::floor(static_cast<double>(N)*theta_t)) ) >= 0,
			dealii::ExcInternalError()
		);
		
		const auto eta_criterium_for_mark_time_refinement{
			eta_sorted[ static_cast<int>(N)
				- static_cast<int>(std::floor(static_cast<double>(N)*theta_t)) ]
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
			
			const auto n_active_cells_on_slab{slab->tria->n_active_cells()};
			DTM::pout << "\t#K = " << n_active_cells_on_slab << std::endl;
			K_max = (K_max > n_active_cells_on_slab) ? K_max : n_active_cells_on_slab;
			
// 			const double theta{ slab->refine_in_time ? theta1 : theta2 };
// 			DTM::pout << "\ttheta = " << theta << std::endl;
			
// 			double mu{theta * eta[n] / n_active_cells_on_slab};
// 			DTM::pout << "\tmu = " << mu << std::endl;
			
			const auto eta_max{
				*std::max_element(eta_it->x[0]->begin(), eta_it->x[0]->end())
			};
			DTM::pout << "\teta_max = " << eta_max << std::endl;
			
// 			while (mu > eta_max) {
// 				mu /= 2.;
// 			}
// 			DTM::pout << "\tmu = " << mu << std::endl;
			
			// mark cells in space for refinement
			
// 			// K. Schwegeler
// 			auto cell{slab->tria->begin_active()};
// 			auto endc{slab->tria->end()};
// 			for ( ; cell != endc; ++cell) {
// 				if ( (*eta_it->x[0])[ cell->index() ] > mu ) {
// 					cell->set_refine_flag(
// 						dealii::RefinementCase<dim>::isotropic_refinement
// 					);
// 				}
// 			}
			
			// mark for refinement with fixed fraction
			// (similar but not identical to Hartmann Ex. Sec. 1.4.2)
			dealii::GridRefinement::refine_and_coarsen_fixed_fraction(
				*slab->tria,
				*eta_it->x[0],
				.8, // top_fraction:    1 will refine every cell
				.0, // bottom_fraction: 0 will coarsen no cells
				slab->tria->n_active_cells()*3 // max elements restriction
			);
			
			// execute refinement in space under the conditions of mesh smoothing
			slab->tria->execute_coarsening_and_refinement();
			
			// refine in time
			if (slab->refine_in_time) {
				grid->refine_slab_in_time(slab);
				slab->refine_in_time = false;
			}
		}
		DTM::pout << "\t#Kmax (before refinement) = " << K_max << std::endl;
	}
}

} // namespace

#include "Heat_DWR__cGp_dG0__cGq_cG1.inst.in"
