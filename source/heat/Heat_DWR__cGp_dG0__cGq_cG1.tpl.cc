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

// #include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/fe/fe_tools.h>

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
	
	
	// primal problem:
	primal_reinit_storage();
	primal_init_data_output();
	primal_do_forward_TMS();
	
	// dual problem
	dual_reinit_storage();
	dual_init_data_output();
	dual_do_backward_TMS();
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
	
	// exact solution (if any)
	function.u_E = std::make_shared< heat::ExactSolution::Test0<dim> > ();
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
	const typename DTM::types::storage_data_vectors<1>::iterator &u
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
	// solve linear system directly
	//
	
	DTM::pout << "dwr-heat: setup direct lss and solve...";
	
	dealii::SparseDirectUMFPACK iA;
	iA.initialize(*primal.K);
	iA.vmult(*u->x[0], *primal.b);
	
	DTM::pout << " (done)" << std::endl;
	
	////////////////////////////////////////////////////////////////////////////
	// distribute hanging node constraints on solution
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
	primal_do_data_output(slab,primal.um,slab->t_m);
	
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
		primal_solve_slab_problem(slab,u);
		
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
		primal_do_data_output(slab,primal.un,tn);
		
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
}


////////////////////////////////////////////////////////////////////////////////
// primal data output
//

template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
primal_init_data_output() {
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
	std::shared_ptr< dealii::Vector<double> > u_trigger,
	const double &t_trigger) {
	
	primal.data_output.set_DoF_data(
		slab->primal.dof
	);
	
	// TODO: construct solution at t_n
	// NOTE: dG(0) is constant in time, thus we can simply output u->x[0]
	
	//const double zeta0 = 1.0;
	//u_trigger.add(zeta0, *u->x[0])
	
	// TODO
	primal.data_output.write_data(
		"solution",
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
		// n == 1: interpolate initial value function u_0 to dual space
		function.u_0->set_time(t0);
		dealii::VectorTools::interpolate(
			*slab->dual.mapping,
			*slab->dual.dof,
			*function.u_0,
			*dual.u0
		);
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
	// construct system matrix K = M + tau A
	//
	
	DTM::pout << "dwr-heat: construct system matrix K = M + tau A...";
	
	dual.K = std::make_shared< dealii::SparseMatrix<double> > ();
	dual.K->reinit(*slab->dual.sp);
	
	*dual.K = 0;
	dual.K->add(slab->tau_n(), *dual.A);
	dual.K->add(1.0, *dual.M);
	
	DTM::pout << " (done)" << std::endl;
	
	
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
	// solve linear system directly
	//
	
	DTM::pout << "dwr-heat: setup direct lss and solve...";
	
	dealii::SparseDirectUMFPACK iA;
	iA.initialize(*dual.K);
	iA.vmult(*z->x[0], *dual.b);
	
	DTM::pout << " (done)" << std::endl;
	
	////////////////////////////////////////////////////////////////////////////
	// distribute hanging node constraints on solution
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
dual_do_backward_TMS() {
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
	dual_do_data_output(slab,z->x[1],slab->t_n);
	
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
		dual_do_data_output(slab,z->x[0],t0);
		
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


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
dual_get_z_t_on_slab_after_restriction_to_primal_space(
	std::shared_ptr< dealii::Vector<double> > &z_result,
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<2>::iterator &z,
	[[maybe_unused]]const double t
) {
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


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
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


////////////////////////////////////////////////////////////////////////////////
// dual data output
//

template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
dual_init_data_output() {
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
	// TODO:
	DTM::pout
		<< "dual solution data output: patches = "
		<< parameter_set->fe.q // auto mode = cG in space: take q patches per K
// 		<< parameter_set->data_output.patches
		<< std::endl;
	
	std::vector<std::string> data_field_names;
	data_field_names.push_back("z");
	
	std::vector< dealii::DataComponentInterpretation::DataComponentInterpretation > dci_field;
	dci_field.push_back(dealii::DataComponentInterpretation::component_is_scalar);
	
	dual.data_output.set_data_field_names(data_field_names);
	dual.data_output.set_data_component_interpretation_field(dci_field);
	
	// TODO:
	dual.data_output.set_data_output_patches(
		parameter_set->fe.q // auto mode = cG in space: take q patches per K
// 		parameter_set->data_output.patches
	);
}


template<int dim>
void
Heat_DWR__cGp_dG0__cGq_cG1<dim>::
dual_do_data_output(
	const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
	std::shared_ptr< dealii::Vector<double> > z_trigger,
	const double &t_trigger) {
	
	dual.data_output.set_DoF_data(
		slab->dual.dof
	);
	
	// TODO: construct solution at t_trigger
	
	//const double xi0 = ...; (xi0(t_trigger))
	//const double xi1 = ...; (xi1(t_trigger))
	//u_trigger.add(xi0, *z->x[0])
	//u_trigger.add(xi1, *z->x[1])
	
	// TODO
	dual.data_output.write_data(
		"dual",
		z_trigger,
		t_trigger
	);
}


////////////////////////////////////////////////////////////////////////////////
// error estimation
//


// // TODO NOTE TEST remove the following:
// template<int dim>
// void
// Heat_DWR__cGp_dG0__cGq_cG1<dim>::
// solve_dual_problem() {

// // NOTE (UK): the following comment is misleading: t_N = 0.5??
// 	// Compute dual.Je at timepoint t_N = 0.5 (within dual_assemble_je())
// 	// and store it in dual.Je_old.
// 	dual_assemble_Je_L2global();
// 	
// 	for (unsigned int n = ((data.T-data.t0)/data.tau_n); n >= 0 ; --n) {
// 			// Compute z_N-1 at time point t_N-1
// 			
// 			dual_assemble_system();

// 			volatile const double ttt{n*data.tau_n};
// 			dual_set_time(ttt);
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
// dual_assemble_rhs_at_t_Nminus1() {
// 	switch (dual.Je_type) {
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
// }


// template<int dim>
// void
// Heat_DWR__cGp_dG0__cGq_cG1<dim>::
// dual_assemble_rhs() {
// 	switch (dual.Je_type) {

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
// 	
// 	for ( ; cell != endc; ++cell) {
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





////////////////////////////////////////////////////////////////////////////////
// old


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
// 		/////////////// begin primal ///////////////////////////////////////////
// 		// solve complete forward TMS
// 		solve_primal_problem();
// 		
// 		///////////////////////// begin dual ///////////////////////////////////
// 		solve_dual_problem();
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
