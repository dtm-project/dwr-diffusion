/**
 * @file Heat_cG_DWR.tpl.cc
 * @author Marius Paul Bruchhaeuser (MPB), Uwe Koecher (UK)
 * @date 2017-08-01, Heat/DWR, MPB, UK
 *
 * @brief Heat/DWR Problem
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

#include <Heat/Heat_cG_DWR.tpl.hh>
#include <Heat/types/boundary_id.hh> 

// DEAL.II includes
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>	// only for solving with cg
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>


// C++ includes
#include <fstream>
#include <vector>
#include <algorithm>
#include <list>
#include <iterator>

#include <iomanip>

namespace Heat {

template<int dim>
void
Heat_cG_DWR<dim>::
set_error_functional_type(
	const Heat::types::error_functional &error_functional_type
) {
	dual.Je_type = error_functional_type;
}


template<int dim>
void
Heat_cG_DWR<dim>::
set_grid(std::shared_ptr< Grid_DWR<dim,1> > _grid) {
	grid = _grid;
}


template<int dim>
void
Heat_cG_DWR<dim>::
set_epsilon(std::shared_ptr< dealii::Function<dim> > _epsilon) {
	epsilon = _epsilon;
}


template<int dim>
void
Heat_cG_DWR<dim>::
set_BoundaryValues(std::shared_ptr< dealii::Function<dim> > _BoundaryValues) {
	BoundaryValues = _BoundaryValues;
}


template<int dim>
void
Heat_cG_DWR<dim>::
set_BoundaryValues_dual(std::shared_ptr< dealii::Function<dim> > _BoundaryValues_dual) {
	BoundaryValues_dual = _BoundaryValues_dual;
}


template<int dim>
void
Heat_cG_DWR<dim>::
set_f(std::shared_ptr< dealii::Function<dim> > f) {
	function.f = f;
}


template<int dim>
void
Heat_cG_DWR<dim>::
set_f_dual(std::shared_ptr< dealii::Function<dim> > f_dual) {
	function.f_dual = f_dual;
}


template<int dim>
void
Heat_cG_DWR<dim>::
set_evaluation_point(dealii::Point<dim> evaluation_point) {
	dual.evaluation_point = evaluation_point;
}


template<int dim>
void
Heat_cG_DWR<dim>::
set_data(
	const unsigned int p_primal,
	const unsigned int p_dual,
	const unsigned int global_refinement,
	const unsigned int time_steps,
	double t0,
	double T,
	double tau_n) {
	
	// TODO: simplify
	data.p_primal = p_primal;
	data.p_dual   = p_dual;
	data.global_refinement = global_refinement;
	data.time_steps = time_steps;
	data.t0 = t0;
	data.T = T;
	data.tau_n = tau_n;
}


template<int dim>
void
Heat_cG_DWR<dim>::
set_data_output_patches(
	unsigned int _data_output_patches_primal,
	unsigned int _data_output_patches_dual) {
	primal.data_output_patches = _data_output_patches_primal;
	dual.data_output_patches = _data_output_patches_dual;
}


template<int dim>
void
Heat_cG_DWR<dim>::
init(const unsigned int global_refinement) {
	// create grid and distribute dofs
	grid->initialize_grids(data.t0, data.T, data.tau_n);
	grid->generate();
	grid->refine_global(global_refinement);
	
	////////////////////////////////////////////////////////////////////////////
	// init error estimators
	//
	error_estimator.DWR = std::make_shared<
		Heat::DWR::ErrorEstimator<dim> > ();
	error_estimator.DWR->set_objects(
		grid,
		epsilon,
		BoundaryValues,
		BoundaryValues_dual,
		function.f,
		function.f_dual
	);
}


template<int dim>
void
Heat_cG_DWR<dim>::
init_storage() {
	std::cout << "Initialize In_u, In_z and In_uprimal" << std::endl;
	
	In_u = std::make_shared<l_data_vectors_storage > ();
	In_z = std::make_shared<l_data_vectors_storage > ();
	In_uprimal = std::make_shared<l_data_vectors_storage > ();
	In_eta = std::make_shared<l_data_vectors_storage > ();
	
	for (unsigned int i{0}; i <= data.time_steps; ++i) {
		//TODO 160
		In_u->emplace_back();
		In_z->emplace_back();
		In_uprimal->emplace_back();
		auto &element = In_u->back();
		auto &element1 = In_z->back();
		auto &element2 = In_uprimal->back();
		element.x = std::make_shared< dealii::Vector<double> > ();
		element1.x = std::make_shared< dealii::Vector<double> > ();
		element2.x = std::make_shared< dealii::Vector<double> > ();
	} // end for loop i-->Initilization of primal and dual list of solution vectors
	
	// Prepare storage for error indicators vector eta(=sum over eta_K) 
	for (unsigned int i{1}; i <= data.time_steps; ++i) {
		In_eta->emplace_back();
		auto &element = In_eta->back();
		element.x = std::make_shared<dealii::Vector<double> > ();
	}
	std::cout << "In_u hat Groesse " << In_u->size() << std::endl;
	std::cout << "In_uprimal hat Groesse " << In_uprimal->size() << std::endl;
}

template<int dim>
void
Heat_cG_DWR<dim>::
primal_init_data_output() {
	////////////////////////////////////////////////////////////////////////////
	// INIT DATA OUTPUT
	//
	Assert(it_In_grid->primal.dof.use_count(), dealii::ExcNotInitialized());
	
	DTM::pout << "Heat DWR: primal solution data output: patches = " << primal.data_output_patches << std::endl;
	
	std::vector<std::string> data_field_names;
	data_field_names.push_back("u");
	
	std::vector< dealii::DataComponentInterpretation::DataComponentInterpretation > dci_field;
	dci_field.push_back(dealii::DataComponentInterpretation::component_is_scalar);
	
	primal.data_output.set_DoF_data(
		it_In_grid->primal.dof
	);
	
	primal.data_output.set_data_field_names(data_field_names);
	primal.data_output.set_data_component_interpretation_field(dci_field);
	primal.data_output.set_data_output_patches(primal.data_output_patches);
}


template<int dim>
void
Heat_cG_DWR<dim>::
dual_init_data_output() {
	////////////////////////////////////////////////////////////////////////////
	// INIT DATA OUTPUT
	//
	Assert(rit_In_grid->dual.dof.use_count(), dealii::ExcNotInitialized());
	
	DTM::pout << "Heat DWR: dual solution   data output: patches = " << dual.data_output_patches << std::endl;
	
	std::vector<std::string> data_field_names;
// 	data_field_names.push_back("u");
	data_field_names.push_back("z");
	
	std::vector< dealii::DataComponentInterpretation::DataComponentInterpretation > dci_field;
// 	dci_field.push_back(dealii::DataComponentInterpretation::component_is_scalar);
	dci_field.push_back(dealii::DataComponentInterpretation::component_is_scalar);
	
// 	dual.data_output.set_DoF_data(
// 		rit_In_grid->dual.dof
// 	);
	dual.data_output.set_DoF_data(
		it_In_grid->dual.dof
	);
	
	dual.data_output.set_data_field_names(data_field_names);
	dual.data_output.set_data_component_interpretation_field(dci_field);
	dual.data_output.set_data_output_patches(dual.data_output_patches);
}


template<int dim>
void
Heat_cG_DWR<dim>::
primal_reinit() {
	// now create vectors and matricies for primal problem
	Assert(it_In_grid->primal.dof.use_count(), dealii::ExcNotInitialized());
	Assert(it_In_grid->dual.dof.use_count(), dealii::ExcNotInitialized());
	Assert(it_In_grid->primal.sp.use_count(), dealii::ExcNotInitialized());
	
	primal.M.reinit(*(it_In_grid->primal.sp));
	primal.A.reinit(*(it_In_grid->primal.sp));
	primal.system_matrix.reinit(*(it_In_grid->primal.sp));
	
	primal.u = std::make_shared< dealii::Vector<double> > ();
	primal.u->reinit(it_In_grid->primal.dof->n_dofs());
	primal.u_old_interpolated = std::make_shared< dealii::Vector<double> > ();
	primal.u_old_interpolated->reinit(it_In_grid->primal.dof->n_dofs());
	dual.u = std::make_shared< dealii::Vector<double> > ();
	dual.u->reinit(it_In_grid->dual.dof->n_dofs());
	
	primal.f.reinit(it_In_grid->primal.dof->n_dofs());
	primal.system_rhs.reinit(it_In_grid->primal.dof->n_dofs());
	primal.testvector.reinit(it_In_grid->primal.dof->n_dofs());
}


template<int dim>
void
Heat_cG_DWR<dim>::
dual_reinit() {
	// now create vectors and matricies for dual problem
	Assert(rit_In_grid->dual.dof.use_count(), dealii::ExcNotInitialized());
	Assert(rit_In_grid->dual.sp.use_count(), dealii::ExcNotInitialized());
	
	dual.M.reinit(*(rit_In_grid->dual.sp));
	dual.A.reinit(*(rit_In_grid->dual.sp));
	dual.system_matrix.reinit(*(rit_In_grid->dual.sp));
	
	dual.z = std::make_shared< dealii::Vector<double> > ();
	dual.z->reinit(rit_In_grid->dual.dof->n_dofs());
	dual.z_old_interpolated = std::make_shared< dealii::Vector<double> > ();
	dual.z_old_interpolated->reinit(rit_In_grid->dual.dof->n_dofs());
	
	dual.Je.reinit(rit_In_grid->dual.dof->n_dofs());
	dual.system_rhs.reinit(rit_In_grid->dual.dof->n_dofs());
	
	//TEST // TODO
	dual.Je_old_interpolated.reinit(rit_In_grid->dual.dof->n_dofs());
	
// 	dual.solution_vectors.resize(0);
// 	dual.solution_vectors.push_back(dual.u);
// 	dual.solution_vectors.push_back(dual.z);
}


template<int dim>
void
Heat_cG_DWR<dim>::
primal_compute_initial_condition() {
	Assert(grid->slabs.front().primal.dof.use_count(), dealii::ExcNotInitialized());
	
	primal.u_old = std::make_shared< dealii::Vector<double> > ();
	primal.u_old->reinit(grid->slabs.front().primal.dof->n_dofs());
	dual.u = std::make_shared< dealii::Vector<double> > ();
	dual.u->reinit(grid->slabs.front().dual.dof->n_dofs());
	
	dealii::VectorTools::interpolate(
		*(grid->slabs.front().primal.dof),
		*(BoundaryValues),
		*(primal.u_old)
	);
	grid->slabs.front().primal.constraints->distribute(*(primal.u_old));

	dealii::FETools::interpolate(*(grid->slabs.front().primal.dof),
								*(primal.u_old),
								*(grid->slabs.front().dual.dof),
								*(grid->slabs.front().dual.constraints),
								*(dual.u));
	grid->slabs.front().dual.constraints->distribute(*(dual.u));
}


template<int dim>
void
Heat_cG_DWR<dim>::
primal_assemble_system() {
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	Assert(it_In_grid->primal.fe.use_count(), dealii::ExcNotInitialized());
	Assert(it_In_grid->primal.mapping.use_count(), dealii::ExcNotInitialized());
	Assert(it_In_grid->primal.constraints.use_count(), dealii::ExcNotInitialized());
	
	// ASSEMBLY ////////////////////////////////////////////////////////////////
	
	// Initialise the system matrix with 0.
	primal.M = 0;
	primal.A = 0;
	primal.system_matrix = 0;
	
	
// 	// TODO: warum p+2?
// 	
	// Setup a Gaussian quadrature formula
	// NOTE: We take p+1 quadrature points
	dealii::QGauss<dim> quad ( it_In_grid->primal.fe->tensor_degree()+2 ); //alter GaussLobatto
	
	// Setup a FE_Values object.
	// This is needed to get the needed information from the FiniteElement
	dealii::FEValues<dim> fe_values(
		*(it_In_grid->primal.mapping),
		*(it_In_grid->primal.fe),
		quad,
		dealii::update_values | 
		dealii::update_gradients | // update shape function gradient values
		dealii::update_hessians | 
		dealii::update_quadrature_points |
		dealii::update_JxW_values
	);
	
	// Setup a (small) full matrix, to store the assembly on each mesh cell
	// efficiently.
	// Afterwards they will be distributed into the global (sparse) matrix A.
	dealii::FullMatrix<double> local_M(
		it_In_grid->primal.fe->dofs_per_cell, it_In_grid->primal.fe->dofs_per_cell
	);
	dealii::FullMatrix<double> local_A(
		it_In_grid->primal.fe->dofs_per_cell, it_In_grid->primal.fe->dofs_per_cell
	);
	
	// Setup a small vector, to store the global dof indices.
	// NOTE: We are using a C++ standart template vector, not a deal.II
	// "Vector".
	// The FiniteElement object "fe" will tell us, how much dofs has each cell.
	std::vector< dealii::types::global_dof_index > local_dof_indices(
		it_In_grid->primal.fe->dofs_per_cell
	);
	
	// Now we do the real work, looping over all cells to compute the cell
	// assemblys. For this we need an iterator (kind of a pointer), which allows
	// us, to iterate over each cell easily.
	// We initialise it with the first active cell of our triangulation.
	auto cell = it_In_grid->primal.dof->begin_active();
	auto endc = it_In_grid->primal.dof->end();
	
	for ( ; cell != endc; ++cell) {
		// First we have to compute the values of the gradients and
		// the JxW values.
		// The reinit of the fe_values object on the current cell will do this.
		fe_values.reinit(cell);
		
		// Initialise the full matrix for the cell assembly with 0.
		local_M = 0;
		local_A = 0;
		
		// Now loop over all shape function combinations and quadrature points
		// to get the assembly.
		for (unsigned int i(0); i < it_In_grid->primal.fe->dofs_per_cell; ++i)
		for (unsigned int j(0); j < it_In_grid->primal.fe->dofs_per_cell; ++j)
		for (unsigned int q(0); q < quad.size(); ++q) {
			local_M(i,j) += (
				fe_values.shape_value(i,q) *
				fe_values.shape_value(j,q) *
				fe_values.JxW(q)
			);
			
			local_A(i,j) += (
				fe_values.shape_grad(i,q) *
				epsilon->value(fe_values.quadrature_point(q), 0) *
				fe_values.shape_grad(j,q) *
				fe_values.JxW(q)
			);
		}
		
		// Store the global indices into the vector local_dof_indices.
		// The cell object will give us the information.
		Assert(
			(local_dof_indices.size() == it_In_grid->primal.fe->dofs_per_cell),
			dealii::ExcNotInitialized()
		);
		cell->get_dof_indices(local_dof_indices);
		
		// Copy the local assembly to the global matrix.
		// We use the constraint object, to set all constraints with that step.
		Assert(
			it_In_grid->primal.constraints.use_count(),
			dealii::ExcNotInitialized()
		);
		
		it_In_grid->primal.constraints->distribute_local_to_global(
			local_M, local_dof_indices, primal.M
		);
		
		it_In_grid->primal.constraints->distribute_local_to_global(
			local_A, local_dof_indices, primal.A
		);
	}
	
	// Put the computed matrices into the system matrix (for later solving Ax=b)
	// Here A = system_matrix =  M + tau_n*A
	primal.system_matrix.copy_from(primal.M);
	primal.system_matrix.add(data.tau_n,primal.A); //dG(0)
}


template<int dim>
void
Heat_cG_DWR<dim>::
primal_set_time(double t_1) {
	BoundaryValues->set_time(t_1);
	function.f->set_time(t_1);
}


template<int dim>
void
Heat_cG_DWR<dim>::
primal_assemble_f() {
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	Assert(it_In_grid->primal.fe.use_count(), dealii::ExcNotInitialized());
	Assert(it_In_grid->primal.mapping.use_count(), dealii::ExcNotInitialized());
	Assert(it_In_grid->primal.constraints.use_count(), dealii::ExcNotInitialized());
	Assert(function.f.use_count(), dealii::ExcNotInitialized());
	primal.f = 0;
	
	dealii::QGauss<dim> quad ( it_In_grid->primal.fe->tensor_degree()+2 ); //GaussLobatto
	
	// Setup a FE_Values object.
	// This is needed to get the needed information from the FiniteElement
	dealii::FEValues<dim> fe_values(
		*(it_In_grid->primal.mapping),
		*(it_In_grid->primal.fe),
		quad,
		dealii::update_values |
		dealii::update_gradients |
		dealii::update_quadrature_points |
		dealii::update_JxW_values
	);
	
	dealii::Vector<double> local_f(
		it_In_grid->primal.fe->dofs_per_cell
	);
	
	// Setup a small vector, to store the global dof indices.
	// NOTE: We are using a C++ standart template vector, not a deal.II
	// "Vector".
	// The FiniteElement object "fe" will tell us, how much dofs has each cell.
	std::vector< dealii::types::global_dof_index > local_dof_indices(
		it_In_grid->primal.fe->dofs_per_cell
	);
	
	// Now we do the real work, looping over all cells to compute the cell
	// assemblies. For this we need an iterator (kind of a pointer), which allows
	// us, to iterate over each cell easily.
	auto cell = it_In_grid->primal.dof->begin_active();
	auto endc = it_In_grid->primal.dof->end();
	
	for ( ; cell != endc; ++cell) {
		// The reinit of the fe_values object on the current cell.
		fe_values.reinit(cell);
		
		// Initialise the vector for the cell assemblies with 0.
		local_f = 0;

		// Now loop over all shape function combinations and quadrature points
		// to get the assembly.
		for (unsigned int i(0); i < it_In_grid->primal.fe->dofs_per_cell; ++i)
		for (unsigned int q(0); q < quad.size(); ++q) {
			local_f(i) +=(
				(fe_values.shape_value(i,q) *
				function.f->value(fe_values.quadrature_point(q), 0)
				)*
				fe_values.JxW(q));
		}
		
		// Store the global indices into the vector local_dof_indices.
		// The cell object will give us the information.
		cell->get_dof_indices(local_dof_indices);
		
		// Copy the local assembly to the global matrix.
		// We use the constraint object, to set all constraints with that step.
		it_In_grid->primal.constraints->distribute_local_to_global(
			local_f, local_dof_indices, primal.f
		);
	}
}


template<int dim>
void
Heat_cG_DWR<dim>::
primal_interpolate_to_next_grid() {
	dealii::VectorTools::interpolate_to_different_mesh(
		*(it_In_grid_previous->primal.dof),
		*(primal.u_old),
		*(it_In_grid->primal.dof),
		*(it_In_grid->primal.constraints),
		*(primal.u_old_interpolated)
	);
}


template<int dim>
void
Heat_cG_DWR<dim>::
primal_assemble_rhs() {
	primal.system_rhs = 0;
	primal.M.vmult(primal.system_rhs, *(primal.u_old_interpolated));
	primal.system_rhs.add(data.tau_n, primal.f);
}


template<int dim>
void
Heat_cG_DWR<dim>::
primal_solve() {
	////////////////////////////////////////////////////////////////////////////
	// apply Dirichlet boundary values
	std::map<dealii::types::global_dof_index, double> boundary_values;
	dealii::VectorTools::interpolate_boundary_values(
		*(it_In_grid->primal.dof),
		static_cast< dealii::types::boundary_id > (
			Heat::types::boundary_id::Dirichlet
		),
		*(BoundaryValues),
		boundary_values
	);
	
	dealii::MatrixTools::apply_boundary_values(
		boundary_values,
		primal.system_matrix,
		*(primal.u),
		primal.system_rhs
	);
	
	////////////////////////////////////////////////////////////////////////////
	// solve linear system directly
	dealii::SparseDirectUMFPACK iA;
	iA.initialize(primal.system_matrix);
	iA.vmult(*(primal.u), primal.system_rhs);
	
	////////////////////////////////////////////////////////////////////////////
	// distribute hanging node constraints on solution
	it_In_grid->primal.constraints->distribute(*primal.u);
}


template<int dim>
void
Heat_cG_DWR<dim>::
interpolate_primal_to_dual() {
	dealii::FETools::interpolate(
		*(it_In_grid->primal.dof),
		*(primal.u),
		*(it_In_grid->dual.dof),
		*(it_In_grid->dual.constraints),
		*(dual.u)
	);
}


template<int dim>
void
Heat_cG_DWR<dim>::
primal_do_data_output(const double n) {
	const double solution_time = n;
	
	primal.data_output.write_data("primal", primal.u, solution_time);
}


template<int dim>
void
Heat_cG_DWR<dim>::
primal_process_solution(const unsigned int cycle) {
	dealii::Vector<double> difference_per_cell (it_In_grid->tria->n_active_cells());
	const dealii::QTrapez<1> q_trapez;
	const dealii::QIterated<dim> q_iterated (q_trapez,20);
	
// 	BoundaryValues->set_time(data.T);
	
	dealii::VectorTools::integrate_difference (*(it_In_grid->primal.mapping),
												*(it_In_grid->primal.dof),
												*(primal.u),
												*(BoundaryValues),
												difference_per_cell,
												q_iterated,//dealii::QGauss<dim>(4),//q_iterated,//dealii::QGauss<dim>(4),// q_iterated (alternativ)
												dealii::VectorTools::L2_norm);
	const double L2_error = difference_per_cell.l2_norm();
	L2Error = L2_error;
	
// 	dealii::VectorTools::integrate_difference (*(it_In_grid->primal.mapping),
// 												*(it_In_grid->primal.dof),
// 												*(primal.u),
// 												*(BoundaryValues),
// 												difference_per_cell,
// 												dealii::QGauss<dim>(4),
// 												dealii::VectorTools::H1_seminorm);
// 	const double H1_error = difference_per_cell.l2_norm();
	
	
	const unsigned int n_active_cells=it_In_grid->tria->n_active_cells();
	const unsigned int n_dofs=it_In_grid->primal.dof->n_dofs();
	
	std::cout 	<< "Cycle " << cycle << ':'
				<< std::endl
				<< " Number of active cells: "
				<< n_active_cells
				<< std::endl
				<< " Number of degrees of freedom: "
				<< n_dofs
				<< std::endl;
				
	primal.convergence_table.add_value("cycle",cycle);
	primal.convergence_table.add_value("cells",n_active_cells);
	primal.convergence_table.add_value("dofs",n_dofs);
	primal.convergence_table.add_value("tau_n",data.tau_n);
	primal.convergence_table.add_value("L2atT",L2_error);
// 	primal.convergence_table.add_value("H1",H1_error);
}

////////////////////////////////////////////////////////////////////////////////
/////////// DUAL ///////////////////////////////////////////////////////////////

template<int dim>
void
Heat_cG_DWR<dim>::
dual_compute_initial_condition() {
	
	switch (dual.Je_type) {
		case Heat::types::error_functional::forbidden:
			AssertThrow(false, dealii::ExcMessage("You need to initialise dual.Je_type."));
			break;
			
		case Heat::types::error_functional::L2_final:
			dual_compute_initial_condition_L2final();
			// Initialize rhs_vector dual.Je_old 
			dual.Je_old.reinit(rit_In_grid->dual.dof->n_dofs());
			break;
			
		case Heat::types::error_functional::L2_global:
			// Compute ones the global space time L2 Error:
			compute_global_STL2_error();
			//
			dual_compute_initial_condition_L2global();
			// Initialize rhs_vectors dual.Je_old and dual.Je
			dual.Je_old.reinit(rit_In_grid->dual.dof->n_dofs());
			dual.Je.reinit(rit_In_grid->dual.dof->n_dofs());
			// Compute dual.Je at timepoint t_N = 0.5 (within dual_assemble_je())
			// and store it in dual.Je_old.
			dual_assemble_Je_L2global();
			dual.Je_old = dual.Je;
			break;
			
		case Heat::types::error_functional::mean_final:
			dual_compute_initial_condition_mean_final();
			// Initialize rhs_vector dual.Je_old 
			dual.Je_old.reinit(rit_In_grid->dual.dof->n_dofs());
			break;
			
		case Heat::types::error_functional::mean_global:
			dual_compute_initial_condition_mean_global();
			// Initialize rhs_vectors dual.Je_old and dual.Je
			dual.Je_old.reinit(rit_In_grid->dual.dof->n_dofs());
			dual.Je.reinit(rit_In_grid->dual.dof->n_dofs());
			// Compute dual.Je at timepoint t_N = 0.5 (within dual_assemble_je())
			// and store it in dual.Je_old.
			dual_assemble_Je_mean_global();
			dual.Je_old = dual.Je;
			break;
			
		case Heat::types::error_functional::point:
			dual_compute_initial_condition_point_final();
			// Initialize rhs_vectors dual.Je_old
			dual.Je_old.reinit(rit_In_grid->dual.dof->n_dofs());
			break;
			
		default:
			AssertThrow(false, dealii::ExcMessage("Your dual.Je_type is unknown, please check your inputs."));
	} // end switch(dual.Je_type)
}


template<int dim>
void
Heat_cG_DWR<dim>::
dual_compute_initial_condition_L2final() {
	Assert(grid->slabs.back().dual.dof.use_count(), dealii::ExcNotInitialized());

////////////////////////////////////////////////////////////////////////////////
	// To solve dual problem backward without error functional, instead an rhs given
	// by -d_tz-laplace(z)
////////////////////////////////////////////////////////////////////////////////
	
// 	dual.z_old = std::make_shared< dealii::Vector<double> > ();
// 	dual.z_old->reinit(grid->slabs.back().dual.dof->n_dofs());
// 	
// 	BoundaryValues_dual->set_time(data.T);
// 	
// 	dealii::VectorTools::interpolate(
// 		*(grid->slabs.back().dual.dof),
// 		*(BoundaryValues_dual),
// 		*(dual.z_old)
// 	);
// 	grid->slabs.back().dual.constraints->distribute(*(dual.z_old));

////////////////////////////////////////////////////////////////////////////////
// 	//To solve dual problem for error functional J(phi)=(phi_N^-,e_N^-)/norm(e_N^-),
// 	//describing the L^2-error at the final trimepoint T = 0.5
////////////////////////////////////////////////////////////////////////////////
	
	dual.z_old = std::make_shared< dealii::Vector<double> > ();
	dual.z_old->reinit(grid->slabs.back().dual.dof->n_dofs());
	std::shared_ptr< dealii::Vector<double> > primal_u_exact;
	primal_u_exact = std::make_shared< dealii::Vector<double> > ();
	primal_u_exact->reinit(grid->slabs.back().primal.dof->n_dofs());
	std::shared_ptr< dealii::Vector<double> > dual_initial_condition;
	dual_initial_condition = std::make_shared< dealii::Vector<double> > ();
	dual_initial_condition->reinit(grid->slabs.back().primal.dof->n_dofs());
	
	BoundaryValues_dual->set_time(data.T);
	
	dealii::VectorTools::interpolate(
		*(grid->slabs.back().primal.mapping),
		*(grid->slabs.back().primal.dof),
		*(BoundaryValues_dual),
		*(primal_u_exact)
	);
	grid->slabs.back().primal.constraints->distribute(*(primal_u_exact));	
	
	dealii::Vector<double> difference_per_cell1 (grid->slabs.back().tria->n_active_cells());
	const dealii::QTrapez<1> q_trapez;
	const dealii::QIterated<dim> q_iterated (q_trapez,20);
	dealii::VectorTools::integrate_difference(
		*(grid->slabs.back().primal.mapping),
		*(grid->slabs.back().primal.dof),
		*(In_uprimal->back().x),
		*(BoundaryValues_dual),
		difference_per_cell1,
		q_iterated,//dealii::QGauss<dim>(4),
		dealii::VectorTools::L2_norm
	);
	const double L_2error_primal = difference_per_cell1.l2_norm();
	
	*(dual_initial_condition) = *(primal_u_exact);
	*(dual_initial_condition) -= *(In_uprimal->back().x); 
	*(dual_initial_condition) *= (1./L_2error_primal);
	
	// Interpolate initial condition into dual FE room.
	dealii::FETools::interpolate(
		*(grid->slabs.back().primal.dof),
		*(dual_initial_condition),
		*(grid->slabs.back().dual.dof),
		*(grid->slabs.back().dual.constraints),
		*(dual.z_old)
	);
	
////////////////////////////////////////////////////////////////////////////////
//// Alternative direkt alles im dualen FE-Raum berechnen
////////////////////////////////////////////////////////////////////////////////
	
// 	dual.z_old = std::make_shared< dealii::Vector<double> > ();
// 	dual.z_old->reinit(grid->slabs.back().dual.dof->n_dofs());
// 	std::shared_ptr< dealii::Vector<double> > primal_u_exact;
// 	primal_u_exact = std::make_shared< dealii::Vector<double> > ();
// 	primal_u_exact->reinit(grid->slabs.back().primal.dof->n_dofs());
// 	std::shared_ptr< dealii::Vector<double> > u_int_exact;
// 	u_int_exact = std::make_shared< dealii::Vector<double> > ();
// 	u_int_exact->reinit(grid->slabs.back().dual.dof->n_dofs());
// 	std::shared_ptr< dealii::Vector<double> > dual_initial_condition;
// 	dual_initial_condition = std::make_shared< dealii::Vector<double> > ();
// 	dual_initial_condition->reinit(grid->slabs.back().dual.dof->n_dofs());
// 	
// 	BoundaryValues_dual->set_time(data.T);
// 	
// // 	dealii::VectorTools::interpolate(
// // 		*(grid->slabs.back().primal.mapping),
// // 		*(grid->slabs.back().primal.dof),
// // 		*(BoundaryValues_dual),
// // 		*(primal_u_exact)
// // 	);
// // 	grid->slabs.back().primal.constraints->distribute(*(primal_u_exact));	
// // 	dealii::FETools::interpolate(
// // 		*(grid->slabs.back().primal.dof),
// // 		*(primal_u_exact),
// // 		*(grid->slabs.back().dual.dof),
// // 		*(grid->slabs.back().dual.constraints),
// // 		*(u_int_exact)
// // 	);
// 	
// 	dealii::VectorTools::interpolate(
// 		*(grid->slabs.back().dual.mapping),
// 		*(grid->slabs.back().dual.dof),
// 		*(BoundaryValues_dual),
// 		*(u_int_exact)
// 	);
// 	grid->slabs.back().dual.constraints->distribute(*(u_int_exact));
// 	
// 	dealii::Vector<double> difference_per_cell2 (grid->slabs.back().tria->n_active_cells());
// 	dealii::VectorTools::integrate_difference(
// 		*(grid->slabs.back().dual.mapping),
// 		*(grid->slabs.back().dual.dof),
// 		*(In_u->back().x),
// 		*(BoundaryValues_dual),
// 		difference_per_cell2,
// 		dealii::QGauss<dim>(4),
// 		dealii::VectorTools::L2_norm
// 	);
// 	const double L_2error_dual = difference_per_cell2.l2_norm();
// 	
// 	*(dual_initial_condition) = *(u_int_exact);
// 	*(dual_initial_condition) -= *(In_u->back().x);
// 	*(dual_initial_condition) *= (1./L_2error_dual);
// 
// 	
// 	*(dual.z_old) = *(dual_initial_condition);
	
}


template<int dim>
void
Heat_cG_DWR<dim>::
dual_compute_initial_condition_L2global() {
	Assert(grid->slabs.back().dual.dof.use_count(), dealii::ExcNotInitialized());
////////////////////////////////////////////////////////////////////////////////
	// Initial condition for global L2Error-functional
////////////////////////////////////////////////////////////////////////////////
	
	dual.z_old = std::make_shared< dealii::Vector<double> > ();
	dual.z_old->reinit(grid->slabs.back().dual.dof->n_dofs());
}


template<int dim>
void
Heat_cG_DWR<dim>::
dual_compute_initial_condition_mean_final() {
	
////////////////////////////////////////////////////////////////////////////////
	// Initial condition for final mean-value-functional
////////////////////////////////////////////////////////////////////////////////
	
	dual.z_old = std::make_shared< dealii::Vector<double> > ();
	dual.z_old->reinit(grid->slabs.back().dual.dof->n_dofs());
	
	for (unsigned int i=0; i < dual.z_old->size(); ++i) {
		(*(dual.z_old))[i] = 1.;
	}
}


template<int dim>
void
Heat_cG_DWR<dim>::
dual_compute_initial_condition_mean_global() {
	Assert(grid->slabs.back().dual.dof.use_count(), dealii::ExcNotInitialized());
////////////////////////////////////////////////////////////////////////////////
	// Initial condition for global mean-functional
////////////////////////////////////////////////////////////////////////////////
	
	dual.z_old = std::make_shared< dealii::Vector<double> > ();
	dual.z_old->reinit(grid->slabs.back().dual.dof->n_dofs());
}


template<int dim>
void
Heat_cG_DWR<dim>::
dual_compute_initial_condition_point_final() {
////////////////////////////////////////////////////////////////////////////////
	// Initial condition for final point-value-functional
////////////////////////////////////////////////////////////////////////////////
	
	dual.z_old = std::make_shared< dealii::Vector<double> > ();
	dual.z_old->reinit(grid->slabs.back().dual.dof->n_dofs());
	
	auto cell = grid->slabs.back().dual.dof->begin_active();
	auto endc = grid->slabs.back().dual.dof->end();
	
	for (; cell != endc; ++cell) {
	for (unsigned int vertex = 0;
		vertex < dealii::GeometryInfo<dim>::vertices_per_cell;
		++vertex) {
		if (cell->vertex(vertex).distance(dual.evaluation_point) < cell->diameter()*1.e-8) {
			(*(dual.z_old))(cell->vertex_dof_index(vertex,0)) = 1;
		}
		else {
			(*(dual.z_old))(cell->vertex_dof_index(vertex,0)) = 0;
		}
	} // end for-loop vertex
	} // end for-loop cell
	
// 	std::cout << "dual.z_oelderchen = " << *(dual.z_old) << std::endl;
}


template<int dim>
void
Heat_cG_DWR<dim>::
dual_assemble_system() {
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	Assert(rit_In_grid->dual.fe.use_count(), dealii::ExcNotInitialized());
	Assert(rit_In_grid->dual.mapping.use_count(), dealii::ExcNotInitialized());
	Assert(rit_In_grid->dual.constraints.use_count(), dealii::ExcNotInitialized());
	
	// ASSEMBLY ////////////////////////////////////////////////////////////////
	
	// Initialise the system matrix with 0.
	dual.M = 0;
	dual.A = 0;
	dual.system_matrix = 0;
	
	// Setup a Gaussian quadrature formula
	// NOTE: We take p+1 quadrature points
	dealii::QGauss<dim> quad ( rit_In_grid->dual.fe->tensor_degree()+2);
	
	// Setup a FE_Values object.
	// This is needed to get the needed information from the FiniteElement
	dealii::FEValues<dim> fe_values(
		*(rit_In_grid->dual.mapping),
		*(rit_In_grid->dual.fe),
		quad,
		dealii::update_values | 
		dealii::update_gradients | // update shape function gradient values
		dealii::update_hessians | 
		dealii::update_quadrature_points |
		dealii::update_JxW_values
	);
	
	// Setup a (small) full matrix, to store the assembly on each mesh cell
	// efficiently.
	// Afterwards they will be distributed into the global (sparse) matrix M, A.
	dealii::FullMatrix<double> local_M(
		rit_In_grid->dual.fe->dofs_per_cell, rit_In_grid->dual.fe->dofs_per_cell
	);
	dealii::FullMatrix<double> local_A(
		rit_In_grid->dual.fe->dofs_per_cell, rit_In_grid->dual.fe->dofs_per_cell
	);
	
	// Setup a small vector, to store the global dof indices.
	// NOTE: We are using a C++ standart template vector, not a deal.II
	// "Vector".
	// The FiniteElement object "fe" will tell us, how much dofs has each cell.
	std::vector< dealii::types::global_dof_index > local_dof_indices(
		rit_In_grid->dual.fe->dofs_per_cell
	);
	
	// Now we do the real work, looping over all cells to compute the cell
	// assemblys. For this we need an iterator (kind of a pointer), which allows
	// us, to iterate over each cell easily.
	// We initialise it with the first active cell of our triangulation.
	auto cell = rit_In_grid->dual.dof->begin_active();
	auto endc = rit_In_grid->dual.dof->end();
	
	for ( ; cell != endc; ++cell) {
		// First we have to compute the values of the gradients and
		// the JxW values.
		// The reinit of the fe_values object on the current cell will do this.
		fe_values.reinit(cell);
		
		// Initialise the full matrix for the cell assembly with 0.
		local_M = 0;
		local_A = 0;

		// Now loop over all shape function combinations and quadrature points
		// to get the assembly.
		for (unsigned int q(0); q < quad.size(); ++q)
		for (unsigned int i(0); i < rit_In_grid->dual.fe->dofs_per_cell; ++i)
		for (unsigned int j(0); j < rit_In_grid->dual.fe->dofs_per_cell; ++j){
			// diffusion convection reaction
			// loop to get the diagonal entries of the Hessian matrix (for Laplace(u_h))
			// Define a variable to calculate Laplace(u_h)
			double laplace_op_of_u_h = 0;
			for (unsigned int k = 0; k < dim; ++k) {
				laplace_op_of_u_h += fe_values.shape_hessian(j,q)[k][k];
			}
			local_M(i,j) += (
				(fe_values.shape_value(i,q)*
				fe_values.shape_value(j,q))*
				fe_values.JxW(q));
			local_A(i,j) +=(
				(fe_values.shape_grad(i,q) *
				(epsilon->value(fe_values.quadrature_point(q), 0)*
				fe_values.shape_grad(j,q)))*
				fe_values.JxW(q));

		}
		
		// Store the global indices into the vector local_dof_indices.
		// The cell object will give us the information.
		Assert(
			(local_dof_indices.size() == rit_In_grid->dual.fe->dofs_per_cell),
			dealii::ExcNotInitialized()
		);
		cell->get_dof_indices(local_dof_indices);
		
		// Copy the local assembly to the global matrix.
		// We use the constraint object, to set all constraints with that step.
		Assert(rit_In_grid->dual.constraints.use_count(), dealii::ExcNotInitialized());
		
		rit_In_grid->dual.constraints->distribute_local_to_global(
			local_M, local_dof_indices, dual.M
		);
		rit_In_grid->dual.constraints->distribute_local_to_global(
			local_A, local_dof_indices, dual.A
		);
	}
	dual.system_matrix.copy_from(dual.M);
	dual.system_matrix.add((data.tau_n/2.),dual.A);
}


template<int dim>
void
Heat_cG_DWR<dim>::
dual_set_time(double t_2) {
	BoundaryValues_dual->set_time(t_2);
// 	function.f_dual->set_time(t_2);
}

////////////////////////////////////////////////////////////////////////////////
//////////  dual_assemble_rhs() ////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<int dim>
void
Heat_cG_DWR<dim>::
dual_assemble_rhs_at_t_Nminus1() {
	switch (dual.Je_type) {
		case Heat::types::error_functional::forbidden:
			AssertThrow(false, dealii::ExcMessage("You need to initialise dual.Je_type."));
			break;
		
		case Heat::types::error_functional::L2_final:
			dual_assemble_Je_L2final();
			// Compute special RHS on last Interval I_N 
			dual.system_rhs = 0;
			(dual.system_rhs).add(1.,dual.Je);
			// End of computation of RHS
			break;
			
		case Heat::types::error_functional::L2_global:
			dual_assemble_Je_L2global();
			dual_interpolate_to_next_grid();
			dual_assemble_rhs_L2global();
			break;
			
		case Heat::types::error_functional::mean_final:
			dual_assemble_Je_mean_final();
			// Compute special RHS on last Interval I_N 
			dual.system_rhs = 0;
			(dual.system_rhs).add(1.,dual.Je);
			// End of computation of RHS
			break;
			
		case Heat::types::error_functional::mean_global:
			dual_assemble_Je_mean_global();
			dual_interpolate_to_next_grid();
			dual_assemble_rhs_mean_global();
			break;
			
		case Heat::types::error_functional::point:
			dual_assemble_Je_point_final();
			// Compute special RHS on last Interval I_N 
			dual.system_rhs = 0;
			(dual.system_rhs).add(1.,dual.Je);
			// End of computation of RHS
			break;
			
		default:
			AssertThrow(false, dealii::ExcMessage("Your dual.Je_type is unknown, please check your inputs."));
	} // end switch(dual.Je-type)
}


template<int dim>
void
Heat_cG_DWR<dim>::
dual_assemble_rhs() {
	switch (dual.Je_type) {
		case Heat::types::error_functional::forbidden:
			AssertThrow(false, dealii::ExcMessage("You need to initialise dual.Je_type."));
			break;
		
		case Heat::types::error_functional::L2_final:
			dual_interpolate_to_next_grid();
			dual_assemble_rhs_L2final();
			break;
			
		case Heat::types::error_functional::L2_global:
			dual_assemble_Je_L2global();
			dual_interpolate_to_next_grid();
			dual_assemble_rhs_L2global();
			break;
			
		case Heat::types::error_functional::mean_final:
			dual_interpolate_to_next_grid();
			dual_assemble_rhs_mean_final();
			break;
			
		case Heat::types::error_functional::mean_global:
			dual_assemble_Je_mean_global();
			dual_interpolate_to_next_grid();
			dual_assemble_rhs_mean_global();
			break;
			
		case Heat::types::error_functional::point:
			dual_interpolate_to_next_grid();
			dual_assemble_rhs_point_final();
			break;
			
		default:
			AssertThrow(false, dealii::ExcMessage("Your dual.Je_type is unknown, please check your inputs."));
	}
}


////////////////////////////////////////////////////////////////////////////////
//////////  L2Error_final-Evaluation (J(phi)=1/(L2-Norm(e_N))*(e_N,phi_N)) //////////////////
////////////////////////////////////////////////////////////////////////////////

template<int dim>
void
Heat_cG_DWR<dim>::
dual_assemble_Je_L2final() {
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	Assert(rit_In_grid->dual.fe.use_count(), dealii::ExcNotInitialized());
	Assert(rit_In_grid->dual.mapping.use_count(), dealii::ExcNotInitialized());
	Assert(rit_In_grid->dual.constraints.use_count(), dealii::ExcNotInitialized());
	
	dual.Je = 0;
	BoundaryValues_dual->set_time(data.T);
	double L2Error_final;
	L2Error_final = 0;
	// Computaion of global exact error J(e)
	dealii::Vector<double> difference_per_cell (grid->slabs.back().tria->n_active_cells());
	const dealii::QTrapez<1> q_trapez;
	const dealii::QIterated<dim> q_iterated (q_trapez,20);
	dealii::VectorTools::integrate_difference(
		*(grid->slabs.back().primal.mapping),
		*(grid->slabs.back().primal.dof),
		*(In_uprimal->back().x),
		*(BoundaryValues_dual),
		difference_per_cell,
		q_iterated,//dealii::QGauss<dim>(4),
		dealii::VectorTools::L2_norm
	);	
	L2Error_final = difference_per_cell.l2_norm();
	
	std::cout << "L2Error_final = " << L2Error_final << std::endl;
	
	auto u_final = std::make_shared< dealii::Vector<double> > ();
	u_final->reinit(rit_In_grid->dual.dof->n_dofs());
	dealii::VectorTools::interpolate_to_different_mesh(
		*(rit_In_grid_previous->dual.dof),
		*(In_u->back().x),
		*(rit_In_grid->dual.dof),
		*(rit_In_grid->dual.constraints),
		*(u_final)
	);
	
 	dealii::QGauss<dim> quad (rit_In_grid->dual.fe->tensor_degree()+2);
	
	// Setup a FE_Values object.
	dealii::FEValues<dim> fe_values(
		*(rit_In_grid->dual.mapping),
		*(rit_In_grid->dual.fe),
		quad,
		dealii::update_values |
		dealii::update_gradients |
		dealii::update_quadrature_points |
		dealii::update_JxW_values);
	
	dealii::Vector<double> local_dual_Je (
		rit_In_grid->dual.fe->dofs_per_cell
	);
	
	// stationärer Fall
	std::vector<double> exact_solution_values(quad.size());
	std::vector<double> u_h_values(quad.size());
	
	// Setup a small vector, to store the global dof indices.
	// NOTE: We are using a C++ standart template vector, not a deal.II
	// "Vector".
	// The FiniteElement object "fe" will tell us, how much dofs has each cell.
	std::vector< dealii::types::global_dof_index > local_dof_indices(
		rit_In_grid->dual.fe->dofs_per_cell
	);
	
	// Now we do the real work, looping over all cells to compute the cell
	// assemblys. For this we need an iterator (kind of a pointer), which allows
	// us, to iterate over each cell easily.
	// We initialise it with the first active cell of our triangulation.
	auto cell = rit_In_grid->dual.dof->begin_active();
	auto endc = rit_In_grid->dual.dof->end();
	
	for ( ; cell != endc; ++cell) {
		// First we have to compute the values of the gradients and
		// the JxW values.
		// The reinit of the fe_values object on the current cell will do this.
		fe_values.reinit(cell);
		
		// Initialise the full matrix for the cell assembly with 0.
		local_dual_Je = 0;
		
	// stationärer Fall	
		// Set up the exact solution vector
		BoundaryValues_dual->value_list(fe_values.get_quadrature_points(),
								   exact_solution_values);
		// Set up the computed solution vector
		// TODO Interpolate z_N to grid of z_N-1
		fe_values.get_function_values(*(u_final),
									  u_h_values);
		
		// Now loop over all shape function combinations and quadrature points
		// to get the assembly.
		for (unsigned int i(0); i < rit_In_grid->dual.fe->dofs_per_cell; ++i)
		for (unsigned int q(0); q < quad.size(); ++q) {
			local_dual_Je(i) +=
				(1./L2Error_final) *
				(((exact_solution_values[q]-u_h_values[q]) *
				fe_values.shape_value(i,q))* 
				fe_values.JxW(q));
		}
		
		// Store the global indices into the vector local_dof_indices.
		// The cell object will give us the information.
		Assert(
			(local_dof_indices.size() == rit_In_grid->dual.fe->dofs_per_cell),
			dealii::ExcNotInitialized()
		);
		cell->get_dof_indices(local_dof_indices);
		
		// Copy the local assembly to the global matrix.
		// We use the constraint object, to set all constraints with that step.
		Assert(rit_In_grid->dual.constraints.use_count(), dealii::ExcNotInitialized());
		
		rit_In_grid->dual.constraints->distribute_local_to_global(
			local_dual_Je, local_dof_indices, dual.Je
		);
		
	}
	
}



////////////////////////////////////////////////////////////////////////////////
//////////  L2Error_global-Evaluation (J(phi)=1/(L2-Norm(e))*(e,phi)) //////////////////
////////////////////////////////////////////////////////////////////////////////

template<int dim>
void
Heat_cG_DWR<dim>::
dual_assemble_Je_L2global() {
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	Assert(rit_In_grid->dual.fe.use_count(), dealii::ExcNotInitialized());
	Assert(rit_In_grid->dual.mapping.use_count(), dealii::ExcNotInitialized());
	Assert(rit_In_grid->dual.constraints.use_count(), dealii::ExcNotInitialized());
	
	dual.Je = 0;
	
 	dealii::QGauss<dim> quad (rit_In_grid->dual.fe->tensor_degree()+2);
	
	// Setup a FE_Values object.
	dealii::FEValues<dim> fe_values(
		*(rit_In_grid->dual.mapping),
		*(rit_In_grid->dual.fe),
		quad,
		dealii::update_values |
		dealii::update_gradients |
		dealii::update_quadrature_points |
		dealii::update_JxW_values);
	
	dealii::Vector<double> local_dual_Je (
		rit_In_grid->dual.fe->dofs_per_cell
	);
	
	// stationärer Fall
	std::vector<double> exact_solution_values(quad.size());
	std::vector<double> u_h_values(quad.size());
	
	// Setup a small vector, to store the global dof indices.
	// NOTE: We are using a C++ standart template vector, not a deal.II
	// "Vector".
	// The FiniteElement object "fe" will tell us, how much dofs has each cell.
	std::vector< dealii::types::global_dof_index > local_dof_indices(
		rit_In_grid->dual.fe->dofs_per_cell
	);
	
	// Now we do the real work, looping over all cells to compute the cell
	// assemblys. For this we need an iterator (kind of a pointer), which allows
	// us, to iterate over each cell easily.
	// We initialise it with the first active cell of our triangulation.
	auto cell = rit_In_grid->dual.dof->begin_active();
	auto endc = rit_In_grid->dual.dof->end();
	
	for ( ; cell != endc; ++cell) {
		// First we have to compute the values of the gradients and
		// the JxW values.
		// The reinit of the fe_values object on the current cell will do this.
		fe_values.reinit(cell);
		
		// Initialise the full matrix for the cell assembly with 0.
		local_dual_Je = 0;
		
	// stationärer Fall	
		// Set up the exact solution vector
		BoundaryValues_dual->value_list(fe_values.get_quadrature_points(),
								   exact_solution_values);
		// Set up the computed solution vector
		fe_values.get_function_values(*(rit_In_uback->x),
									  u_h_values);
		
		// Now loop over all shape function combinations and quadrature points
		// to get the assembly.
		for (unsigned int i(0); i < rit_In_grid->dual.fe->dofs_per_cell; ++i)
		for (unsigned int q(0); q < quad.size(); ++q) {
			local_dual_Je(i) +=
				(1./L2Error_global) *
				(((exact_solution_values[q]-u_h_values[q]) *
				fe_values.shape_value(i,q))* 
				fe_values.JxW(q));
		}
		
		// Store the global indices into the vector local_dof_indices.
		// The cell object will give us the information.
		Assert(
			(local_dof_indices.size() == rit_In_grid->dual.fe->dofs_per_cell),
			dealii::ExcNotInitialized()
		);
		cell->get_dof_indices(local_dof_indices);
		
		// Copy the local assembly to the global matrix.
		// We use the constraint object, to set all constraints with that step.
		Assert(rit_In_grid->dual.constraints.use_count(), dealii::ExcNotInitialized());
		
		rit_In_grid->dual.constraints->distribute_local_to_global(
			local_dual_Je, local_dof_indices, dual.Je
		);
		
	}
	
}


////////////////////////////////////////////////////////////////////////////////
//////////  Meanvalue_final-Evaluation (J(phi)=int_I int_Omega phi_N^- dx dt //////////////////
////////////////////////////////////////////////////////////////////////////////

template<int dim>
void
Heat_cG_DWR<dim>::
dual_assemble_Je_mean_final() {
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	Assert(rit_In_grid->dual.fe.use_count(), dealii::ExcNotInitialized());
	Assert(rit_In_grid->dual.mapping.use_count(), dealii::ExcNotInitialized());
	Assert(rit_In_grid->dual.constraints.use_count(), dealii::ExcNotInitialized());
	
	dual.Je = 0;

 	dealii::QGauss<dim> quad (rit_In_grid->dual.fe->tensor_degree()+2);
	
	// Setup a FE_Values object.
	dealii::FEValues<dim> fe_values(
		*(rit_In_grid->dual.mapping),
		*(rit_In_grid->dual.fe),
		quad,
		dealii::update_values |
		dealii::update_gradients |
		dealii::update_quadrature_points |
		dealii::update_JxW_values);
	
	dealii::Vector<double> local_dual_Je (
		rit_In_grid->dual.fe->dofs_per_cell
	);
	
	// Setup a small vector, to store the global dof indices.
	// NOTE: We are using a C++ standart template vector, not a deal.II
	// "Vector".
	// The FiniteElement object "fe" will tell us, how much dofs has each cell.
	std::vector< dealii::types::global_dof_index > local_dof_indices(
		rit_In_grid->dual.fe->dofs_per_cell
	);
	
	// Now we do the real work, looping over all cells to compute the cell
	// assemblys. For this we need an iterator (kind of a pointer), which allows
	// us, to iterate over each cell easily.
	// We initialise it with the first active cell of our triangulation.
	auto cell = rit_In_grid->dual.dof->begin_active();
	auto endc = rit_In_grid->dual.dof->end();
	
	for ( ; cell != endc; ++cell) {
		// First we have to compute the values of the gradients and
		// the JxW values.
		// The reinit of the fe_values object on the current cell will do this.
		fe_values.reinit(cell);
		
		// Initialise the full matrix for the cell assembly with 0.
		local_dual_Je = 0;
		
		// Now loop over all shape function combinations and quadrature points
		// to get the assembly.
		for (unsigned int i(0); i < rit_In_grid->dual.fe->dofs_per_cell; ++i)
		for (unsigned int q(0); q < quad.size(); ++q) {
			local_dual_Je(i) +=
				(fe_values.shape_value(i,q)* 
				fe_values.JxW(q));
		}
		
		// Store the global indices into the vector local_dof_indices.
		// The cell object will give us the information.
		Assert(
			(local_dof_indices.size() == rit_In_grid->dual.fe->dofs_per_cell),
			dealii::ExcNotInitialized()
		);
		cell->get_dof_indices(local_dof_indices);
		
		// Copy the local assembly to the global matrix.
		// We use the constraint object, to set all constraints with that step.
		Assert(rit_In_grid->dual.constraints.use_count(), dealii::ExcNotInitialized());
		
		rit_In_grid->dual.constraints->distribute_local_to_global(
			local_dual_Je, local_dof_indices, dual.Je
		);
		
	}
	
}


////////////////////////////////////////////////////////////////////////////////
//////////  Meanvalue_global-Evaluation (J(phi)=int_I int_Omega phi dx dt //////////////////
////////////////////////////////////////////////////////////////////////////////

template<int dim>
void
Heat_cG_DWR<dim>::
dual_assemble_Je_mean_global() {
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	Assert(rit_In_grid->dual.fe.use_count(), dealii::ExcNotInitialized());
	Assert(rit_In_grid->dual.mapping.use_count(), dealii::ExcNotInitialized());
	Assert(rit_In_grid->dual.constraints.use_count(), dealii::ExcNotInitialized());
	
	dual.Je = 0;
	
 	dealii::QGauss<dim> quad (rit_In_grid->dual.fe->tensor_degree()+2);
	
	// Setup a FE_Values object.
	dealii::FEValues<dim> fe_values(
		*(rit_In_grid->dual.mapping),
		*(rit_In_grid->dual.fe),
		quad,
		dealii::update_values |
		dealii::update_gradients |
		dealii::update_quadrature_points |
		dealii::update_JxW_values);
	
	dealii::Vector<double> local_dual_Je (
		rit_In_grid->dual.fe->dofs_per_cell
	);
	
	// Setup a small vector, to store the global dof indices.
	// NOTE: We are using a C++ standart template vector, not a deal.II
	// "Vector".
	// The FiniteElement object "fe" will tell us, how much dofs has each cell.
	std::vector< dealii::types::global_dof_index > local_dof_indices(
		rit_In_grid->dual.fe->dofs_per_cell
	);
	
	// Now we do the real work, looping over all cells to compute the cell
	// assemblys. For this we need an iterator (kind of a pointer), which allows
	// us, to iterate over each cell easily.
	// We initialise it with the first active cell of our triangulation.
	auto cell = rit_In_grid->dual.dof->begin_active();
	auto endc = rit_In_grid->dual.dof->end();
	
	for ( ; cell != endc; ++cell) {
		// First we have to compute the values of the gradients and
		// the JxW values.
		// The reinit of the fe_values object on the current cell will do this.
		fe_values.reinit(cell);
		
		// Initialise the full matrix for the cell assembly with 0.
		local_dual_Je = 0;
		
		// Now loop over all shape function combinations and quadrature points
		// to get the assembly.
		for (unsigned int i(0); i < rit_In_grid->dual.fe->dofs_per_cell; ++i)
		for (unsigned int q(0); q < quad.size(); ++q) {
			local_dual_Je(i) +=
				(fe_values.shape_value(i,q)*
				fe_values.JxW(q));
		}
		
		// Store the global indices into the vector local_dof_indices.
		// The cell object will give us the information.
		Assert(
			(local_dof_indices.size() == rit_In_grid->dual.fe->dofs_per_cell),
			dealii::ExcNotInitialized()
		);
		cell->get_dof_indices(local_dof_indices);
		
		// Copy the local assembly to the global matrix.
		// We use the constraint object, to set all constraints with that step.
		Assert(rit_In_grid->dual.constraints.use_count(), dealii::ExcNotInitialized());
		
		rit_In_grid->dual.constraints->distribute_local_to_global(
			local_dual_Je, local_dof_indices, dual.Je
		);
		
	}
	
}


////////////////////////////////////////////////////////////////////////////////
//////////  Pointvalue_final-Evaluation (J(phi)= delta_x_0 delta_t_N (phi) /////
////////////////////////////////////////////////////////////////////////////////

template<int dim>
void
Heat_cG_DWR<dim>::
dual_assemble_Je_point_final() {
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	Assert(rit_In_grid->dual.fe.use_count(), dealii::ExcNotInitialized());
	Assert(rit_In_grid->dual.mapping.use_count(), dealii::ExcNotInitialized());
	Assert(rit_In_grid->dual.constraints.use_count(), dealii::ExcNotInitialized());
	
	dual.Je = 0;

	// Now we do the real work, looping over all cells to compute the cell
	// assemblys. For this we need an iterator (kind of a pointer), which allows
	// us, to iterate over each cell easily.
	// We initialise it with the first active cell of our triangulation.
	auto cell = rit_In_grid->dual.dof->begin_active();
	auto endc = rit_In_grid->dual.dof->end();
	
	for ( ; cell != endc; ++cell) {
		for (unsigned int vertex = 0;
			 vertex < dealii::GeometryInfo<dim>::vertices_per_cell;
			 ++vertex) {
			if (cell->vertex(vertex).distance(dual.evaluation_point) < cell->diameter()*1e-8) {
				dual.Je(cell->vertex_dof_index(vertex,0)) = 1;
				return;
			} // end if-loop 
		} // end for-loop vertex
	} // end for-loop cell
}


template<int dim>
void
Heat_cG_DWR<dim>::
dual_interpolate_to_next_grid() {
	dealii::VectorTools::interpolate_to_different_mesh(
		*(rit_In_grid_previous->dual.dof),
		*(dual.z_old),
		*(rit_In_grid->dual.dof),
		*(rit_In_grid->dual.constraints),
		*(dual.z_old_interpolated)
	);
	
	// Only needed if within dual_assemble_Je has to be sth computed because of 
	// the used error functional, for exmpl L^2error at final timepoint dual.Je 
	// is always 0.
	dealii::VectorTools::interpolate_to_different_mesh(
		*(rit_In_grid_previous->dual.dof),
		dual.Je_old,
		*(rit_In_grid->dual.dof),
		*(rit_In_grid->dual.constraints),
		dual.Je_old_interpolated
	);
}


template<int dim>
void
Heat_cG_DWR<dim>::
dual_assemble_rhs_L2final() {
	dual.system_rhs = 0;
	(dual.A).vmult(dual.system_rhs,*(dual.z_old_interpolated));
	dual.system_rhs *= (-((data.tau_n)/2.));
	(dual.M).vmult_add(dual.system_rhs,*(dual.z_old_interpolated));
}


template<int dim>
void
Heat_cG_DWR<dim>::
dual_assemble_rhs_L2global() {
	dual.system_rhs = 0;
	(dual.A).vmult(dual.system_rhs,*(dual.z_old_interpolated));
	dual.system_rhs *= (-((data.tau_n)/2.));
	(dual.M).vmult_add(dual.system_rhs,*(dual.z_old_interpolated));
	(dual.system_rhs).add(((data.tau_n)/2.),dual.Je_old_interpolated);
	(dual.system_rhs).add(((data.tau_n)/2.),dual.Je);
}


template<int dim>
void
Heat_cG_DWR<dim>::
dual_assemble_rhs_mean_final() {
	dual.system_rhs = 0;
	(dual.A).vmult(dual.system_rhs,*(dual.z_old_interpolated));
	dual.system_rhs *= (-((data.tau_n)/2.));
	(dual.M).vmult_add(dual.system_rhs,*(dual.z_old_interpolated));
}


template<int dim>
void
Heat_cG_DWR<dim>::
dual_assemble_rhs_mean_global() {
	dual.system_rhs = 0;
	(dual.A).vmult(dual.system_rhs,*(dual.z_old_interpolated));
	dual.system_rhs *= (-((data.tau_n)/2.));
	(dual.M).vmult_add(dual.system_rhs,*(dual.z_old_interpolated));
	(dual.system_rhs).add(((data.tau_n)/2.),dual.Je_old_interpolated);
	(dual.system_rhs).add(((data.tau_n)/2.),dual.Je);
}


template<int dim>
void
Heat_cG_DWR<dim>::
dual_assemble_rhs_point_final() {
	dual.system_rhs = 0;
	(dual.A).vmult(dual.system_rhs,*(dual.z_old_interpolated));
	dual.system_rhs *= (-((data.tau_n)/2.));
	(dual.M).vmult_add(dual.system_rhs,*(dual.z_old_interpolated));
}


template<int dim>
void
Heat_cG_DWR<dim>::
dual_solve() {
	////////////////////////////////////////////////////////////////////////////
	// apply Dirichlet boundary values
	std::map<dealii::types::global_dof_index, double> boundary_values;
	dealii::VectorTools::interpolate_boundary_values(
		*(rit_In_grid->dual.dof),
		static_cast< dealii::types::boundary_id > (
			Heat::types::boundary_id::Dirichlet
		),
		dealii::ZeroFunction<dim> (),
		boundary_values
	);
	
	dealii::MatrixTools::apply_boundary_values(
		boundary_values,
		dual.system_matrix,
		*(dual.z),
		dual.system_rhs
	);
	
	////////////////////////////////////////////////////////////////////////////
	// solve linear system
	dealii::SparseDirectUMFPACK iA;
	iA.initialize(dual.system_matrix);
	iA.vmult(*(dual.z), dual.system_rhs);
	
	////////////////////////////////////////////////////////////////////////////
	// distribute hanging node constraints on solution
	rit_In_grid->dual.constraints->distribute(*(dual.z));
}


template<int dim>
void
Heat_cG_DWR<dim>::
dual_process_solution(const unsigned int cycle) {
	dealii::Vector<double> difference_per_cell (rit_In_grid->tria->n_active_cells());
	const dealii::QTrapez<1> q_trapez;
	const dealii::QIterated<dim> q_iterated (q_trapez,20);
	
	dealii::VectorTools::integrate_difference (*(rit_In_grid->dual.mapping),
												*(rit_In_grid->dual.dof),
												*(dual.z),
												*(BoundaryValues_dual),
												difference_per_cell,
												q_iterated, //dealii::QGauss<dim>(6),// q_iterated (alternativ)
												dealii::VectorTools::L2_norm);
	const double L2_error = difference_per_cell.l2_norm();
	
	dealii::VectorTools::integrate_difference (*(rit_In_grid->dual.mapping),
												*(rit_In_grid->dual.dof),
												*(dual.z),
												*(BoundaryValues_dual),
												difference_per_cell,
												dealii::QGauss<dim>(6),
												dealii::VectorTools::H1_seminorm);
	const double H1_error = difference_per_cell.l2_norm();
	
	
	const unsigned int n_active_cells=rit_In_grid->tria->n_active_cells();
	const unsigned int n_dofs=rit_In_grid->dual.dof->n_dofs();
	
	std::cout 	<< "Cycle " << cycle << ':'
				<< std::endl
				<< " Number of active cells: "
				<< n_active_cells
				<< std::endl
				<< " Number of degrees of freedom: "
				<< n_dofs
				<< std::endl;
				
	dual.convergence_table.add_value("cycle",cycle);
	dual.convergence_table.add_value("cells",n_active_cells);
	dual.convergence_table.add_value("dofs",n_dofs);
	dual.convergence_table.add_value("tau_n",data.tau_n);
	dual.convergence_table.add_value("L2",L2_error);
	dual.convergence_table.add_value("H1",H1_error);	
}


template<int dim>
void
Heat_cG_DWR<dim>::
dual_do_data_output(const double n) {
	const double solution_time = n;
	
	dual.data_output.write_data("dual", dual.z, solution_time);
}


template<int dim>
void
Heat_cG_DWR<dim>::
do_data_output(const double cycle) {
	const double solution_time = cycle;
	
	primal.data_output.write_data("primal", primal.u, solution_time);
	dual.data_output.write_data("dual", dual.solution_vectors, solution_time);
}


template<int dim>
void
Heat_cG_DWR<dim>::
compute_Ieff() {
	switch (dual.Je_type) {
		case Heat::types::error_functional::forbidden:
			AssertThrow(false, dealii::ExcMessage("You need to initialise dual.Je_type."));
			break;
		
		case Heat::types::error_functional::L2_final:
			compute_Ieff_L2final();
			break;
			
		case Heat::types::error_functional::L2_global:
			compute_Ieff_L2global();
			break;
			
		case Heat::types::error_functional::mean_final:
			compute_Ieff_mean_final();
			break;
			
		case Heat::types::error_functional::mean_global:
			compute_Ieff_mean_global();
			break;
			
		case Heat::types::error_functional::point:
			compute_Ieff_point_final();
			break;
		
		default:
			AssertThrow(false, dealii::ExcMessage("Your dual.Je_type is unknown, please check your inputs."));
	}	
}


template<int dim>
void
Heat_cG_DWR<dim>::
compute_Ieff_L2final() {

	// Compute eta_K^n within fct. estimate() of class DWR_ErrorEstimator and store
	// the local contributions of one time-interval I_n within the vector erroro_indicators
	// and all these vectors within the list In_eta.
	error_estimator.DWR->estimate(
		In_u,
		In_z,
		In_eta
	);
	
	// Sum up the local contributions of all cells within one time-interval of the 
	// local vector error_indicators (sum_(K in T_h)n_K^n). Then sum up all these
	// added values and store it within the double eta (=sum_(n=1)^N (sum_(K in T_h)n_K^n))
	auto In_eta_test(In_eta->begin());
	auto endIn_eta_test(In_eta->end());
	// Variable for estimated error eta
	double eta;
	eta = 0;
	for (; In_eta_test != endIn_eta_test; ++In_eta_test) {
// 		for (unsigned int i=0; i < In_etat_test->x->size(); ++i) {
// 			(*(In_etat_test->x))[i] = std::fabs((*(In_etat_test->x))[i]);
// 		}

		// local sum for each time-interval I_n
		double eta_k;
		eta_k = 0;
		eta_k = std::accumulate (In_eta_test->x->begin(),
                                In_eta_test->x->end(), 0.);
		std::cout << "Groesse eta = " << In_eta_test->x->size() << std::endl;
		std::cout << "Summe eta_K = "<< eta_k << std::endl;
		// Add the local contribution of one time-interval I_n to the global eta.
		eta += eta_k;
	} // end of loop In_eta_test
	// Variable for global exact error J(e) and computation of it
	double L2Error_final;
	L2Error_final = 0;
	// Computaion of global exact error J(e)
	dealii::Vector<double> difference_per_cell (grid->slabs.back().tria->n_active_cells());
	const dealii::QTrapez<1> q_trapez;
	const dealii::QIterated<dim> q_iterated (q_trapez,20);
	BoundaryValues_dual->set_time(data.T);
	dealii::VectorTools::integrate_difference(
		*(grid->slabs.back().primal.mapping),
		*(grid->slabs.back().primal.dof),
		*(In_uprimal->back().x),
		*(BoundaryValues_dual),
		difference_per_cell,
		q_iterated,//dealii::QGauss<dim>(4),
		dealii::VectorTools::L2_norm
	);	
	L2Error_final = difference_per_cell.l2_norm();
// 	L2Error_global = L2Error;
	// Variable for effectivity index I_eff
	double I_eff;
	I_eff = 0.;
	// Computaion of effectivity index:
	I_eff = eta/L2Error_final;
	// Output of estimated error eta, exact error J(e) and effectivity index I_eff:
	std::cout << "eta = " << eta << std::endl;
	std::cout << "L2-Fehler final = " << L2Error_final << std::endl;
	std::cout << "I_eff = " << eta/L2Error_final << std::endl;
	
	// Add value of I_eff to table
	primal.convergence_table.add_value("J(e)",L2Error_final);
	primal.convergence_table.add_value("Eta",eta);
	primal.convergence_table.add_value("I_eff", I_eff);
}


template<int dim>
void
Heat_cG_DWR<dim>::
compute_Ieff_L2global() {
	// Compute eta_K^n within fct. estimate() of class DWR_ErrorEstimator ans store
	// the local contributions of one time-interval I_n within the vector erroro_indicators
	// and all these vectors within the list In_eta.
	error_estimator.DWR->estimate(
		In_u,
		In_z,
		In_eta
	);

	////////////////////////////////////////////////////////////////////////////
	////// Computaion of I_eff  ////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////
	
	// Sum up the local contributions of all cells within one time-interval of the 
	// local vector error_indicators (sum_(K in T_h)n_K^n). Then sum up all these
	// added values and store it within the double eta (=sum_(n=1)^N (sum_(K in T_h)n_K^n))
	auto In_eta_test(In_eta->begin());
	auto endIn_eta_test(In_eta->end());
	// Variable for estimated error eta
	double eta;
	eta = 0;
	for (; In_eta_test != endIn_eta_test; ++In_eta_test) {
// 		for (unsigned int i=0; i < In_etat_test->x->size(); ++i) {
// 			(*(In_etat_test->x))[i] = std::fabs((*(In_etat_test->x))[i]);
// 		}

		std::cout << "Groesse eta = " << In_eta_test->x->size() << std::endl;
		std::cout << "Summe eta_K = "<< std::accumulate (In_eta_test->x->begin(),
                                In_eta_test->x->end(), 0.) << std::endl;
		// local sum for each time-interval I_n
		double eta_k;
		eta_k = 0;
		eta_k = std::accumulate (In_eta_test->x->begin(),
                                In_eta_test->x->end(), 0.);
		// Add the local contribution of one time-interval I_n to the global eta.
		eta += eta_k;
	} // end of loop In_eta_test
	
// 	// Computation of the exact error J(e) = J(u)-J(u_kh):
	// In now done within the function compute_global_STL2_error() (see below)
	
// 	// Variable for effectivity index I_eff
	double I_eff;
	I_eff = 0.;
	// Computaion of effectivity index:
	I_eff = eta/L2Error_global;
	// Output of estimated error eta, exact error J(e) and effectivity index I_eff:
	std::cout << "eta = " << eta << std::endl;
	std::cout << "gl Fehler = " << L2Error_global << std::endl;
	std::cout << "I_eff = " << eta/L2Error_global << std::endl;
	
	// Add value of I_eff to table
	primal.convergence_table.add_value("J(e)",L2Error_global);
	primal.convergence_table.add_value("Eta",eta);
	primal.convergence_table.add_value("I_eff", I_eff);

}


template<int dim>
void
Heat_cG_DWR<dim>::
compute_Ieff_mean_final() {
	// Compute eta_K^n within fct. estimate() of class DWR_ErrorEstimator ans store
	// the local contributions of one time-interval I_n within the vector erroro_indicators
	// and all these vectors within the list In_eta.
	error_estimator.DWR->estimate(
		In_u,
		In_z,
		In_eta
	);
	
	// Sum up the local contributions of all cells within one time-interval of the 
	// local vector error_indicators (sum_(K in T_h)n_K^n). Then sum up all these
	// added values and store it within the double eta (=sum_(n=1)^N (sum_(K in T_h)n_K^n))
	auto In_eta_test(In_eta->begin());
	auto endIn_eta_test(In_eta->end());
	// Variable for estimated error eta
	double eta;
	eta = 0.;
	for (; In_eta_test != endIn_eta_test; ++In_eta_test) {
// 		for (unsigned int i=0; i < In_etat_test->x->size(); ++i) {
// 			(*(In_etat_test->x))[i] = std::fabs((*(In_etat_test->x))[i]);
// 		}

		std::cout << "Groesse eta = " << In_eta_test->x->size() << std::endl;
		std::cout << "Summe eta_K = "<< std::accumulate (In_eta_test->x->begin(),
                                In_eta_test->x->end(), 0.) << std::endl;
		// local sum for each time-interval I_n
		double eta_k;
		eta_k = 0.;
		eta_k = std::accumulate (In_eta_test->x->begin(),
                                In_eta_test->x->end(), 0.);
		// Add the local contribution of one time-interval I_n to the global eta.
		eta += eta_k;
	} // end of loop In_eta_test
	
	// Variable for global exact error J(e) and computation of it
	double MeanValue_final;
	MeanValue_final = 0;
	// Computaion of global exact error J(e)
	dealii::Vector<double> difference_per_cell (grid->slabs.back().tria->n_active_cells());
	const dealii::QTrapez<1> q_trapez;
	const dealii::QIterated<dim> q_iterated (q_trapez,20);
	BoundaryValues_dual->set_time(data.T);
	dealii::VectorTools::integrate_difference(
		*(grid->slabs.back().primal.mapping),
		*(grid->slabs.back().primal.dof),
		*(In_uprimal->back().x),
		*(BoundaryValues_dual),
		difference_per_cell,
		q_iterated,//dealii::QGauss<dim>(4),
		dealii::VectorTools::mean
	);	
	MeanValue_final = std::accumulate(difference_per_cell.begin(),
                                difference_per_cell.end(), 0.);

	
	// Variable for effectivity index I_eff
	double I_eff;
	I_eff = 0.;
	// Computaion of effectivity index:
	I_eff = eta/MeanValue_final;
	// Output of estimated error eta, exact error J(e) and effectivity index I_eff:
	std::cout << "eta = " << eta << std::endl;
	std::cout << "Mittelwert final = " << MeanValue_final << std::endl;
	std::cout << "I_eff = " << eta/MeanValue_final << std::endl;
	
	// Add value of I_eff to table
	primal.convergence_table.add_value("J(e)",MeanValue_final);
	primal.convergence_table.add_value("Eta",eta);
	primal.convergence_table.add_value("I_eff", I_eff);
}


template<int dim>
void
Heat_cG_DWR<dim>::
compute_Ieff_mean_global() {
	// Compute eta_K^n within fct. estimate() of class DWR_ErrorEstimator ans store
	// the local contributions of one time-interval I_n within the vector erroro_indicators
	// and all these vectors within the list In_eta.
	error_estimator.DWR->estimate(
		In_u,
		In_z,
		In_eta
	);
	
	// Sum up the local contributions of all cells within one time-interval of the 
	// local vector error_indicators (sum_(K in T_h)n_K^n). Then sum up all these
	// added values and store it within the double eta (=sum_(n=1)^N (sum_(K in T_h)n_K^n))
	auto In_eta_test(In_eta->begin());
	auto endIn_eta_test(In_eta->end());
	// Variable for estimated error eta
	double eta;
	eta = 0.;
	for (; In_eta_test != endIn_eta_test; ++In_eta_test) {
// 		for (unsigned int i=0; i < In_etat_test->x->size(); ++i) {
// 			(*(In_etat_test->x))[i] = std::fabs((*(In_etat_test->x))[i]);
// 		}

		std::cout << "Groesse eta = " << In_eta_test->x->size() << std::endl;
		std::cout << "Summe eta_K = "<< std::accumulate (In_eta_test->x->begin(),
                                In_eta_test->x->end(), 0.) << std::endl;
		// local sum for each time-interval I_n
		double eta_k;
		eta_k = 0.;
		eta_k = std::accumulate (In_eta_test->x->begin(),
                                In_eta_test->x->end(), 0.);
		// Add the local contribution of one time-interval I_n to the global eta.
		eta += eta_k;
	} // end of loop In_eta_test
	
	auto In_u_error(In_uprimal->begin());
	auto endIn_u_error(In_uprimal->end());
	auto In_error(grid->slabs.begin());
	auto endIn_error(grid->slabs.end());
	// Variable for local L2Error
	double MeanValue_exact;
	MeanValue_exact = 0.;
	const dealii::QTrapez<1> q_trapez;
	const dealii::QIterated<dim> q_iterated (q_trapez,20);
// 	// Vector for global errors on each cell K (E_K in dealii notation)
// 	dealii::Vector<double> global_diff (it_In_grid->tria->n_active_cells());
	// Loop over all time-intervals I_n (n=1,...,N(=number of grids))
	for (unsigned int n{0};n <= grid->slabs.size(); ++n,++In_u_error) {
		
		if (n == grid->slabs.size()) {
			++In_error;
			it_In_grid = In_error;
			BoundaryValues->set_time(n*data.tau_n);
			dealii::Vector<double> difference_per_cell (it_In_grid->tria->n_active_cells());
			dealii::VectorTools::integrate_difference (*(it_In_grid->primal.mapping),
													*(it_In_grid->primal.dof),
													*(In_u_error->x),
													*(BoundaryValues),
													difference_per_cell,
													q_iterated,//dealii::QGauss<dim>(4),//q_iterated, //dealii::QGauss<dim>(6),// q_iterated (alternativ)
													dealii::VectorTools::mean);
			double MeanValue_local = std::accumulate(difference_per_cell.begin(),
                                difference_per_cell.end(), 0.);
			MeanValue_exact += (data.tau_n/2.)*MeanValue_local;
		} // end if (n == grid->slabs.size())
		else if (n == 0) {
			it_In_grid = In_error;
			BoundaryValues->set_time(n*data.tau_n);
			dealii::Vector<double> difference_per_cell (it_In_grid->tria->n_active_cells());
			dealii::VectorTools::integrate_difference (*(it_In_grid->primal.mapping),
													*(it_In_grid->primal.dof),
													*(In_u_error->x),
													*(BoundaryValues),
													difference_per_cell,
													q_iterated,//dealii::QGauss<dim>(4),//q_iterated, //dealii::QGauss<dim>(6),// q_iterated (alternativ)
													dealii::VectorTools::mean);
			double MeanValue_local = std::accumulate(difference_per_cell.begin(),
                                difference_per_cell.end(), 0.);
			MeanValue_exact += (data.tau_n/2.)*MeanValue_local;
		}
		else if (n == 1) {
			it_In_grid = In_error;
			BoundaryValues->set_time(n*data.tau_n);
			dealii::Vector<double> difference_per_cell (it_In_grid->tria->n_active_cells());
			dealii::VectorTools::integrate_difference (*(it_In_grid->primal.mapping),
													*(it_In_grid->primal.dof),
													*(In_u_error->x),
													*(BoundaryValues),
													difference_per_cell,
													q_iterated,//dealii::QGauss<dim>(4),//q_iterated, //dealii::QGauss<dim>(6),// q_iterated (alternativ)
													dealii::VectorTools::mean);
			double MeanValue_local = std::accumulate(difference_per_cell.begin(),
                                difference_per_cell.end(), 0.);
			MeanValue_exact += (data.tau_n)*MeanValue_local;
		}
		else {
		++In_error;
		it_In_grid = In_error;
			BoundaryValues->set_time(n*data.tau_n);
			dealii::Vector<double> difference_per_cell (it_In_grid->tria->n_active_cells());
			dealii::VectorTools::integrate_difference (*(it_In_grid->primal.mapping),
													*(it_In_grid->primal.dof),
													*(In_u_error->x),
													*(BoundaryValues),
													difference_per_cell,
													q_iterated,//dealii::QGauss<dim>(4),//q_iterated, //dealii::QGauss<dim>(6),// q_iterated (alternativ)
													dealii::VectorTools::mean);
			double MeanValue_local = std::accumulate(difference_per_cell.begin(),
                                difference_per_cell.end(), 0.);
			MeanValue_exact += (data.tau_n)*MeanValue_local;
		}//end else

	} // end of loop n over all time-intervals I_n
	
	double I_eff;
	I_eff = 0.;
	// Computaion of effectivity index:
	I_eff = eta/MeanValue_exact;
	// Output of estimated error eta, exact error J(e) and effectivity index I_eff:
	std::cout << "eta = " << eta << std::endl;
	std::cout << "gl Mittelwert = " << MeanValue_exact << std::endl;
	std::cout << "I_eff = " << eta/MeanValue_exact << std::endl;
	
	// Add value of I_eff to table
	primal.convergence_table.add_value("J(e)",MeanValue_exact);
	primal.convergence_table.add_value("Eta",eta);
	primal.convergence_table.add_value("I_eff", I_eff);
}


template<int dim>
void
Heat_cG_DWR<dim>::
compute_Ieff_point_final() {
	// Compute eta_K^n within fct. estimate() of class DWR_ErrorEstimator ans store
	// the local contributions of one time-interval I_n within the vector erroro_indicators
	// and all these vectors within the list In_eta.
	error_estimator.DWR->estimate(
		In_u,
		In_z,
		In_eta
	);
	
	// Sum up the local contributions of all cells within one time-interval of the 
	// local vector error_indicators (sum_(K in T_h)n_K^n). Then sum up all these
	// added values and store it within the double eta (=sum_(n=1)^N (sum_(K in T_h)n_K^n))
	auto In_eta_test(In_eta->begin());
	auto endIn_eta_test(In_eta->end());
	// Variable for estimated error eta
	double eta;
	eta = 0.;
	for (; In_eta_test != endIn_eta_test; ++In_eta_test) {
// 		for (unsigned int i=0; i < In_etat_test->x->size(); ++i) {
// 			(*(In_etat_test->x))[i] = std::fabs((*(In_etat_test->x))[i]);
// 		}

		std::cout << "Groesse eta = " << In_eta_test->x->size() << std::endl;
		std::cout << "Summe eta_K = "<< std::accumulate (In_eta_test->x->begin(),
                                In_eta_test->x->end(), 0.) << std::endl;
		// local sum for each time-interval I_n
		double eta_k;
		eta_k = 0.;
		eta_k = std::accumulate (In_eta_test->x->begin(),
                                In_eta_test->x->end(), 0.);
		// Add the local contribution of one time-interval I_n to the global eta.
		eta += eta_k;
	} // end of loop In_eta_test
	
	// Variable for exact error J(e) at the final time-point t_N=T and computation of it
	double point_value_difference_final;
	point_value_difference_final = 0;
	// Computaion of exact error J(e)

	BoundaryValues_dual->set_time(data.T);
	dealii::Vector<double> result(1);
	
	dealii::VectorTools::point_difference(
		*(grid->slabs.back().primal.mapping),
		*(grid->slabs.back().primal.dof),
		*(In_uprimal->back().x),
		*(BoundaryValues_dual),
		result,
		dual.evaluation_point
	);	
	
	point_value_difference_final = std::fabs(result(0));

	
	// Variable for effectivity index I_eff
	double I_eff;
	I_eff = 0.;
	// Computaion of effectivity index:
	I_eff = std::fabs(eta/point_value_difference_final);
	// Output of estimated error eta, exact error J(e) and effectivity index I_eff:
	std::cout << "eta = " << eta << std::endl;
	std::cout << "Point value final = " << point_value_difference_final << std::endl;
	std::cout << "I_eff = " << I_eff << std::endl;
	
	// Add value of I_eff to table
	primal.convergence_table.add_value("J(e)",point_value_difference_final);
	primal.convergence_table.add_value("Eta",eta);
	primal.convergence_table.add_value("I_eff", I_eff);
}


template<int dim>
void
Heat_cG_DWR<dim>::
compute_global_STL2_error() {
	auto In_u_error(In_uprimal->begin());
	auto endIn_u_error(In_uprimal->end());
	auto In_error(grid->slabs.begin());
	auto endIn_error(grid->slabs.end());
	// Variable for local L2Error
	double L2Error_local;
	L2Error_local = 0;
// 	// Vector for global errors on each cell K (E_K in dealii notation)
// 	dealii::Vector<double> global_diff (it_In_grid->tria->n_active_cells());
	// Loop over all time-intervals I_n (n=1,...,N(=number of grids))
	for (unsigned int n{0};n <= grid->slabs.size(); ++n,++In_u_error) {
		
		if (n == grid->slabs.size()) {
			++In_error;
			it_In_grid = In_error;
			dealii::Vector<double> difference_per_cell (it_In_grid->tria->n_active_cells());
			const dealii::QTrapez<1> q_trapez;
			const dealii::QIterated<dim> q_iterated (q_trapez,20);
		
			BoundaryValues->set_time(n*data.tau_n);
			dealii::VectorTools::integrate_difference (*(it_In_grid->primal.mapping),
													*(it_In_grid->primal.dof),
													*(In_u_error->x),
													*(BoundaryValues),
													difference_per_cell,
													q_iterated,//dealii::QGauss<dim>(4),//q_iterated, //dealii::QGauss<dim>(6),// q_iterated (alternativ)
													dealii::VectorTools::L2_norm);
			
			double L2_error_error_end = difference_per_cell.l2_norm();
			L2_error_error_end *= L2_error_error_end;
			L2_error_error_end *= (data.tau_n/2.);
			L2Error_local += L2_error_error_end;
		} // end if (n == grid->slabs.size())
		else if (n == 0) {
			it_In_grid = In_error;
			dealii::Vector<double> difference_per_cell (it_In_grid->tria->n_active_cells());
			const dealii::QTrapez<1> q_trapez;
			const dealii::QIterated<dim> q_iterated (q_trapez,20);
		
			BoundaryValues->set_time(n*data.tau_n);
			dealii::VectorTools::integrate_difference (*(it_In_grid->primal.mapping),
													*(it_In_grid->primal.dof),
													*(In_u_error->x),
													*(BoundaryValues),
													difference_per_cell,
													q_iterated,//dealii::QGauss<dim>(4),//q_iterated, //dealii::QGauss<dim>(6),// q_iterated (alternativ)
													dealii::VectorTools::L2_norm);
			
			double L2_error_error_end = difference_per_cell.l2_norm();
			L2_error_error_end *= L2_error_error_end;
			L2_error_error_end *= (data.tau_n/2.);
			L2Error_local += L2_error_error_end;
		}
		else if (n == 1) {
			it_In_grid = In_error;
			dealii::Vector<double> difference_per_cell (it_In_grid->tria->n_active_cells());
			const dealii::QTrapez<1> q_trapez;
			const dealii::QIterated<dim> q_iterated (q_trapez,20);
		
			BoundaryValues->set_time(n*data.tau_n);
			dealii::VectorTools::integrate_difference (*(it_In_grid->primal.mapping),
													*(it_In_grid->primal.dof),
													*(In_u_error->x),
													*(BoundaryValues),
													difference_per_cell,
													q_iterated,//dealii::QGauss<dim>(4),//q_iterated, //dealii::QGauss<dim>(6),// q_iterated (alternativ)
													dealii::VectorTools::L2_norm);
			
			double L2_error_error_end = difference_per_cell.l2_norm();
			L2_error_error_end *= L2_error_error_end;
			L2_error_error_end *= (data.tau_n);
			L2Error_local += L2_error_error_end;
		}
		else {
		++In_error;
		it_In_grid = In_error;
		dealii::Vector<double> difference_per_cell (it_In_grid->tria->n_active_cells());
		const dealii::QTrapez<1> q_trapez;
		const dealii::QIterated<dim> q_iterated (q_trapez,20);
		
		BoundaryValues->set_time(n*data.tau_n);
		dealii::VectorTools::integrate_difference (*(it_In_grid->primal.mapping),
													*(it_In_grid->primal.dof),
													*(In_u_error->x),
													*(BoundaryValues),
													difference_per_cell,
													q_iterated,//dealii::QGauss<dim>(4),//q_iterated, //dealii::QGauss<dim>(6),// q_iterated (alternativ)
													dealii::VectorTools::L2_norm);
		
		double L2_error_error = difference_per_cell.l2_norm();
		L2_error_error *= L2_error_error;
		L2_error_error *= data.tau_n;
		L2Error_local += L2_error_error;
		
		}//end else

	} // end of loop n over all time-intervals I_n
	L2Error_global = std::sqrt(L2Error_local);
	
	////////////////////////////////////////////////////////////////////////////
	/////////////// Rechteckregel //////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////
	
// 	auto In_u_error(In_u->begin());
// 	auto endIn_u_error(In_u->end());
// 	auto In_error(grid->slabs.begin());
// 	auto endIn_error(grid->slabs.end());
// 	Variable for local L2Error
// 	double L2Error_local;
// 	L2Error_local = 0;
// 	Vector for global errors on each cell K (E_K in dealii notation)
// 	dealii::Vector<double> global_diff (it_In_grid->tria->n_active_cells());
// 	Loop over all time-intervals I_n (n=1,...,N(=number of grids))
// 	for (unsigned int n{1};n <= grid->slabs.size(); ++n,++In_error) {
// 		
// 			++In_u_error;
// 			it_In_grid = In_error;
// 			dealii::Vector<double> difference_per_cell (it_In_grid->tria->n_active_cells());
// 			const dealii::QTrapez<1> q_trapez;
// 			const dealii::QIterated<dim> q_iterated (q_trapez,10);
// 		std::cout << "tau_n ist = " << data.tau_n << std::endl;
// 			BoundaryValues->set_time(n*data.tau_n);
// 			dealii::VectorTools::integrate_difference (*(it_In_grid->dual.mapping),
// 													*(it_In_grid->dual.dof),
// 													*(In_u_error->x),
// 													*(BoundaryValues),
// 													difference_per_cell,
// 													q_iterated, //dealii::QGauss<dim>(6),// q_iterated (alternativ)
// 													dealii::VectorTools::L2_norm);
// 			
// 			double L2_error_error_end = difference_per_cell.l2_norm();
// 			L2_error_error_end *= L2_error_error_end;
// 			L2_error_error_end *= (data.tau_n);
// 			L2Error_local += L2_error_error_end;
// 
// 
// 	} // end of loop n over all time-intervals I_n
// 	L2Error_global = std::sqrt(L2Error_local);
}


template<int dim>
void
Heat_cG_DWR<dim>::
refine_grids_dwr() {
// // 	auto error_indicators = std::make_shared< dealii::Vector<double> > ();
// // 	error_indicators->reinit(it_In_grid->tria->n_active_cells());
// // 	
// // 	error_estimator.DWR->estimate(
// // 		In_u,
// // 		In_z,
// // 		In_eta
// // 	);
// // 	
// // 	////////////////////////////////////////////////////////////////////////////
// // 	/// Effectivity index I_eff(:= eta=error_indicators/J(u)-J(u_h) ) check:
// // 	
// // 	/// Compute J(u)-J(u_h)
// // 	
// // 	dealii::Vector<double> difference_per_cell (it_In_grid->tria->n_active_cells());
// // 	const dealii::QTrapez<1> q_trapez;
// // 	const dealii::QIterated<dim> q_iterated (q_trapez,20);
// // 	
// // 	dealii::VectorTools::integrate_difference (
// // 		*(it_In_grid->primal.mapping),
// // 		*(it_In_grid->primal.dof),
// // 		*primal.u,
// // 		*BoundaryValues,
// // 		difference_per_cell,
// // 		q_iterated, //dealii::QGauss<dim> ( it_In_grid->primal.fe->tensor_degree()+1 ),
// // 		dealii::VectorTools::L2_norm);
// // 	const double L2_error = difference_per_cell.l2_norm();
// // 	std::cout << "   L2-Fehler = " << L2_error << std::endl;
// // 	
// // 	/// Compute eta = sum of all error_indicators-entries
// // 	const double eta = std::accumulate (error_indicators->begin(),
// //                                 error_indicators->end(), 0.);
// // 	std::cout  << "   eta = " << eta << std::endl;
// // 	
// // 	/// Compute I_eff
// // 	const double I_eff = (eta/L2_error);
// // 	
// // 	/// Output I_eff on screen
// // 	std::cout << "  ==> I_eff = "<< I_eff << std::endl;
// // 	////////////////////////////////////////////////////////////////////////////
	
	// Build up a loop over all grids:
	auto Inth(grid->slabs.begin());
	auto endInth(grid->slabs.end());
	auto In_etath(In_eta->begin());
	
	for (; Inth != endInth; ++Inth) {
	
		for (unsigned int i=0; i < In_etath->x->size(); ++i) {
			(*(In_etath->x))[i] = std::fabs((*(In_etath->x))[i]);
		}
		
		////////////////////////////////////////////////////////////////////////////
		/// Schwegeler mesh-refinement strategy
		const double theta = 1.2; 										//theta aus (0.25,5).
		double eta_max = 0.;
		for (unsigned int i=0; i < In_etath->x->size(); ++i) {		//gives back the max. entry of error_indicators.
			if((*(In_etath->x))[i] > eta_max) {
				eta_max = (*(In_etath->x))[i];
			}
		}
		std::cout << "   eta_max = " << eta_max << std::endl;
		double eta_ks = std::accumulate (In_etath->x->begin(),
										 In_etath->x->end(), 0.);
		double Zellen = Inth->tria->n_active_cells();					//Number of active cells.

		double mu = theta*(eta_ks/Zellen);								//mu.
		std::cout << "   mu = " << mu << std::endl;
		if(mu > eta_max) {mu=mu/2.;}									//while mu > eta_max: mu:=mu/2.
		int Anz_ZuVerfZellen = 0;
		for (unsigned int i=0; i < In_etath->x->size(); ++i) {		//determines how many cells schould be refined
			if((*(In_etath->x))[i] > mu) {							//(those cells, whose related error_indicators entry
				Anz_ZuVerfZellen = Anz_ZuVerfZellen + 1.;				//is > than mu).
			}
		}
		std::cout << "Anz_ZuVerfZellen=" << Anz_ZuVerfZellen << std::endl;
		double topfractioncells = Anz_ZuVerfZellen/Zellen; 				//percentage of to be refined cells.
		std::cout << "topfractioncells=" << topfractioncells << std::endl;
		dealii::GridRefinement::refine_and_coarsen_fixed_number (*(Inth->tria),	//mark cells, which schould be refined
																*(In_etath->x),
																topfractioncells, 0.0);
		
		
		
		////////////////////////////////////////////////////////////////////////////
		/// dealii step14 mesh refinement strategy
		
	// 	dealii::GridRefinement::refine_and_coarsen_fixed_fraction(
	// 		*(Inth->tria),
	// 		*(In_etath->x),
	// 		0.8, 0.02
	// 	);

// 		Inth->tria->execute_coarsening_and_refinement();				// refine grid at the above assumptions.
		
	////////////////////////////////////////////////////////////////////////////////
	////// If you want to compare the results to those aimed by global refinement //
	////////////////////////////////////////////////////////////////////////////////
		
		Inth->tria->refine_global();
	
	++In_etath;
	} // end for-loop Inth (over all grids)
}


template<int dim>
void
Heat_cG_DWR<dim>::
primal_and_dual_solution_output() {
	auto Inth_u_output(In_uprimal->begin());
	auto Inth_z_output(In_z->begin());
	auto In_grid_output(grid->slabs.begin());
	for (unsigned int n{0}; n <= (data.T-data.t0)/data.tau_n; ++n,++Inth_u_output,++Inth_z_output) {
		if (n == 0) {
			it_In_grid = In_grid_output;
			primal_init_data_output();
			dual_init_data_output();
			primal.data_output.write_data("primal", (Inth_u_output->x), n);
			dual.data_output.write_data("dual", (Inth_z_output->x), n);
		}
		else {
			it_In_grid = In_grid_output;
			primal_init_data_output();
			dual_init_data_output();
			primal.data_output.write_data("primal", Inth_u_output->x, n);
			dual.data_output.write_data("dual", Inth_z_output->x, n);
			++In_grid_output;
		}
	}
}


template<int dim>
void
Heat_cG_DWR<dim>::
solve_primal_problem() {
	auto Inth_primal(grid->slabs.begin());
	auto endIn_primal(grid->slabs.end());
	auto Inth_primal_prev(grid->slabs.begin());
	auto endIn_primal_prev(grid->slabs.end());
	auto In_uth(In_u->begin());
	auto endIn_uth(In_u->end());
	auto In_uthprimal(In_uprimal->begin());
	auto endIn_uthprimal(In_uprimal->end());

	for (unsigned int n{0}; n <= ((data.T-data.t0)/data.tau_n); ++n) {
		if (n == 0) {
			// Compute initial condition u_0 and store it in primal.u_old,
			// interpolate it to dual FE-room and store it in dual.u and last
			// but not least store this in the first element of list In_u.
			
			// Output of time_step 1 at time-point t_0
			data.primal_time = (n*data.tau_n); 
			++data.primal_timestep_number;
			std::cout << "Time step " << data.primal_timestep_number << " at t = "
					<< data.primal_time << std::endl;
			// Set iterators
			it_In_grid = Inth_primal;
			it_In_grid_previous = Inth_primal_prev; //needed for primal_interpolate_to_next_grid
			
			std::cout << "Zellen it = " << it_In_grid->tria->n_active_cells() << std::endl;
			std::cout << "Zellen it_prev = " << it_In_grid_previous->tria->n_active_cells() << std::endl;
			
			// Set time to t0 and compute u_0:
			primal_set_time(data.t0);
			primal_compute_initial_condition();

			// Store initial condition u_0, interpolated in dual FE room (dual.u) 
			// in the first element of list In_u
			In_u->front().x->reinit(grid->slabs.front().dual.dof->n_dofs());
			*(In_u->front().x) = *(dual.u);
			// Store initial condition (primal.u_old) in the first element of list In-uprimal
			In_uprimal->front().x->reinit(grid->slabs.front().primal.dof->n_dofs());
			*(In_uprimal->front().x) = *(primal.u_old);
			
		}
		else if (n == 1) {
			// Compute u_1 on same grid as u_0.

			// Output of time_step 2 at time-point t_1
			data.primal_time = (n*data.tau_n); 
			++data.primal_timestep_number;
			std::cout << "Time step " << data.primal_timestep_number << " at t = "
					<< data.primal_time << std::endl;

			++In_uth; //increase iterator of list In_u
			++In_uthprimal; //increase iterator of list In_uprimal
			// Set iterators
			it_In_grid = Inth_primal;
			it_In_grid_previous = Inth_primal_prev; //for this timestep it_In_grid=it_In_grid_previous holds
			
			std::cout << "Zellen it = " << it_In_grid->tria->n_active_cells() << std::endl;
			std::cout << "Zellen it_prev = " << it_In_grid_previous->tria->n_active_cells() << std::endl;
			
			primal_reinit();
			primal_assemble_system();
			primal_set_time(data.primal_time);
// 				std::cout << "new time = " << (n*data.tau_n)-(data.tau_n/2.) << std::endl;
// 				function.f->set_time((n*data.tau_n)-(data.tau_n/2.));//TEST MPR statt rechtsseitige Boxregel
			primal_assemble_f();
			primal_interpolate_to_next_grid();
			primal_assemble_rhs();
			primal_solve();
// 				primal.M.print(std::cout);
// 				std::cout << "" << std::endl;
			//Save current solution for next time step in primal.u_old 
			// (needed within primal_assemble_rhs() of next timestep)
			primal.u_old = std::make_shared< dealii::Vector<double> > ();
			primal.u_old->reinit(it_In_grid->primal.dof->n_dofs());
			*(primal.u_old) = *(primal.u);
			// Save current  solution in list In_uprimal
			In_uthprimal->x->reinit(it_In_grid->primal.dof->n_dofs());
			*(In_uthprimal->x) = *(primal.u);
			
			// interpolate current solution to dual fe-room
			interpolate_primal_to_dual();

			// Save current interpolated solution in list In_u
			In_uth->x->reinit(it_In_grid->dual.dof->n_dofs());
			*(In_uth->x) = *(dual.u);

		}
		else {
			// Compute u_2,...,u_N

			// Output of time_steps 3,...,N+1 at time-points t_2,...,t_N
			data.primal_time = (n*data.tau_n); 
			++data.primal_timestep_number;
			std::cout << "Time step " << data.primal_timestep_number << " at t = "
					<< data.primal_time << std::endl;
			++In_uth; //increase iterator of list In_u
			++In_uthprimal; //increase iterator of list In_uprimal
			++Inth_primal; //increase iterator of list In (grids)
			// Set iterators
			// For t_n, n > 1 it_In_grid_previous should always be the iterator
			// that points on the previous element of the list, to which
			// the iterator it_In_grid points ;-)
			it_In_grid = Inth_primal;
			it_In_grid_previous = Inth_primal_prev; 
			
			std::cout << "Zellen it = " << it_In_grid->tria->n_active_cells() << std::endl;
			std::cout << "Zellen it_prev = " << it_In_grid_previous->tria->n_active_cells() << std::endl;
			
			primal_reinit();
			primal_assemble_system();
			primal_set_time(data.primal_time);
// 				std::cout << "new time = " << (n*data.tau_n)-(data.tau_n/2.) << std::endl;
// 				function.f->set_time((n*data.tau_n)-(data.tau_n/2.));//TEST MPR statt rechtsseitige Boxregel
			primal_assemble_f();
			primal_interpolate_to_next_grid();
			primal_assemble_rhs();
			primal_solve();
			
			//Save current solution for next time step in primal.u_old 
			// (needed within primal_assemble_rhs() of next timestep)
			primal.u_old = std::make_shared< dealii::Vector<double> > ();
			primal.u_old->reinit(it_In_grid->primal.dof->n_dofs());
			*(primal.u_old) = *(primal.u);
			// Save current  solution in list In_uprimal
			In_uthprimal->x->reinit(it_In_grid->primal.dof->n_dofs());
			*(In_uthprimal->x) = *(primal.u);

			// interpolate current solution to dual fe-room
			interpolate_primal_to_dual();

			// Save current interpolated solution (dual.u) in list In_u
			In_uth->x->reinit(it_In_grid->dual.dof->n_dofs());
			*(In_uth->x) = *(dual.u);

			++Inth_primal_prev; // increase iterator of list "In-1" 
		}

	} // end primal-loop n
}


template<int dim>
void
Heat_cG_DWR<dim>::
solve_dual_problem() {
	auto Inth_dual(grid->slabs.rbegin());
	auto endIn_dual(grid->slabs.rend());
	auto Inth_dual_prev(grid->slabs.rbegin());
	auto endIn_dual_prev(grid->slabs.rend());
	auto In_zth(In_z->rbegin());
	auto endIn_zth(In_z->rend());
	auto In_uth_test(In_u->rbegin());
	auto endIn_uth_test(In_u->rend());
	
	for (unsigned int n = ((data.T-data.t0)/data.tau_n); n >= 0 ; --n) {
		if (n == ((data.T-data.t0)/data.tau_n)) {
			// Compute "initial condition" z_N and store it in dual.z_old and
			// afterwards store this in the last element of list In_z (recognize
			// that the list-iterator of In_z ist starting at the last element (In_z->rbegin())
			
			// Set iterator for primal solution u_kh (needed only for global L2-Error)
			rit_In_uback = In_uth_test;
			// Output of time_step 1 at time-point t_N
			data.dual_time = n*data.tau_n;
			++data.dual_timestep_number;
			std::cout << "Time step " << data.dual_timestep_number << " at t = "
						<< data.dual_time << std::endl;
			// Set iterators
			rit_In_grid = Inth_dual;
			rit_In_grid_previous = Inth_dual_prev;
			std::cout << "Zellen it = " << rit_In_grid->tria->n_active_cells() << std::endl;
			std::cout << "Zellen it_prev = " << rit_In_grid_previous->tria->n_active_cells() << std::endl;
			// Initialize rhs_vectors dual.Je_old and dual.Je
			dual.Je_old.reinit(rit_In_grid->dual.dof->n_dofs());
			// Set time to t_N and compute z_N (at timepoint t_N = 0.5)
			dual_set_time(data.T);
			
			dual_compute_initial_condition();

			// Store initial condition z_N (dual.z_old) in the last element of list In_z
			In_z->back().x->reinit(grid->slabs.back().dual.dof->n_dofs());
			*(In_z->back().x) = *(dual.z_old);
			
		}
		else if (n == ((data.T-data.t0)/data.tau_n)-1) {
			// Compute z_N-1 at time point t_N-1
			
			// Output of time_step N+1 at time-point t_0
			data.dual_time = n*data.tau_n;
			++data.dual_timestep_number;
			std::cout << "Time step " << data.dual_timestep_number << " at t = "
						<< data.dual_time << std::endl;
			// Increase and set iterator for primal solution u_kh (needed only for global L2-Error)
			++In_uth_test;
			rit_In_uback = In_uth_test;

			++In_zth; // "increase" iterator of list In_z. (running backward from last to first element)
			++Inth_dual; // "increase" iterator of list In (grids).
			// Set iterators
			rit_In_grid = Inth_dual;
			rit_In_grid_previous = Inth_dual_prev;
			
			std::cout << "Zellen it = " << rit_In_grid->tria->n_active_cells() << std::endl;
			std::cout << "Zellen it_prev = " << rit_In_grid_previous->tria->n_active_cells() << std::endl;
			
			dual_reinit();
			dual_assemble_system();
			dual_set_time(data.dual_time);
			
			dual_assemble_rhs_at_t_Nminus1();

			dual_solve();

			// Save current solution (dual.z) in list In_z;
			In_zth->x->reinit(rit_In_grid->dual.dof->n_dofs());
			*(In_zth->x) = *(dual.z);
			
			// Save current solution (dual.z) and rhs_value (dual.Je) for 
			// next time step in dual.z_old and dual.Je_old
			// (needed within dual_assemble_rhs() of next time step) 
			dual.z_old = std::make_shared< dealii::Vector<double> > ();
			dual.z_old->reinit(rit_In_grid->dual.dof->n_dofs());
			*(dual.z_old) = *(dual.z);
			dual.Je_old.reinit(rit_In_grid->dual.dof->n_dofs());
			dual.Je_old = dual.Je;
			
			++Inth_dual_prev;
		}
		else if (n == 0) {
			// Compute z_0 at time point t_0

			// Output of time_step N+1 at time-point t_0
			data.dual_time = n*data.tau_n;
			++data.dual_timestep_number;
			std::cout << "Time step " << data.dual_timestep_number << " at t = "
						<< data.dual_time << std::endl;
			// Increase and set iterator for primal solution u_kh (needed only for global L2-Error)
			++In_uth_test;
			rit_In_uback = In_uth_test;
			++In_zth; // increase iterator of list In_z, points now on first element of list In-Z
			// Set iterators
			rit_In_grid = Inth_dual;
			rit_In_grid_previous = Inth_dual_prev;
			
			std::cout << "Zellen it = " << rit_In_grid->tria->n_active_cells() << std::endl;
			std::cout << "Zellen it_prev = " << rit_In_grid_previous->tria->n_active_cells() << std::endl;
			
			dual_reinit();
			dual_assemble_system();
			dual_set_time(data.dual_time);
			
			dual_assemble_rhs();
			
			dual_solve();
			
			// Store z_0 (dual.z) in the first element of list In_z.
			In_zth->x->reinit(rit_In_grid->dual.dof->n_dofs());
			*(In_zth->x) = *(dual.z);
			
			break;
		}
		else {
			// Compute z_N-2,...,z_1
			
			// Output of time_steps 2,...,N at time-points t_N-1,...,t_1
			data.dual_time = n*data.tau_n;
			++data.dual_timestep_number;
			std::cout << "Time step " << data.dual_timestep_number << " at t = "
						<< data.dual_time << std::endl;
			// Increase and set iterator for primal solution u_kh (needed only for global L2-Error)
			++In_uth_test;
			rit_In_uback = In_uth_test;

			++In_zth; // "increase" iterator of list In_z. (running backward from last to first element)
			++Inth_dual; // "increase" iterator of list In (grids).
			// Set iterators
			rit_In_grid = Inth_dual;
			rit_In_grid_previous = Inth_dual_prev;
			
			std::cout << "Zellen it = " << rit_In_grid->tria->n_active_cells() << std::endl;
			std::cout << "Zellen it_prev = " << rit_In_grid_previous->tria->n_active_cells() << std::endl;
			
			dual_reinit();
			dual_assemble_system();
			dual_set_time(data.dual_time);

			dual_assemble_rhs();
			
			dual_solve();
			
			// Save current solution (dual.z) in list In_z
			In_zth->x->reinit(rit_In_grid->dual.dof->n_dofs());
			*(In_zth->x) = *(dual.z);
			
			// Save current solution (dual.z) and rhs_value (dual.Je) for 
			// next time step in dual.z_old and dual.Je_old
			// (needed within dual_assemble_rhs() of next time step) 
			dual.z_old = std::make_shared< dealii::Vector<double> > ();
			dual.z_old->reinit(rit_In_grid->dual.dof->n_dofs());
			*(dual.z_old) = *(dual.z);
			dual.Je_old.reinit(rit_In_grid->dual.dof->n_dofs());
			dual.Je_old = dual.Je;
			
			++Inth_dual_prev;
		}
	} //end for-loop n dual
}


template<int dim>
void
Heat_cG_DWR<dim>::
run() {
////////////////////////////////////////////////////////////////////////////////
////////// Creating Grids and Lists ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
	
	// Initialize list of grids, set objects for ErrorEstimator class 
	init(data.global_refinement); //TODO 160 //Für Konvergenztest in der Zeit
													// einfach data.time_steps durch 
													// bspsw 160 (max Anz an Zeitschritten) ersetzen
	std::cout << "n von In = " << grid->slabs.size() << std::endl;
	
	//TEST iterator it_In_grid_previous
// 	auto Inth_test(grid->slabs.begin());
// 	auto endIn_test(grid->slabs.end());
// 	for (unsigned int i{1}; i <= 5; ++i,++Inth_test) {
// 		Inth_test->tria->refine_global(i);
// 		std::cout << "Zellen = " << Inth_test->tria->n_active_cells() << std::endl;
// 	}
	//TEST ENDE iterator it_In_grid_previous
	
// 	//TEST BEGIN
// 	grid->slabs.front().tria->refine_global(1);
// 	grid->slabs.push_back(grid->slabs.front());
// 	auto Inth(grid->slabs.begin());
// 	auto endIn(grid->slabs.end());
// 	for (unsigned int n{1}; n <= 25; ++n) {
// 	grid->slabs.push_back(grid->slabs.front());
// 	}

// 	std::cout << "n von In = " << grid->slabs.size() << std::endl;
// 	//TEST END


	// Prepare storage for solutions vectors of primal and dual problem
	init_storage();
	
////////////////////////////////////////////////////////////////////////////////
/////////////// DWR-loop ///////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

	// Start DWR-loop cycle =^ one DWR-step
	const int endcycle = 2;
	for (unsigned int cycle = 0; cycle < endcycle /*true*/; ++cycle) {
		std::cout << "======================" << std::endl;
		std::cout << "Refinement Cycle: " << cycle << std::endl;
		std::cout << "======================" << std::endl;
		
		// Set time and running variables new at the beginning of every DWR-step
		data.primal_time = 0.;
		data.dual_time = data.T;
		data.primal_timestep_number = 0;
		data.dual_timestep_number = 0;
		
		// Prepare grids and dofs
		grid->set_boundary_indicators();
		grid->distribute();
		
		////////////////////////////////////////////////////////////////////////
		/////////////// begin primal ///////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////
		
		solve_primal_problem();
		
		// For convergence results
		primal_process_solution(cycle);

		////////////////////////////////////////////////////////////////////////
		///////////////////////// end primal ///////////////////////////////////
		////////////////////////////////////////////////////////////////////////


		////////////////////////////////////////////////////////////////////////
		///////////////////////// begin dual ///////////////////////////////////
		////////////////////////////////////////////////////////////////////////
		
		solve_dual_problem();

		// For convergence results
		dual_process_solution(cycle);
		
		////////////////////////////////////////////////////////////////////////
		///////////////////////// end dual /////////////////////////////////////
		////////////////////////////////////////////////////////////////////////
		
		// Compute I_eff:
		std::cout << "In_eta hat Groesse = " << In_eta->size() << std::endl;
	
		compute_Ieff();

		// Solution Output
		if (cycle == endcycle-1) {
			primal_and_dual_solution_output();
		} // end if Soltion Output (cycle == endcycle-1)
		
		// Refine Grids
		refine_grids_dwr();
		
		// 	//TEST
		// 	for (unsigned int n{1}; n <= 75; ++n) {
		// 	grid->slabs.push_back(grid->slabs.front()); // use insert instead of push_back
		// 	}
		// 	std::cout << "Anzahl Gitter = " << grid->slabs.size() << std::endl;
		// 	In_u->clear();
		// 	In_z->clear();
		// 	In_uprimal->clear();
		// 	In_eta->clear();
		// 	// Refine time_mesh global:
		// // 		//TEST gehört zu Zeitkonvergenz
		// 		data.tau_n = (data.tau_n/4.);
		// // 		//ENDE TEST gehört zu Zeitkonvergenz
	
	} // end DWR-loop
	
	////////////////////////////////////////////////////////////////////////////
	//////////////// end DWR-loop "cycle" //////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////
	
	// Shell output of a table
	primal.convergence_table.set_precision("L2atT",5);
	primal.convergence_table.set_precision("J(e)",5);
	primal.convergence_table.set_precision("Eta",5);
	primal.convergence_table.set_precision("I_eff",5);
	primal.convergence_table.set_scientific("L2atT",true);
	primal.convergence_table.set_scientific("J(e)",true);
	primal.convergence_table.set_scientific("Eta",true);
	std::cout << std::endl;
	primal.convergence_table.write_text(std::cout);
	
	primal.convergence_table.evaluate_convergence_rates("J(e)",dealii::ConvergenceTable::reduction_rate);
	primal.convergence_table.evaluate_convergence_rates("J(e)",dealii::ConvergenceTable::reduction_rate_log2);

	primal.convergence_table.evaluate_convergence_rates("Eta",dealii::ConvergenceTable::reduction_rate);
	primal.convergence_table.evaluate_convergence_rates("Eta",dealii::ConvergenceTable::reduction_rate_log2);	
	////////////////////////////////////////////////////////////////////////////
	//LATEX 
	primal.convergence_table.set_tex_caption("cells","\\# cells");
	primal.convergence_table.set_tex_caption("dofs","\\# dofs");
	primal.convergence_table.set_tex_caption("L2atT","$L^2$ at $T$");
	primal.convergence_table.set_tex_caption("J(e)","J(e)");
	primal.convergence_table.set_tex_caption("Eta","\\eta");
	primal.convergence_table.set_tex_caption("I_eff","I_{eff}");
	primal.convergence_table.set_tex_format("cycle","r");
	primal.convergence_table.set_tex_format("cells","r");
	primal.convergence_table.set_tex_format("dofs","r");
	////////////////////////////////////////////////////////////////////////////
	//LATEX END
	std::cout << std::endl;
	primal.convergence_table.write_text(std::cout);
	////////////////////////////////////////////////////////////////////////////
	// LATEX
	std::string conv_filename = "convergence";
	conv_filename += "-globalref";
	conv_filename += ".tex";
	std::ofstream table_file(conv_filename.c_str());
	primal.convergence_table.write_tex(table_file);
	////////////////////////////////////////////////////////////////////////////
	// LATEX END

} // end function run()

} // namespace

#include "Heat_cG_DWR.inst.in"
