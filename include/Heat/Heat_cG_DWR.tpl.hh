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

#ifndef __Heat_cG_DWR_tpl_hh
#define __Heat_cG_DWR_tpl_hh

// PROJECT includes
#include <Heat/Grid/Grid_DWR.tpl.hh>
#include <Heat/ErrorEstimator/ErrorEstimators.hh>
#include <Heat/Storage/Data_vectors.tpl.hh>
#include <Heat/types/error_functional.hh>

// DTM++ includes
#include <DTM++/io/DataOutput.tpl.hh>

// DEAL.II includes
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/fe/fe.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

// C++ includes
#include <memory>
#include <string>
#include <algorithm>
#include <list>
#include <iterator>

namespace Heat {

template<int dim>
class Heat_cG_DWR {
public:
	Heat_cG_DWR() = default;
	virtual ~Heat_cG_DWR() = default;
	
	virtual void set_error_functional_type(
		const typename Heat::types::error_functional &error_functional_type
	);
	
	virtual void set_grid(std::shared_ptr< Heat::Grid_DWR<dim,1> > grid);
	
	virtual void set_epsilon(std::shared_ptr< dealii::Function<dim> > epsilon);
	
	virtual void set_BoundaryValues(std::shared_ptr< dealii::Function<dim> > BoundaryValues);
	virtual void set_BoundaryValues_dual(std::shared_ptr< dealii::Function<dim> > BoundaryValues_dual);
	
	virtual void set_f(std::shared_ptr< dealii::Function<dim> > f);
	virtual void set_f_dual(std::shared_ptr< dealii::Function<dim> > f_dual);
	
	virtual void set_evaluation_point(dealii::Point<dim> evaluation_point);
	
	virtual void set_data(
		const unsigned int p_primal,
		const unsigned int p_dual,
		const unsigned int global_refinement,
		double t0,
		double T,
		double tau_n
	);
	
	virtual void set_data_output_patches(
		unsigned int data_output_patches_primal,
		unsigned int data_output_patches_dual
	);
	
	virtual void run();

protected:
	virtual void init(const unsigned int global_refinement);
	virtual void init_storage();
	
	// primal problem
	virtual void solve_primal_problem();
	
	virtual void primal_reinit();
	virtual void primal_compute_initial_condition();
	virtual void primal_assemble_system();
	virtual void primal_set_time(double t_1);
	virtual void primal_assemble_f();
	virtual void primal_interpolate_to_next_grid();
	virtual void primal_assemble_rhs();
	virtual void primal_solve();
	
	virtual void interpolate_primal_to_dual();
		
	virtual void primal_process_solution(const unsigned int cycle);
	// end primal problem
	
	// dual problem
	virtual void solve_dual_problem();
	
	virtual void dual_reinit();
	
	virtual void dual_compute_initial_condition();
	virtual void dual_compute_initial_condition_L2final();
	virtual void dual_compute_initial_condition_L2global();
	virtual void dual_compute_initial_condition_mean_final();
	virtual void dual_compute_initial_condition_mean_global();
	virtual void dual_compute_initial_condition_point_final();
	
	virtual void dual_assemble_system();
	virtual void dual_set_time(double t_2);
	
	virtual void dual_assemble_rhs_at_t_Nminus1();
	virtual void dual_assemble_rhs();
	virtual void dual_assemble_Je_L2final();
	virtual void dual_assemble_Je_L2global();
	virtual void dual_assemble_Je_mean_final();
	virtual void dual_assemble_Je_mean_global();
	virtual void dual_assemble_Je_point_final();
	virtual void dual_interpolate_to_next_grid();
	virtual void dual_assemble_rhs_L2final();
	virtual void dual_assemble_rhs_L2global();
	virtual void dual_assemble_rhs_mean_final();
	virtual void dual_assemble_rhs_mean_global();
	virtual void dual_assemble_rhs_point_final();
	virtual void dual_solve();
	
	virtual void dual_process_solution(const unsigned int cycle);
	// end dual problem
	
	virtual void refine_grids_dwr();
	
	// Compute I_effs
	virtual void compute_Ieff();
	virtual void compute_Ieff_L2global();
	virtual void compute_Ieff_L2final();
	virtual void compute_Ieff_mean_final();
	virtual void compute_Ieff_mean_global();
	virtual void compute_Ieff_point_final();
	// Compute global errors
	virtual void compute_global_STL2_error();
	
	// Solution output
	virtual void primal_and_dual_solution_output();
	virtual void primal_init_data_output();
	virtual void primal_do_data_output(const double n);
	virtual void dual_init_data_output();
	virtual void dual_do_data_output(const double n);
	virtual void do_data_output(const double cycle);
	
	struct {
		dealii::SparseMatrix<double> M; 				///< primal problem mass matrix
		dealii::SparseMatrix<double> A; 				///< primal problem stiffness matrix
		dealii::SparseMatrix<double> system_matrix; 	///< primal problem system matrix (M+tau_n*A)
		std::shared_ptr< dealii::Vector<double> > u; 	///< primal problem solution
		std::shared_ptr< dealii::Vector<double> > u_old; ///< primal solution from one time step ago
		std::shared_ptr< dealii::Vector<double> > u_old_interpolated; ///< u_old interpolated to next grid
		dealii::Vector<double> f;       				///< primal problem f
		dealii::Vector<double> f_old;       				///< NEW
		dealii::Vector<double> system_rhs; 				///< primal problem system_rhs vector 
														///((tau_n*f_0) + (M*u_old_interpolated))
// TODO
		dealii::Vector<double> testvector;
		
		// Data Output
		DTM::DataOutput<dim> data_output;
		unsigned int data_output_patches;
		// Convergence Table
		dealii::ConvergenceTable convergence_table;
	} primal;
	
	struct {
		std::shared_ptr< dealii::Vector<double> > u; ///< primal problem solution on dual space
		dealii::SparseMatrix<double> M;				 ///< dual problem mass matrix
		dealii::SparseMatrix<double> A;              ///< dual problem stiffness matrix
		dealii::SparseMatrix<double> system_matrix;  ///< dual problem system matrix (M+(tau_n/2)*A)
		std::shared_ptr< dealii::Vector<double> > z; ///< dual problem solution
		std::shared_ptr< dealii::Vector<double> > z_old;	///< dual solution from time step ago
		std::shared_ptr< dealii::Vector<double> > z_old_interpolated;	///< z_old interpolated to next grid
		
		Heat::types::error_functional Je_type;		 ///< target functional type for rhs of dual problem
		dealii::Vector<double> Je;                   ///< dual problem rhs
		dealii::Vector<double> Je_old;                   ///< only for convergence tests
		//TEST
		dealii::Vector<double> Je_old_interpolated;		///< Je_old interpolated to next grid
		dealii::Vector<double> system_rhs;			 ///< dual problem system_rhs vector 
													 ///  ((M-(tau_n/2)*A)*z_old_interpolated+(tau_n/2)*(Je_old+Je_new))
		dealii::Point<dim> evaluation_point;
		
		// Data Output
		DTM::DataOutput<dim> data_output;
		unsigned int data_output_patches;
		std::vector< std::shared_ptr< dealii::Vector<double> > > solution_vectors;
		// Convergence Table
		dealii::ConvergenceTable convergence_table;
	} dual;
	
	// list of primal and dual solutions
	data_vectors_storage In_u; ///< primal solutions interpolated into dual FE room list
	data_vectors_storage In_z; ///< dual solutions list
	data_vectors_storage In_uprimal; ///< primal solutions list
	data_vectors_storage In_eta; ///< error_indicators list
	
	std::shared_ptr< Heat::Grid_DWR<dim,1> > grid;
	
	// iterators for list of grids
	typename In_grid_data_type<dim,1>::iterator it_In_grid;
	typename In_grid_data_type<dim,1>::iterator it_In_grid_previous;/// only for primal/dual_interp_to_next_grid fct.
	typename In_grid_data_type<dim,1>::reverse_iterator rit_In_grid;/// only for rbegin(),rend()
	typename In_grid_data_type<dim,1>::reverse_iterator rit_In_grid_previous; /// only for primal/dual_interp_to_next_grid fct.
	
	
	//TEST TODO
	typename l_data_vectors_storage::reverse_iterator rit_In_uback;
	
	std::shared_ptr< dealii::Function<dim> > epsilon;
	std::shared_ptr< dealii::Function<dim> > BoundaryValues;
	std::shared_ptr< dealii::Function<dim> > BoundaryValues_dual;
	
	double L2Error;
	double L2Error_global;
	
	struct {
		std::shared_ptr< dealii::Function<dim> > f; ///< Force function.
		std::shared_ptr< dealii::Function<dim> > f_dual;
	} function;
	
	struct {
		std::shared_ptr< Heat::DWR::ErrorEstimator<dim> > DWR;
	} error_estimator;
	
	
	struct {
		unsigned int p_primal; ///< polynomial degree p of primal problem
		unsigned int p_dual; ///< polynomial degree p of dual problem
		unsigned int global_refinement; ///< global_refinement for initial mesh
		
		double t0; ///< initial timepoint
		double T; ///< final timepoint
		double tau_n; ///< timestep size
		
		double primal_time; ///< primal time variable (t_primal)
		double dual_time; ///< dual time variable (t_dual)
		unsigned int primal_timestep_number; ///< primal run variable (current primal timestep)
		unsigned int dual_timestep_number; ///< dual run variable (current dual timestep)
	} data;
};

} // namespace

#endif
