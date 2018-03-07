/**
 * @file Heat_DWR__cGp_dG0__cGq_cG1.tpl.hh
 * 
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * 
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

#ifndef __Heat_DWR__cGp_dG0__cGq_cG1_tpl_hh
#define __Heat_DWR__cGp_dG0__cGq_cG1_tpl_hh

// PROJECT includes
#include <heat/parameters/ParameterSet.hh>

#include <heat/grid/Grid_DWR.tpl.hh>


// #include <Heat/ErrorEstimator/ErrorEstimators.hh>
// #include <Heat/types/error_functional.hh>



// DTM++ includes
#include <DTM++/base/LogStream.hh>
// #include <DTM++/base/Problem.hh>
#include <DTM++/io/DataOutput.tpl.hh>
#include <DTM++/types/storage_data_vectors.tpl.hh>

// DEAL.II includes
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

// C++ includes
#include <memory>
#include <algorithm>
#include <list>
#include <iterator>

namespace heat {

template<int dim>
class Heat_DWR__cGp_dG0__cGq_cG1 {
public:
	Heat_DWR__cGp_dG0__cGq_cG1() = default;
	virtual ~Heat_DWR__cGp_dG0__cGq_cG1() = default;
	
	virtual void set_input_parameters(
		std::shared_ptr< dealii::ParameterHandler > parameter_handler
	);
	
	virtual void set_grid(
		std::shared_ptr< heat::Grid_DWR<dim,1> > grid
	);
	
	virtual void run();

protected:
	std::shared_ptr< heat::dwr::ParameterSet > parameter_set;
	
	std::shared_ptr< heat::Grid_DWR<dim,1> > grid;
	
	/// function: keep shared_ptr to Function<dim> for several quantities
	struct {
// 		std::shared_ptr< dealii::Function<dim> > epsilon;
// 		std::shared_ptr< dealii::Function<dim> > f;
// 		std::shared_ptr< dealii::Function<dim> > u_D; // Dirichlet boundary val. fun.
		
		/// initial value function
		std::shared_ptr< dealii::Function<dim> > u_0;
	} function;
	
	/// primal: data structures for forward time marching
	struct {
		// storage container
		struct {
			/// primal solutions list (time dG(r)-Q_{G(r+1)} method)
			std::shared_ptr< DTM::types::storage_data_vectors<1> > um; // dof on t_m
			std::shared_ptr< DTM::types::storage_data_vectors<1> > u;  // time dof
			std::shared_ptr< DTM::types::storage_data_vectors<1> > un; // dof on t_n
		} storage;
		
		struct {
			/// iterator for slab struct elements for forward time marching
			typename DTM::types::spacetime::dwr::slabs<dim>::iterator slab_previous;
			typename DTM::types::spacetime::dwr::slabs<dim>::iterator slab;
			
			// iterator for storage vectors on I_n (time dG method)
			typename DTM::types::storage_data_vectors<1>::iterator um;
			typename DTM::types::storage_data_vectors<1>::iterator u;
			typename DTM::types::storage_data_vectors<1>::iterator un;
		} iterator;
		
		// Data Output
		DTM::DataOutput<dim> data_output;
	} primal;
	
	/// dual: data structures for backward time marching and error estimation
	struct {
		// storage container
		struct {
			/// primal solutions interpolated into dual FE room list (time cG(s)-Q_{GL(s+1)} method)
			std::shared_ptr< DTM::types::storage_data_vectors<2> > u;
			
			/// dual solutions list
			std::shared_ptr< DTM::types::storage_data_vectors<2> > z;
			
			/// error_indicators list
			std::shared_ptr< DTM::types::storage_data_vectors<1> > eta;
		} storage;
		
		// Data Output
		DTM::DataOutput<dim> data_output;
	} dual;
	
	////////////////////////////////////////////////////////////////////////////
	// protected member functions:
	//
	
	virtual void init_functions();
	
	virtual void init_grid();
	
	virtual void reinit_storage();
	
	// primal problem:
	
	virtual void solve_primal_problem();
	
	virtual void primal_reinit_data_output(
		const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab
	);
	
	virtual void primal_do_data_output(
// 		const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab
	);
	
	
	
// 	// primal problem

// 	
// 	virtual void primal_reinit();
// 	
// 	virtual void primal_compute_initial_condition();
// 	
// 	virtual void primal_assemble_system();
// 	
// 	virtual void primal_set_time(volatile double t_1);
// 	
// 	virtual void primal_assemble_f();
// 	
// 	virtual void primal_interpolate_to_next_grid();
// 	
// 	virtual void primal_assemble_rhs();
// 	
// 	virtual void primal_solve();
// 	
	
	
	
// 	virtual void interpolate_primal_to_dual();
	
// 	// dual problem
// 	virtual void solve_dual_problem();
// 	
// 	virtual void dual_reinit();
// 	
// 	virtual void dual_compute_initial_condition();
// // 	virtual void dual_compute_initial_condition_L2global();
// 	
// 	virtual void dual_assemble_system();
// 	
// 	virtual void dual_assemble_rhs_at_t_Nminus1();
// 	virtual void dual_assemble_rhs();
// 	
// 	virtual void dual_assemble_Je_L2global();
// 	
// 	virtual void dual_interpolate_to_next_grid();
// 	
// 	virtual void dual_assemble_rhs_L2global();
	
// 	virtual void dual_solve();
// 	// end dual problem
	
// 	virtual void refine_grids_dwr();
	
// 	// Compute I_effs
// 	virtual void compute_Ieff();
// 	virtual void compute_Ieff_L2global();
// 
// 	// Compute global errors
// 	virtual void compute_global_STL2_error(); // maybe not here / STL: space-time L2
	
// 	// Solution output
// 	virtual void primal_and_dual_solution_output();

// 	virtual void primal_do_data_output(const double n);
// 	virtual void dual_init_data_output();
// 	virtual void dual_do_data_output(const double n);
// 	virtual void do_data_output(const double cycle);
	
// 	/// primal: data structures for forward time marching
// 	struct {
// 		

		
// 		dealii::SparseMatrix<double> M; ///< primal problem mass matrix
// 		dealii::SparseMatrix<double> A; ///< primal problem stiffness matrix
// 		dealii::SparseMatrix<double> system_matrix; ///< primal problem system matrix (M+tau_n*A)
		
// 		struct {
// 			std::shared_ptr< dealii::Vector<double> > u; 	///< primal problem solution
// 			std::shared_ptr< dealii::Vector<double> > u_old; ///< primal solution from one time step ago
// 			std::shared_ptr< dealii::Vector<double> > u_old_interpolated; ///< u_old interpolated to next grid
// 		} slab;
		
// 		dealii::Vector<double> f; ///< primal problem f
// 		dealii::Vector<double> f_old; ///< NEW
		
// 		/// primal problem system_rhs vector ((tau_n*f_0) + (M*u_old_interpolated))
// 		dealii::Vector<double> system_rhs;
		
// 		// Data Output
// 		DTM::DataOutput<dim> data_output;
// 		unsigned int data_output_patches;
// 	} primal;
	
// 	struct {
// 		// storage container
// 		struct {
// 			std::shared_ptr< storage_data_vectors > u; ///< primal solutions interpolated into dual FE room list
// 			std::shared_ptr< storage_data_vectors > z; ///< dual solutions list
// 			
// 			std::shared_ptr< storage_data_vectors > eta; ///< error_indicators list
// 		} storage;
		
// 		struct {
// 			/// reverse iterator for slab struct elements for backward time marching
// 			typename DTM::types::spacetime::DWR::slabs<dim>::reverse_iterator slab;
// 			typename DTM::types::spacetime::DWR::slabs<dim>::reverse_iterator slab_previous;
// 		} iterator;
		
// 		// TODO:
// 		std::shared_ptr< dealii::Vector<double> > u; ///< primal problem solution on dual space
// 		dealii::SparseMatrix<double> M;				 ///< dual problem mass matrix
// 		dealii::SparseMatrix<double> A;              ///< dual problem stiffness matrix
// 		dealii::SparseMatrix<double> system_matrix;  ///< dual problem system matrix (M+(tau_n/2)*A)
// 		std::shared_ptr< dealii::Vector<double> > z; ///< dual problem solution
// 		std::shared_ptr< dealii::Vector<double> > z_old;	///< dual solution from time step ago
// 		std::shared_ptr< dealii::Vector<double> > z_old_interpolated;	///< z_old interpolated to next grid
		
// 		Heat::types::error_functional Je_type;		 ///< target functional type for rhs of dual problem
// 		dealii::Vector<double> Je;                   ///< dual problem rhs
// 		dealii::Vector<double> Je_old;                   ///< only for convergence tests
// 		dealii::Vector<double> Je_old_interpolated; ///< Je_old interpolated to next grid
// 		dealii::Vector<double> system_rhs; ///< dual problem system_rhs vector 
// 										   ///  ((M-(tau_n/2)*A)*z_old_interpolated+(tau_n/2)*(Je_old+Je_new))
		
// 		// Data Output
// 		DTM::DataOutput<dim> data_output;
// 		unsigned int data_output_patches;
// 	} dual;
	

	

	
// 	struct {
// 		std::shared_ptr< Heat::DWR::ErrorEstimator<dim> > DWR;
// 	} error_estimator;
	
// 	struct {
// 	} data;
};

} // namespace

#endif
