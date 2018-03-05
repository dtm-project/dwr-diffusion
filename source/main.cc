/**
 * @file main.cc
 * @author Marius Paul Bruchhaeuser (MPB), Uwe Koecher (UK)
 * @date 2017-08-01, Heat/DWR, UK
 * @date 2016-01-15, condiff/SUPG, UK
 * @date 2016-01-12, UK
 * @date 2015-11-11, UK
 *
 * @brief DTM++.Project/DWR/DWR-Heat: Solve the heat-eq with DWR.
 * 
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

// DEFINES

////////////////////////////////////////////////////////////////////////////////
// MPI Usage: Limit the numbers of threads to 1, since MPI+X has a poor
// performance (for at least Trilinos/Epetra).
// Undefine the variable USE_MPI_WITHOUT_THREADS if you want to use MPI+X,
// this would use as many threads per process as deemed useful by TBB.
#define USE_MPI_WITHOUT_THREADS

// We will further restrict to use a single process only,
// so we can enable threading parallelism safely by
// #undef USE_MPI_WITHOUT_THREADS

#ifdef USE_MPI_WITHOUT_THREADS
#define MPIX_THREADS 28 //28
#else
#define MPIX_THREADS dealii::numbers::invalid_unsigned_int
#endif
////////////////////////////////////////////////////////////////////////////////


// PROJECT includes
#include <DTM++/base/LogStream.hh>

#include <Heat/Heat_cG_DWR.tpl.hh>
#include <Heat/BoundaryValues/BoundaryValues.hh>
#include <Heat/Forces/Forces.hh>
#include <Heat/Grid/Grid_DWR.tpl.hh>
#include <Heat/types/error_functional.hh>

// DEAL.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/multithread_info.h>

// C++ includes
#include <iostream>
#include <fstream>
#include <memory>


int main(int argc, char *argv[]) {
	// Init MPI (or MPI+X)
	dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, MPIX_THREADS);
	
	// EVALUATE wall time now.
	auto wall_time_start = MPI_Wtime();
	
	// Prepare DTM++ process logging to file
	DTM::pout.open();
	
	// Get MPI Variables
	const unsigned int MyPID(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
	const unsigned int NumProc(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
	
	try {
		////////////////////////////////////////////////////////////////////////
		// Prepare output logging for deal.II
		//
		
		// Attach deallog to process output
		dealii::deallog.attach(DTM::pout);
		dealii::deallog.depth_console(0);
		DTM::pout
			<< "****************************************"
			<< "****************************************"
			<< std::endl
		;
		
		DTM::pout
			<< "Hej, here is process " << MyPID+1 << " from " << NumProc
			<< std::endl
		;
		
		//
		////////////////////////////////////////////////////////////////////////
		
		
		////////////////////////////////////////////////////////////////////////
		// Restrict usage to a single process (NumProc == 1) only.
		//
		
		AssertThrow(NumProc == 1, dealii::ExcMessage("MPI mode not supported."));
		
		//
		////////////////////////////////////////////////////////////////////////
		
		
		////////////////////////////////////////////////////////////////////////
		// Init application
		//
		const unsigned int DIM=2;
		
		const unsigned int p_primal = 1;
		const unsigned int p_dual   = 2;
		
		const unsigned int global_refine = 3;
		
		double t0 = 0.;
		double T  = 0.5;
		double tau_n = 0.02;
		const unsigned int time_steps = ((T-t0)/tau_n);
		
// 		dealii::Point<DIM> evaluation_point(0.1875, 0.125);
		dealii::Point<DIM> evaluation_point(0.5, 0.5);
		
		unsigned int data_output_patches_primal = p_primal;
		unsigned int data_output_patches_dual = p_dual;
		
		// choose error functional / goal functional type J()
		auto error_functional_type = Heat::types::error_functional::L2_final;
		
		////////////////////////////////////////////////////////////////////////
		// grid
		//
		auto grid = std::make_shared< Heat::Grid_DWR<DIM,1> > ();
		grid->set_data(p_primal, p_dual);

		////////////////////////////////////////////////////////////////////////
		// functions
		//
		auto epsilon = std::make_shared< dealii::ConstantFunction<DIM> > (1.e-00); ///< diffusion coefficient
		auto alpha = std::make_shared< dealii::ConstantFunction<DIM> > (50.e-00); ///< coefficient for Hartmann fct.
		
		// Use (alpha) instead of (epsilon) for BV_Hartmann only!
		auto BoundaryValues = std::make_shared< Heat::BoundaryValues_MH<DIM> > (epsilon); ///< boundary values/ exact solution
		
		// Use (epsilon,alpha,BV) instead of (epsilon,BV) for Hartmann only! 
		auto f = std::make_shared< Heat::Moving_Hump<DIM> > (epsilon,BoundaryValues); ///< rhs
		
		// Use (alpha) instead of (epsilon) for BV_Hartmann only!
		auto BoundaryValues_dual = std::make_shared< Heat::BoundaryValues_MH<DIM> > (epsilon); ///< boundary values/ exact solution
		
		// Use (epsilon,alpha,BV) instead of (epsilon,BV) for Hartmann only!
		auto f_dual = std::make_shared< Heat::Moving_Hump<DIM> > (epsilon,BoundaryValues); ///< rhs
		
		////////////////////////////////////////////////////////////////////////
		// Begin application
		//
		
		auto problem = std::make_shared< Heat::Heat_cG_DWR<DIM> > ();
		
		problem->set_error_functional_type(error_functional_type);
		problem->set_grid(grid);
		
		problem->set_epsilon(epsilon);
		problem->set_BoundaryValues(BoundaryValues);
		problem->set_BoundaryValues_dual(BoundaryValues_dual);
		problem->set_f(f);
		problem->set_f_dual(f_dual);
		problem->set_evaluation_point(evaluation_point);
		
		problem->set_data(p_primal, p_dual, global_refine, time_steps, t0, T, tau_n);
		
		problem->set_data_output_patches(
			data_output_patches_primal,
			data_output_patches_dual
		);
		
		problem->run();
		
		DTM::pout << std::endl << "Goodbye." << std::endl;
		
		//
		// End application
		////////////////////////////////////////////////////////////////////////////<
	}
	catch (std::exception &exc) {
		// EVALUATE program run time in terms of the consumed wall time.
		auto wall_time_end = MPI_Wtime();
		DTM::pout
			<< std::endl
			<< "Elapsed wall time: " << wall_time_end-wall_time_start
			<< std::endl
		;
		
		if (!dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) {
			std::cerr
				<< std::endl
				<< "****************************************"
				<< "****************************************"
				<< std::endl << std::endl
				<< "An EXCEPTION occured: Please READ the following output CAREFULLY!"
				<< std::endl
			;
			
			std::cerr << exc.what() << std::endl;
			
			std::cerr
				<< std::endl
				<< "APPLICATION TERMINATED unexpectedly due to an exception."
				<< std::endl << std::endl
				<< "****************************************"
				<< "****************************************"
				<< std::endl << std::endl
			;
		}
		
		// LOG error message to individual process output file.
		DTM::pout
			<< std::endl
			<< "****************************************"
			<< "****************************************"
			<< std::endl << std::endl
			<< "An EXCEPTION occured: Please READ the following output CAREFULLY!"
			<< std::endl
		;
		
		DTM::pout << exc.what() << std::endl;
		
		DTM::pout
			<< std::endl
			<< "APPLICATION TERMINATED unexpectedly due to an exception."
			<< std::endl << std::endl
			<< "****************************************"
			<< "****************************************"
			<< std::endl << std::endl
		;
		
		// Close output file stream
		DTM::pout.close();
		
		return 1;
	}
	catch (...) {
		// EVALUATE program run time in terms of the consumed wall time.
		auto wall_time_end = MPI_Wtime();
		DTM::pout
			<< std::endl
			<< "Elapsed wall time: " << wall_time_end-wall_time_start
			<< std::endl
		;
		
		if (!dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) {
			std::cerr
				<< std::endl
				<< "****************************************"
				<< "****************************************"
				<< std::endl << std::endl
				<< "An UNKNOWN EXCEPTION occured!"
				<< std::endl
			;
			
			std::cerr
				<< std::endl
				<< "----------------------------------------"
				<< "----------------------------------------"
				<< std::endl << std::endl
				<< "Further information:" << std::endl
				<< "\tThe main() function catched an exception"
				<< std::endl
				<< "\twhich is not inherited from std::exception."
				<< std::endl
				<< "\tYou have probably called 'throw' somewhere,"
				<< std::endl
				<< "\tif you do not have done this, please contact the authors!"
				<< std::endl << std::endl
				<< "----------------------------------------"
				<< "----------------------------------------"
				<< std::endl
			;
			
			std::cerr
				<< std::endl
				<< "APPLICATION TERMINATED unexpectedly due to an exception."
				<< std::endl << std::endl
				<< "****************************************"
				<< "****************************************"
				<< std::endl << std::endl
			;
		}
		
		// LOG error message to individual process output file.
		DTM::pout
			<< std::endl
			<< "****************************************"
			<< "****************************************"
			<< std::endl << std::endl
			<< "An UNKNOWN EXCEPTION occured!"
			<< std::endl
		;
		
		DTM::pout
			<< std::endl
			<< "----------------------------------------"
			<< "----------------------------------------"
			<< std::endl << std::endl
			<< "Further information:" << std::endl
			<< "\tThe main() function catched an exception"
			<< std::endl
			<< "\twhich is not inherited from std::exception."
			<< std::endl
			<< "\tYou have probably called 'throw' somewhere,"
			<< std::endl
			<< "\tif you do not have done this, please contact the authors!"
			<< std::endl << std::endl
			<< "----------------------------------------"
			<< "----------------------------------------"
			<< std::endl
		;
		
		DTM::pout
			<< std::endl
			<< "APPLICATION TERMINATED unexpectedly due to an exception."
			<< std::endl << std::endl
			<< "****************************************"
			<< "****************************************"
			<< std::endl << std::endl
		;
		
		// Close output file stream
		DTM::pout.close();
		
		return 1;
	}

	// EVALUATE program run time in terms of the consumed wall time.
	auto wall_time_end = MPI_Wtime();
	DTM::pout
		<< std::endl
		<< "Elapsed wall time: " << wall_time_end-wall_time_start
		<< std::endl
	;
	
	// Close output file stream
	DTM::pout.close();
	
	return 0;
}
