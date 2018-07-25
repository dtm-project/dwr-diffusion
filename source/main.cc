/**
 * @file main.cc
 * 
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * 
 * @date 2018-07-25, tested and running instationary version, UK, MPB
 * @date 2018-03-06, new implementation, UK
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
#define MPIX_THREADS 1
// #define MPIX_THREADS 28
// #define MPIX_THREADS dealii::numbers::invalid_unsigned_int
////////////////////////////////////////////////////////////////////////////////

// PROJECT includes
#include <DTM++/base/LogStream.hh>

#include <heat/parameters/ParameterHandler.hh>
#include <heat/Heat_DWR__cGp_dG0__cGq_cG1.tpl.hh>
#include <heat/grid/Grids.hh>

// DEAL.II includes
#include <deal.II/base/exceptions.h>
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
		// Init application
		//
		
		// Attach deallog to process output
		dealii::deallog.attach(DTM::pout);
		dealii::deallog.depth_console(0);
		DTM::pout
			<< "****************************************"
			<< "****************************************"
			<< std::endl;
		
		DTM::pout
			<< "Hej, here is process " << MyPID+1 << " from " << NumProc << std::endl;
		
		// Check if an input parameter file is given.
		AssertThrow(
			!(argc < 2),
			dealii::ExcMessage (
				std::string ("===>\tUSAGE: ./dwr-heat <Input_Parameter_File.prm>"))
		);
		
		// Check if the given input parameter file can be opened.
		const std::string input_parameter_filename(argv[1]);
		{
			std::ifstream input_parameter_file(input_parameter_filename.c_str());
			// Test if an Input Parameter File is potentially given.
			AssertThrow(
				input_parameter_file,
				dealii::ExcMessage (
					std::string ("===>\tERROR: Input parameter file <")
					+ input_parameter_filename + "> not found."
				)
			);
		}
		
		// Prepare input parameter handling:
		auto parameter_handler =
			std::make_shared< heat::dwr::ParameterHandler > ();
		parameter_handler->parse_input(argv[1]);
		
		// Get minimal set of input parameters to drive the correct simulator
		unsigned int dimension;
		{
			// Problem dimension
			dimension = static_cast<unsigned int> (
				parameter_handler->get_integer("dim")
			);
		}
		
		//
		////////////////////////////////////////////////////////////////////////
		
		
		////////////////////////////////////////////////////////////////////////
		// Restrict usage to a single process (NumProc == 1) only.
		//
		
		AssertThrow(NumProc == 1, dealii::ExcMessage("MPI mode not supported."));
		
		//
		////////////////////////////////////////////////////////////////////////
		
		
		////////////////////////////////////////////////////////////////////////
		// Begin application
		//
		DTM::pout
			<< std::endl
			<< "*********************************************************"
			<< std::endl
			<< "dwr-heat solver: primal cG(p)-dG(0) with dual cG(q)-cG(1)"
			<< std::endl;
		
		// create simulator
		//std::shared_ptr< DTM::Problem > problem;
		DTM::pout
			<< "dwr-heat: dimension dim = " << dimension << std::endl
			<< std::endl;
		
		// select, setup and run simulator
		switch (dimension) {
		case 2: {
			auto problem = std::make_shared< heat::Heat_DWR__cGp_dG0__cGq_cG1<2> > ();
			
			// TODO: put the following behind the switch block
			problem->set_input_parameters(parameter_handler);
			
			auto grid = std::make_shared< heat::Grid_DWR_0<2,1> > ();
			problem->set_grid(grid);
			
			problem->run();
			
			break;
		}
		
		case 3: [[fallthrough]]
		default:
			dealii::ExcNotImplemented();
		}
		
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
