/**
 * @file LogStream.cc
 * @author Uwe Koecher (UK)
 * @date 2015-10-26, UK
 */

/*  Copyright (C) 2012-2015 by Uwe Koecher                                    */
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
/*  along with DTM++.  If not, see <http://www.gnu.org/licenses/>.            */

// PROJECT includes
#include <DTM++/base/LogStream.hh>

// MPI includes

// DEAL.II includes

// C++ includes

namespace DTM {

void
LogStream::open() {
	// Get MPI Variables
	const unsigned int MyPID(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
// 	const unsigned int NumProc(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
	
	// Create individual output file for each running process
	std::ostringstream filename;
	filename << "proc" << MyPID << ".log";
	std::ofstream::open(filename.str().c_str(), std::ios_base::out);
}

DTM::LogStream pout;

} // namespace
