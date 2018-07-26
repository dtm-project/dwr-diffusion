/**
 * @file Problem.cc
 * @author Uwe Koecher (UK)
 * @date 2016-02-12, UK
 */

/*  Copyright (C) 2012-2016 by Uwe Koecher                                    */
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
#include <DTM++/base/Problem.hh>

// MPI includes

// DEAL.II includes
#include <deal.II/base/exceptions.h>

// C++ includes

namespace DTM {

void
Problem::
set_input_parameters(
	std::shared_ptr< dealii::ParameterHandler > /*parameter_handler*/
) {
	AssertThrow(
		false,
		dealii::ExcNotImplemented()
	);
}


void
Problem::
run() {
	AssertThrow(
		false,
		dealii::ExcNotImplemented()
	);
}

} // namespace
