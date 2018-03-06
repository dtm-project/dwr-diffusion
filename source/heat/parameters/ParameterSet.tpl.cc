/**
 * @file   ParameterSet.cc
 * @author Uwe Koecher (UK)
 * @date 2018-03-06, UK
 * @date 2017-09-11, UK
 * @date 2017-02-07, UK
 *
 * @brief Keeps all parsed input parameters in a struct.
 */

/*  Copyright (C) 2012-2018 by Uwe Koecher                                    */
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
#include <heat/parameters/ParameterSet.hh>

// DEAL.II includes
#include <deal.II/base/exceptions.h>

// C++ includes
#include <limits>

namespace heat {
namespace dwr {

ParameterSet::
ParameterSet(
	std::shared_ptr< dealii::ParameterHandler > handler) {
	Assert(handler.use_count(), dealii::ExcNotInitialized());
	
	dim = static_cast<unsigned int> (handler->get_integer("dim"));
	
	handler->enter_subsection("Mesh Specification"); {
		use_mesh_input_file = handler->get_bool("use mesh input file");
		mesh_input_filename = handler->get("mesh input filename");
		
		GridGenerator = handler->get("GridGenerator");
		GridGenerator_Options = handler->get("GridGenerator Options");
		
		Grid_Class = handler->get("Grid Class");
		
		global_refinement = static_cast<unsigned int> (
			handler->get_integer("global refinement")
		);
		
		fe.element_type = handler->get("element type");
		fe.p = static_cast<unsigned int> (handler->get_integer("p"));
		fe.q = static_cast<unsigned int> (handler->get_integer("q"));
	}
	handler->leave_subsection();
	
	handler->enter_subsection("Time Integration"); {
		t0 = handler->get_double("initial time");
		T = handler->get_double("final time");
		tau_n = handler->get_double("time step size");
	}
	handler->leave_subsection();
	
	handler->enter_subsection("Parameter Specification"); {
		diffusion_epsilon_function = handler->get(
			"diffusion epsilon function"
		);
		
		diffusion_epsilon_options = handler->get(
			"diffusion epsilon options"
		);
		
		
		force_function = handler->get(
			"force function"
		);
		
		force_options = handler->get(
			"force options"
		);
		
		
		force_assembler_n_quadrature_points = static_cast<unsigned int> (
			handler->get_integer(
				"force assembler quadrature points"
			)
		);
		if (handler->get_bool("force assembler quadrature auto mode")) {
			force_assembler_n_quadrature_points += fe.p + 1;
		}
		
		
		dirichlet_boundary_u_D_function = handler->get(
			"dirichlet boundary u_D function"
		);
		
		dirichlet_boundary_u_D_options = handler->get(
			"dirichlet boundary u_D options"
		);
		
		
		dirichlet_assembler_n_quadrature_points = static_cast<unsigned int> (
			handler->get_integer(
				"dirichlet assembler quadrature points"
			)
		);
		if (handler->get_bool("dirichlet assembler quadrature auto mode")) {
			dirichlet_assembler_n_quadrature_points += fe.p + 1;
		}
		
		
		normal_traction_boundary_t_N_function = handler->get(
			"normal traction boundary t_N function"
		);
		
		normal_traction_boundary_t_N_options = handler->get(
			"normal traction boundary t_N options"
		);
		
		
		initial_displacement_u0_function = handler->get(
			"initial displacement u0 function"
		);
		
		initial_displacement_u0_options = handler->get(
			"initial displacement u0 options"
		);
	}
	handler->leave_subsection();
	
	
	handler->enter_subsection("Output Quantities"); {
		data_output.trigger = handler->get_double("data output trigger time");
		
		if (handler->get_bool("data output patches auto mode")) {
			data_output.patches = fe.p;
		}
		else {
			data_output.patches = static_cast<unsigned int> (
				handler->get_integer("data output patches")
			);
		}
	}
	handler->leave_subsection();
}

}} // namespace
