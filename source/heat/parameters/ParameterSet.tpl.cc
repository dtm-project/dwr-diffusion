/**
 * @file   ParameterSet.cc
 * @author Uwe Koecher (UK)
 *
 * @date 2018-07-25, new parameters dwr, UK
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
		
		Grid_Class = handler->get("Grid Class");
		Grid_Class_Options = handler->get("Grid Class Options");
		
		TriaGenerator = handler->get("TriaGenerator");
		TriaGenerator_Options = handler->get("TriaGenerator Options");
		
		global_refinement = static_cast<unsigned int> (
			handler->get_integer("global refinement")
		);
		
		fe.element_type = handler->get("element type");
		fe.p = static_cast<unsigned int> (handler->get_integer("primal p"));
		fe.q = static_cast<unsigned int> (handler->get_integer("dual q"));
	}
	handler->leave_subsection();
	
	handler->enter_subsection("Time Integration"); {
		t0 = handler->get_double("initial time");
		T = handler->get_double("final time");
		tau_n = handler->get_double("time step size");
		
		primal_time_discretisation = handler->get("primal time discretisation");
		dual_time_discretisation = handler->get("dual time discretisation");
	}
	handler->leave_subsection();
	
	handler->enter_subsection("DWR"); {
		dwr.loops = static_cast<unsigned int> (handler->get_integer("loops"));
		
		
		dwr.refine_and_coarsen.time.strategy = handler->get(
			"refine and coarsen time strategy"
		);
		
		dwr.refine_and_coarsen.time.top_fraction = handler->get_double(
			"refine and coarsen time top fraction"
		);
		
		
		dwr.refine_and_coarsen.space.strategy = handler->get(
			"refine and coarsen space strategy"
		);
		
		dwr.refine_and_coarsen.space.top_fraction = handler->get_double(
			"refine and coarsen space top fraction"
		);
		
		dwr.refine_and_coarsen.space.bottom_fraction = handler->get_double(
			"refine and coarsen space bottom fraction"
		);
		
		dwr.refine_and_coarsen.space.max_growth_factor_n_active_cells =
			static_cast<unsigned int> (handler->get_integer(
			"refine and coarsen space max growth factor n_active_cells"
		));
		
		dwr.refine_and_coarsen.space.theta1 = handler->get_double(
			"refine and coarsen space Schwegler theta1"
		);
		
		dwr.refine_and_coarsen.space.theta2 = handler->get_double(
			"refine and coarsen space Schwegler theta2"
		);
	}
	handler->leave_subsection();
	
	handler->enter_subsection("Parameter Specification"); {
		epsilon_function = handler->get(
			"epsilon function"
		);
		
		epsilon_options = handler->get(
			"epsilon options"
		);
		
		
		density_function = handler->get(
			"density function"
		);
		
		density_options = handler->get(
			"density options"
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
		
		
		initial_value_u0_function = handler->get(
			"initial value u0 function"
		);
		
		initial_value_u0_options = handler->get(
			"initial value u0 options"
		);
		
		
		exact_solution_function = handler->get(
			"exact solution function"
		);
		
		exact_solution_options = handler->get(
			"exact solution options"
		);
	}
	handler->leave_subsection();
	
	
	handler->enter_subsection("Output Quantities"); {
		data_output.primal.dwr_loop = handler->get("primal data output dwr loop");
		
		data_output.primal.trigger_type = handler->get("primal data output trigger type");
		data_output.primal.trigger = handler->get_double("primal data output trigger time");
		
		if (handler->get_bool("primal data output patches auto mode")) {
			data_output.primal.patches = fe.p;
		}
		else {
			data_output.primal.patches = static_cast<unsigned int> (
				handler->get_integer("primal data output patches")
			);
		}
		
		data_output.dual.dwr_loop = handler->get("dual data output dwr loop");
		
		data_output.dual.trigger_type = handler->get("dual data output trigger type");
		data_output.dual.trigger = handler->get_double("dual data output trigger time");
		
		if (handler->get_bool("dual data output patches auto mode")) {
			data_output.dual.patches = fe.q;
		}
		else {
			data_output.dual.patches = static_cast<unsigned int> (
				handler->get_integer("dual data output patches")
			);
		}
	}
	handler->leave_subsection();
}

}} // namespace
