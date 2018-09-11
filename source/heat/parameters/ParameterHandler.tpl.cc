/**
 * @file   ParameterHandler.cc
 * @author Uwe Koecher (UK)
 *
 * @date 2018-07-25, new parameters dwr, UK
 * @date 2018-03-06, included from ewave, UK
 * @date 2017-10-25, UK
 * @date 2017-02-06, UK
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
#include <heat/parameters/ParameterHandler.hh>

// DEAL.II includes
#include <deal.II/base/parameter_handler.h>

// C++ includes

namespace heat {
namespace dwr {

ParameterHandler::
ParameterHandler() {
	declare_entry(
		"dim",
		"2",
		dealii::Patterns::Integer(),
		"dim"
	);
	
	enter_subsection("Parameter Specification"); {
		declare_entry(
			"epsilon function",
			"invalid",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"epsilon options",
			"",
			dealii::Patterns::Anything()
		);
		
		
		declare_entry(
			"density function",
			"invalid",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"density options",
			"",
			dealii::Patterns::Anything()
		);
		
		
		declare_entry(
			"force function",
			"invalid",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"force options",
			"",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"force assembler quadrature auto mode",
			"true",
			dealii::Patterns::Bool()
		);
		
		declare_entry(
			"force assembler quadrature points",
			"0",
			dealii::Patterns::Integer()
		);
		
		
		declare_entry(
			"dirichlet boundary u_D function",
			"invalid",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"dirichlet boundary u_D options",
			"",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"dirichlet assembler quadrature auto mode",
			"false",
			dealii::Patterns::Bool()
		);
		
		declare_entry(
			"dirichlet assembler quadrature points",
			"0",
			dealii::Patterns::Integer()
		);
		
		
		declare_entry(
			"normal traction boundary t_N function",
			"invalid",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"normal traction boundary t_N options",
			"",
			dealii::Patterns::Anything()
		);
		
		
		declare_entry(
			"initial value u0 function",
			"invalid",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"initial value u0 options",
			"",
			dealii::Patterns::Anything()
		);
		
		
		declare_entry(
			"exact solution function",
			"invalid",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"exact solution options",
			"",
			dealii::Patterns::Anything()
		);
	}
	leave_subsection();
	
	
	enter_subsection("Mesh Specification"); {
		declare_entry(
			"use mesh input file",
			"false",
			dealii::Patterns::Bool(),
			"determines whether to use an input file or a deal.II GridGenerator"
		);
		
		declare_entry(
			"mesh input filename",
			"./input/.empty",
			dealii::Patterns::Anything(),
			"filename of the mesh which can be read in with dealii::GridIn"
		);
		
		declare_entry(
			"Grid Class",
			"invalid",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"Grid Class Options",
			"invalid",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"TriaGenerator",
			"invalid",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"TriaGenerator Options",
			"invalid",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"global refinement",
			"0",
			dealii::Patterns::Integer(),
			"Global refinements of the intial mesh"
		);
		
		
		declare_entry(
			"element type",
			"canocial",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"primal p",
			"1",
			dealii::Patterns::Integer(),
			"Polynomial degree p of primal problem"
		);
		
		declare_entry(
			"dual q",
			"1",
			dealii::Patterns::Integer(),
			"Polynomial degree q of dual problem"
		);
	}
	leave_subsection();
	
	
	enter_subsection("Time Integration"); {
		declare_entry(
			"initial time",
			"0.",
			dealii::Patterns::Double(),
			"initial time t0"
		);
		
		declare_entry(
			"final time",
			"0.",
			dealii::Patterns::Double(),
			"final time T"
		);
		
		declare_entry(
			"time step size",
			"1e-2",
			dealii::Patterns::Double(),
			"initial time step size"
		);
		
		declare_entry(
			"primal time discretisation",
			"dG0_Q_G1",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"dual time discretisation",
			"cG1_Q_G1",
			dealii::Patterns::Anything()
		);
	}
	leave_subsection();
	
	
	enter_subsection("DWR"); {
		declare_entry(
			"loops",
			"2",
			dealii::Patterns::Integer()
		);
		
		
		declare_entry(
			"refine and coarsen time strategy",
			"global",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"refine and coarsen time top fraction",
			"1.0",
			dealii::Patterns::Double()
		);
		
		
		declare_entry(
			"refine and coarsen space strategy",
			"global",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"refine and coarsen space top fraction",
			"1.0",
			dealii::Patterns::Double()
		);
		
		declare_entry(
			"refine and coarsen space bottom fraction",
			"0.0",
			dealii::Patterns::Double()
		);
		
		declare_entry(
			"refine and coarsen space max growth factor n_active_cells",
			"4",
			dealii::Patterns::Integer()
		);
		
		declare_entry(
			"refine and coarsen space Schwegler theta1",
			"1.0",
			dealii::Patterns::Double()
		);
		
		declare_entry(
			"refine and coarsen space Schwegler theta2",
			"0.0",
			dealii::Patterns::Double()
		);
	}
	leave_subsection();
	
	
	enter_subsection("Output Quantities"); {
		declare_entry(
			"primal data output dwr loop",
			"all",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"primal data output trigger type",
			"fixed",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"primal data output trigger time",
			"-1.",
			dealii::Patterns::Double()
		);
		
		declare_entry(
			"primal data output patches auto mode",
			"true",
			dealii::Patterns::Bool(),
			"primal data output patches auto mode => using p data output patches"
		);
		
		declare_entry(
			"primal data output patches",
			"1",
			dealii::Patterns::Integer()
		);
		
		declare_entry(
			"dual data output dwr loop",
			"all",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"dual data output trigger type",
			"fixed",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"dual data output trigger time",
			"-1.",
			dealii::Patterns::Double()
		);
		
		declare_entry(
			"dual data output patches auto mode",
			"true",
			dealii::Patterns::Bool(),
			"dual data output patches auto mode => using q data output patches"
		);
		
		declare_entry(
			"dual data output patches",
			"1",
			dealii::Patterns::Integer()
		);
	}
	leave_subsection();
}

}} // namespace
