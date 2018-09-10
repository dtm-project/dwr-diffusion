/**
 * @file   ParameterSet.hh
 * @author Uwe Koecher (UK)
 *
 * @date 2018-07-25, new parameters dwr, UK
 * @date 2018-03-06, included from ewave, UK
 * @date 2017-10-25, UK
 * @date 2017-02-06, UK
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

#ifndef __ParameterSet_hh
#define __ParameterSet_hh

// PROJECT includes

// DEAL.II includes
#include <deal.II/base/parameter_handler.h>

// C++ includes
#include <string>
#include <memory>

namespace heat {
namespace dwr {

struct ParameterSet {
	ParameterSet( std::shared_ptr< dealii::ParameterHandler > handler );
	
	unsigned int dim;
	
	// parameter specification
	std::string epsilon_function;
	std::string epsilon_options;
	
	std::string density_function;
	std::string density_options;
	
	std::string force_function;
	std::string force_options;
	unsigned int force_assembler_n_quadrature_points;
	
	std::string dirichlet_boundary_u_D_function;
	std::string dirichlet_boundary_u_D_options;
	unsigned int dirichlet_assembler_n_quadrature_points;
	
	std::string normal_traction_boundary_t_N_function;
	std::string normal_traction_boundary_t_N_options;
	
	std::string initial_value_u0_function;
	std::string initial_value_u0_options;
	
	std::string exact_solution_function;
	std::string exact_solution_options;
	
	// mesh specification
	bool use_mesh_input_file;
	std::string mesh_input_filename;
	
	std::string Grid_Class;
	std::string Grid_Class_Options;
	std::string TriaGenerator;
	std::string TriaGenerator_Options;
	
	unsigned int global_refinement;
	
	struct {
		std::string element_type;
		unsigned int p;
		unsigned int q;
	} fe;
	
	// time integration
	double t0;
	double T;
	double tau_n;
	
	// dwr
	struct {
		unsigned int loops;
	} dwr;
	
	// data output
	struct {
		struct {
			std::string dwr_loop;
			std::string trigger_type;
			
			double trigger;
			unsigned int patches;
		} primal;
		
		struct {
			std::string dwr_loop;
			std::string trigger_type;
			
			double trigger;
			unsigned int patches;
		} dual;
	} data_output;
};

}} // namespace

#endif
