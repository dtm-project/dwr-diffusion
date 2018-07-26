/**
 * @file TriaGenerator.tpl.cc
 * @author Uwe Koecher
 *
 * @date 2018-07-26, included from biot/piot as TriaGenerator, UK
 * @date 2015-11-16, biot, UK
 * @date 2015-09-16, GridGenerator from input file, UK
 * @date 2015-05-15, UK
 * @date 2013-08-15, UK
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

// DTM++ includes
#include <DTM++/base/LogStream.hh>

// Project includes
#include <heat/grid/TriaGenerator.tpl.hh>

// DEAL.II includes
#include <deal.II/grid/grid_generator.h>

// class declaration
namespace heat {

template<int dim>
void
TriaGenerator<dim>::
generate(
	const std::string &TriaGenerator_Type,
	const std::string &TriaGenerator_Options,
	std::shared_ptr< dealii::Triangulation<dim> > tria ) {
	// check if tria is initialized
	Assert(
		tria.use_count(),
		dealii::ExcNotInitialized()
	);
	
	////////////////////////////////////////////////////////////////////////////
	// parse the input string, arguments are splitted with spaces
	//
	std::string argument;
	std::vector< std::string > options;
	for (auto &character : TriaGenerator_Options) {
		if (!std::isspace(character) && (character!='\"') ) {
			argument += character;
		}
		else {
			if (argument.size()) {
				options.push_back(argument);
				argument.clear();
			}
		}
	}
	
	if (argument.size()) {
		options.push_back(argument);
		argument.clear();
	}
	
	////////////////////////////////////////////////////////////////////////////
	//
	if (TriaGenerator_Type.compare("hyper_cube") == 0) {
		AssertThrow(
			options.size() == 2,
			dealii::ExcMessage(
				"TriaGenerator Options invalid, "
				"please check your input file data."
			)
		);
		
		dealii::GridGenerator::hyper_cube(
			*tria,
			std::stod(options.at(0)),
			std::stod(options.at(1))
		);
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	//
	if (TriaGenerator_Type.compare("subdivided_hyper_rectangle") == 0) {
		AssertThrow(
			options.size() == 9,
			dealii::ExcMessage(
				"TriaGenerator Options invalid, "
				"please check your input file data."
			)
		);
		
		////////////////////////////////////////////////////////////////////////
		// Omega=(-x^1_val, x^1_val) x (-x^2_val, x^2_val) x (-x^3_val, x^3_val)
		//
		dealii::Point<dim> p1,p2;
		std::vector< unsigned int > repetitions(dim);
		{
			double x_i_val_a;
			double x_i_val_b;
			double x_i_h;
			
			for (unsigned int i(0); i < dim; ++i) {
				try {
					x_i_val_a = std::stod(options.at(i));
					x_i_val_b = std::stod(options.at(i+3));
					x_i_h = std::stod(options.at(i+6));
				}
				catch (std::invalid_argument &exc) {
					throw dealii::ExcMessage("TriaGenerator Options invalid, please check your input file data.");
				}
				catch (std::out_of_range &exc) {
					throw dealii::ExcMessage("TriaGenerator Options invalid, please check your input file data.");
				}
				
				// compute the repetitions for the grid:
				repetitions[i] = std::floor((x_i_val_b - x_i_val_a) / x_i_h);
				
				// set bottom left (p1) and upper right (p2) points
				p1[i] = x_i_val_a;
				p2[i] = x_i_val_b;
			}
		}
		
		dealii::GridGenerator::subdivided_hyper_rectangle(
			*tria,
			repetitions, p1, p2,
			false // colorize boundaries
		);
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// 
	AssertThrow(
		false,
		dealii::ExcMessage("TriaGenerator_Type unknown, please check your input file data.")
	);
	
}

} // namespaces

#include "TriaGenerator.inst.in"
