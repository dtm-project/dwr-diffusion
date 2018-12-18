/**
 * @file NeumannBoundary_Selector.tpl.cc
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhäuser (MPB)
 *
 * @date 2018-11-15, NeumannBoundary, UK, MPB
 * @date 2018-07-30, DirichletBoundary, contributed by MPB from Force_Selector (UK)
 * @date 2018-07-26, dwr, UK
 * @date 2018-05-28, piot/ewave, UK
 * @date 2016-05-30, UK
 * @date 2016-02-11, UK
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

#include <DTM++/base/LogStream.hh>

#include <diffusion/NeumannBoundary/NeumannBoundary_Selector.tpl.hh>
#include <diffusion/NeumannBoundary/NeumannBoundaries.hh>

// C++ includes
#include <vector>

namespace diffusion {
namespace neumann_boundary {

template<int dim>
void
Selector<dim>::
create_function(
		const std::string &_type,
		const std::string &_options,
		std::shared_ptr< dealii::Function<dim> > &function
	) const {
	
	////////////////////////////////////////////////////////////////////////////
	// parse the input string, arguments are splitted with spaces
	//
	std::string argument;
	std::vector< std::string > options;
	for (auto &character : _options) {
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
	
	DTM::pout << "* found configuration: neumann_boundary function = " << _type << std::endl;
	DTM::pout << "* found configuration: neumann_boundary options = " << std::endl;
	for (auto &option : options) {
		DTM::pout << "\t" << option << std::endl;
	}
	DTM::pout << std::endl;
	
	DTM::pout << "* generating function" << std::endl;
	
	////////////////////////////////////////////////////////////////////////////
	// 
	if (_type.compare("ZeroFunction") == 0) {
		AssertThrow(
			options.size() == 0,
			dealii::ExcMessage(
				"neumann_boundary options invalid, "
				"please check your input file data."
			)
		);
		
		function =
			std::make_shared< dealii::Functions::ZeroFunction<dim> > (1);
		
		DTM::pout
			<< "neumann_boundary selector: created zero function" << std::endl
			<< std::endl;
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// 
	if (_type.compare("ConstantFunction") == 0) {
		AssertThrow(
			options.size() == 1,
			dealii::ExcMessage(
				"neumann_boundary options invalid, "
				"please check your input file data."
			)
		);
		
		function = std::make_shared<
			dealii::Functions::ConstantFunction<dim> > (
			std::stod(options.at(0)),
			1
		);
		
		DTM::pout
			<< "neumann_boundary selector: created ConstantFunction "
			<< "as neumann_boundary function, with " << std::endl
			<< "\tf(1) = " << std::stod(options.at(0)) << " . " << std::endl
			<< std::endl;
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// 
	if (_type.compare("NeumannBoundary_KoecherBruchhaeuser2a") == 0) {
		AssertThrow(
			options.size() == 3,
			dealii::ExcMessage(
				"neumann_boundary options invalid, "
				"please check your input file data."
			)
		);
		
		function = std::make_shared< diffusion::neumann_boundary::KoecherBruchhaeuser2a<dim> >(
			std::stod(options.at(0)), // s
			std::stod(options.at(1)), // a
			std::stod(options.at(2))  // epsilon
		);
		
		DTM::pout
			<< "neumann_boundary selector: created KoecherBruchhaeuser2a "
			<< "as neumann_boundary function, with " << std::endl
			<< "\ts = " << std::stod(options.at(0)) << std::endl
			<< "\ta = " << std::stod(options.at(1)) << std::endl
			<< "\tepsilon = " << std::stod(options.at(2)) << std::endl
			<< std::endl;
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// 
	AssertThrow(
		false,
		dealii::ExcMessage("neumann_boundary function unknown, please check your input file data.")
	);
}

}} //namespaces

#include "NeumannBoundary_Selector.inst.in"
