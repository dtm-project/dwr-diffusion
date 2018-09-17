/**
 * @file ExactSolution_Selector.tpl.cc
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhäuser (MPB)
 *
 * @date 2018-07-30, dwr, MPB
 * @date 2018-07-26, dwr, UK
 * @date 2018-05-28, piot/ewave, UK
 * @date 2016-05-30, UK
 * @date 2016-02-11, UK
 */

/*  Copyright (C) 2012-2018 by Uwe Koecher,  Marius Paul Bruchhäuser          */
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

#include <heat/ExactSolution/ExactSolutions.hh>
#include <heat/ExactSolution/ExactSolution_Selector.tpl.hh>

// MPI includes

// DEAL.II includes

// C++ includes
#include <vector>

namespace heat {
namespace exact_solution {

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
	
	DTM::pout << "* found configuration: exact_solution function = " << _type << std::endl;
	DTM::pout << "* found configuration: exact_solution options = " << std::endl;
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
				"exact_solution options invalid, "
				"please check your input file data."
			)
		);
		
		function =
			std::make_shared< dealii::Functions::ZeroFunction<dim> > (1);
		
		DTM::pout
			<< "exact_solution selector: created zero function" << std::endl
			<< std::endl;
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// 
	if (_type.compare("ConstantFunction") == 0) {
		AssertThrow(
			options.size() == 1,
			dealii::ExcMessage(
				"exact_solution options invalid, "
				"please check your input file data."
			)
		);
		
		function = std::make_shared<
			dealii::Functions::ConstantFunction<dim> > (
			std::stod(options.at(0)),
			1
		);
		
		DTM::pout
			<< "exact_solution selector: created ConstantFunction "
			<< "as exact_solution function, with " << std::endl
			<< "\tf(1) = " << std::stod(options.at(0)) << " . " << std::endl
			<< std::endl;
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// 
	if (_type.compare("ExactSolution_Hartmann142") == 0) {
		AssertThrow(
			options.size() == 1,
			dealii::ExcMessage(
				"exact_solution options invalid, "
				"please check your input file data."
			)
		);
		
		function = std::make_shared< heat::exact_solution::Hartmann142<dim> >(
			std::stod(options.at(0)) // a
		);
		
		DTM::pout
			<< "exact_solution selector: created Hartmann142 "
			<< "as exact_solution function, with " << std::endl
			<< "\ta = " << std::stod(options.at(0)) << std::endl
			<< std::endl;
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// 
	if (_type.compare("ExactSolution_KoecherBruchhaeuser1") == 0) {
		AssertThrow(
			options.size() == 1,
			dealii::ExcMessage(
				"exact_solution options invalid, "
				"please check your input file data."
			)
		);
		
		function = std::make_shared< heat::exact_solution::KoecherBruchhaeuser1<dim> >(
			std::stod(options.at(0)) // a
		);
		
		DTM::pout
			<< "exact_solution selector: created KoecherBruchhaeuser1 "
			<< "as exact_solution function, with " << std::endl
			<< "\ta = " << std::stod(options.at(0)) << std::endl
			<< std::endl;
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// 
	AssertThrow(
		false,
		dealii::ExcMessage("exact_solution function unknown, please check your input file data.")
	);
}

}} //namespaces

#include "ExactSolution_Selector.inst.in"
