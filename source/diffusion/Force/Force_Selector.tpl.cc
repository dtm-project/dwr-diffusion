/**
 * @file Force_Selector.tpl.cc
 * @author Uwe Koecher (UK)
 *
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

#include <diffusion/Force/Force_Selector.tpl.hh>
#include <diffusion/Force/Forces.hh>

// MPI includes

// DEAL.II includes

// C++ includes
#include <vector>

namespace diffusion {
namespace force {

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
	
	DTM::pout << "* found configuration: force function = " << _type << std::endl;
	DTM::pout << "* found configuration: force options = " << std::endl;
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
				"force options invalid, "
				"please check your input file data."
			)
		);
		
		function =
			std::make_shared< dealii::Functions::ZeroFunction<dim> > (1);
		
		DTM::pout
			<< "force selector: created zero function" << std::endl
			<< std::endl;
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// 
	if (_type.compare("ConstantFunction") == 0) {
		AssertThrow(
			options.size() == 1,
			dealii::ExcMessage(
				"force options invalid, "
				"please check your input file data."
			)
		);
		
		function = std::make_shared<
			dealii::Functions::ConstantFunction<dim> > (
			std::stod(options.at(0)),
			1
		);
		
		DTM::pout
			<< "force selector: created ConstantFunction "
			<< "as force function, with " << std::endl
			<< "\tf(1) = " << std::stod(options.at(0)) << " . " << std::endl
			<< std::endl;
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// 
	if (_type.compare("Force_Hartmann142") == 0) {
		AssertThrow(
			options.size() == 2,
			dealii::ExcMessage(
				"force options invalid, "
				"please check your input file data."
			)
		);
		
		function = std::make_shared< diffusion::force::Hartmann142<dim> >(
			std::stod(options.at(0)), // a
			std::stod(options.at(1))  // epsilon
		);
		
		DTM::pout
			<< "force selector: created Hartmann142 "
			<< "as force function, with " << std::endl
			<< "\ta = " << std::stod(options.at(0)) << " , and " << std::endl
			<< "\tepsilon = " << std::stod(options.at(1)) << " . " << std::endl
			<< std::endl;
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// 
	if (_type.compare("Force_KoecherBruchhaeuser1") == 0) {
		AssertThrow(
			options.size() == 4,
			dealii::ExcMessage(
				"force options invalid, "
				"please check your input file data."
			)
		);
		
		function = std::make_shared< diffusion::force::KoecherBruchhaeuser1<dim> >(
			std::stod(options.at(0)), // s
			std::stod(options.at(1)), // a
			std::stod(options.at(2)), // rho
			std::stod(options.at(3))  // epsilon
		);
		
		DTM::pout
			<< "force selector: created KoecherBruchhaeuser1 "
			<< "as force function, with " << std::endl
			<< "\ts = " << std::stod(options.at(0)) << " , and " << std::endl
			<< "\ta = " << std::stod(options.at(1)) << " , and " << std::endl
			<< "\trho = " << std::stod(options.at(2)) << " , and " << std::endl
			<< "\tepsilon = " << std::stod(options.at(3)) << " . " << std::endl
			<< std::endl;
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// 
	if (_type.compare("Force_KoecherBruchhaeuser2") == 0) {
		AssertThrow(
			options.size() == 4,
			dealii::ExcMessage(
				"force options invalid, "
				"please check your input file data."
			)
		);
		
		function = std::make_shared< diffusion::force::KoecherBruchhaeuser2<dim> >(
			std::stod(options.at(0)), // s
			std::stod(options.at(1)), // a
			std::stod(options.at(2)), // rho
			std::stod(options.at(3))  // epsilon
		);
		
		DTM::pout
			<< "force selector: created KoecherBruchhaeuser2 "
			<< "as force function, with " << std::endl
			<< "\ts = " << std::stod(options.at(0)) << " , and " << std::endl
			<< "\ta = " << std::stod(options.at(1)) << " , and " << std::endl
			<< "\trho = " << std::stod(options.at(2)) << " , and " << std::endl
			<< "\tepsilon = " << std::stod(options.at(3)) << " . " << std::endl
			<< std::endl;
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// 
	AssertThrow(
		false,
		dealii::ExcMessage("force function unknown, please check your input file data.")
	);
}

}} //namespaces

#include "Force_Selector.inst.in"
