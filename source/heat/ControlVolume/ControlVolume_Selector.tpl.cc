/**
 * @file ControlVolume_Selector.tpl.cc
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhäuser (MPB)
 *
 * @date 2018-11-19, included from NeumannBoundary, UK, MPB
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

#include <heat/ControlVolume/ControlVolume_Selector.tpl.hh>
#include <heat/ControlVolume/ControlVolumes.hh>

// C++ includes
#include <vector>

namespace heat {
namespace control_volume {

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
	
	DTM::pout << "* found configuration: control_volume function = " << _type << std::endl;
	DTM::pout << "* found configuration: control_volume options = " << std::endl;
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
				"control_volume options invalid, "
				"please check your input file data."
			)
		);
		
		function =
			std::make_shared< dealii::Functions::ZeroFunction<dim> > (1);
		
		DTM::pout
			<< "control_volume selector: created zero function" << std::endl
			<< std::endl;
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// 
	if (_type.compare("ConstantFunction") == 0) {
		AssertThrow(
			options.size() == 1,
			dealii::ExcMessage(
				"control_volume options invalid, "
				"please check your input file data."
			)
		);
		
		function = std::make_shared<
			dealii::Functions::ConstantFunction<dim> > (
			std::stod(options.at(0)),
			1
		);
		
		DTM::pout
			<< "control_volume selector: created ConstantFunction "
			<< "as control_volume function, with " << std::endl
			<< "\tf(1) = " << std::stod(options.at(0)) << " . " << std::endl
			<< std::endl;
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// 
	if (_type.compare("ControlVolume_hyper_rectangle") == 0) {
		AssertThrow(
			options.size() == 3+3+1+1,
			dealii::ExcMessage(
				"control_volume options invalid, "
				"please check your input file data."
			)
		);
		
		dealii::Point<dim> p1;
		dealii::Point<dim> p2;
		double t1;
		double t2;
		
		p1[0] = std::stod(options.at(0));
		p2[0] = std::stod(options.at(3));
		
		if (dim > 1) {
			p1[1] = std::stod(options.at(1));
			p2[1] = std::stod(options.at(4));
		}
		
		if (dim > 2) {
			p1[2] = std::stod(options.at(2));
			p2[2] = std::stod(options.at(5));
		}
		
		t1 = std::stod(options.at(6));
		t2 = std::stod(options.at(7));
		
		function = std::make_shared< heat::control_volume::hyper_rectangle<dim> >(
			p1,p2,t1,t2
		);
		
		DTM::pout
			<< "control_volume selector: created hyper_rectangle "
			<< "as control_volume function." << std::endl
			<< std::endl;
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// 
	if (_type.compare("ControlVolume_moving_hyper_rectangle") == 0) {
		AssertThrow(
			options.size() == 1+3+3+1+1,
			dealii::ExcMessage(
				"control_volume options invalid, "
				"please check your input file data."
			)
		);
		
		double r1;
		dealii::Point<dim> p1;
		dealii::Point<dim> p2;
		double t1;
		double t2;
		
		r1 = std::stod(options.at(0));
		
		p1[0] = std::stod(options.at(1));
		p2[0] = std::stod(options.at(4));
		
		if (dim > 1) {
			p1[1] = std::stod(options.at(2));
			p2[1] = std::stod(options.at(5));
		}
		
		if (dim > 2) {
			p1[2] = std::stod(options.at(3));
			p2[2] = std::stod(options.at(6));
		}
		
		t1 = std::stod(options.at(7));
		t2 = std::stod(options.at(8));
		
		function = std::make_shared< heat::control_volume::moving_hyper_rectangle<dim> >(
			r1, p1,p2, t1,t2
		);
		
		DTM::pout
			<< "control_volume selector: created moving_hyper_rectangle "
			<< "as control_volume function." << std::endl
			<< std::endl;
		
		return;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// 
	AssertThrow(
		false,
		dealii::ExcMessage("control_volume function unknown, please check your input file data.")
	);
}

}} //namespaces

#include "ControlVolume_Selector.inst.in"
