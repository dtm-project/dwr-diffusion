/**
 * @file TriaGenerator.tpl.hh
 * @author Uwe Koecher (UK)
 * 
 * @date 2018-07-26, included from biot/piot as TriaGenerator, UK
 * @date 2015-11-16, biot, UK
 * @date 2015-09-16, TriaGenerator from input file, UK
 * @date 2015-05-15, DTM++/AcousticWave Module, UK
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


#ifndef __TriaGenerator_tpl_hh
#define __TriaGenerator_tpl_hh

// Project includes
#include <deal.II/grid/tria.h>

// C++ includes
#include <string>
#include <vector>

namespace heat {

template<int dim>
class TriaGenerator {
public:
	TriaGenerator() = default;
	virtual ~TriaGenerator() = default;
	
	virtual void generate(
		const std::string &TriaGenerator_Type,
		const std::string &TriaGenerator_Options,
		std::shared_ptr< dealii::Triangulation<dim> > tria
	);
};

}

#endif
