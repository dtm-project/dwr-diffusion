/**
 * @file Grid_DWR_Dirichlet_and_Neumann_1.tpl.hh
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * 
 * @date 2018-07-26, UK
 * @date 2018-03-06, UK
 */

/*  Copyright (C) 2012-2018 by Uwe Koecher and contributors                   */
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

#ifndef __Grid_DWR_Dirichlet_and_Neumann_1_tpl_hh
#define __Grid_DWR_Dirichlet_and_Neumann_1_tpl_hh

// PROJECT includes
#include <heat/grid/Grid_DWR.tpl.hh>

// DEAL.II includes

// C++ includes

namespace heat {
namespace grid {

template<int dim, int spacedim>
class Grid_DWR_Dirichlet_and_Neumann_1 : public heat::Grid_DWR<dim,spacedim> {
public:
	Grid_DWR_Dirichlet_and_Neumann_1(
		const std::string &Grid_Class_Options,
		const std::string &TriaGenerator,
		const std::string &TriaGenerator_Options) :
		heat::Grid_DWR<dim,spacedim> (TriaGenerator, TriaGenerator_Options),
		Grid_Class_Options(Grid_Class_Options) { };
	
	virtual ~Grid_DWR_Dirichlet_and_Neumann_1() = default;
	
	virtual void set_boundary_indicators();
	
private:
	const std::string Grid_Class_Options;
};

}} // namespace

#endif
