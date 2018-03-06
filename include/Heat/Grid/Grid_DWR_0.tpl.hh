/**
 * @file Grid_DWR.tpl.hh
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * @date 2018-03-06, UK
 */

/*  Copyright (C) 2012-2018 by Uwe Koecher, Marius Paul Bruchhaeuser          */
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

#ifndef __Grid_DWR_0_tpl_hh
#define __Grid_DWR_0_tpl_hh

// PROJECT includes
#include <Heat/Grid/Grid_DWR.tpl.hh>

// DEAL.II includes

// C++ includes

namespace Heat {

template<int dim, int spacedim>
class Grid_DWR_0 : public Heat::Grid_DWR<dim,spacedim> {
public:
	Grid_DWR_0() = default;
	virtual ~Grid_DWR_0() = default;
	
	virtual void generate();
	virtual void set_boundary_indicators();
};

} // namespace

#endif
