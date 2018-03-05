/**
 * @file boundary_id.hh
 * @author Uwe Koecher (UK), Marius Paul Bruchhaeuser (MPB)
 * @date 2017-11-08, UK, MPB
 *
 * @brief boundary_id
 * 
 */

/*  Copyright (C) 2012-2017 by Uwe Koecher, Marius Paul Bruchhaeuser          */ 
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


#ifndef __boundary_id_hh
#define __boundary_id_hh

namespace Heat {
namespace types {

enum class boundary_id : unsigned int {
	Dirichlet = 1,
	Neumann = 2
};

}}

#endif
