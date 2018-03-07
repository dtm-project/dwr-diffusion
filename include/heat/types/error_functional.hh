/**
 * @file error_functional.hh
 * @author Uwe Koecher (UK), Marius Paul Bruchhaeuser (MPB)
 * @date 2018-02-08, UK
 *
 * @brief enum class for different error funcitonals
 * 
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


#ifndef __error_functional_hh
#define __error_functional_hh

namespace Heat {
namespace types {

enum class error_functional : unsigned int {
	forbidden 	= 0,
	L2_final 	= 1,
	L2_global	= 2,
	mean_final	= 3,
	mean_global	= 4,
	point		= 5
};

}}

#endif
