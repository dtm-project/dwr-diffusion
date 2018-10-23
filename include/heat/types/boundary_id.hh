/**
 * @file boundary_id.hh
 *
 * @author Uwe Koecher (UK)
 *
 * @date 2018-03-07, UK
 * @date 2017-11-08, UK
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


#ifndef __boundary_id_hh
#define __boundary_id_hh

namespace heat {
namespace types {

enum class boundary_id : unsigned int {
	forbidden = 0,
	
	/**
	 * Colour for marking \f$ \Gamma_D \f$ from the partition
	 * \f$ \partial \Omega = \Gamma_D \cup \Gamma_N \f$ with
	 * \f$ \Gamma_D \neq \emptyset \f$.
	 */
	Dirichlet = 2 << 0,
	
	/**
	 * Colour for marking \f$ \Gamma_N \f$ from the partition
	 * \f$ \partial \Omega = \Gamma_D \cup \Gamma_N \f$ with
	 * \f$ \Gamma_D \neq \emptyset \f$.
	 */
	Neumann   = 2 << 1
};

}}

#endif
