/**
 * @file Data_vecectors.tpl.hh
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * 
 * @date 2018-03-05, updated data structures and types, UK
 * @date 2017-07-27, Heat/DWR, UK, MPB
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

#ifndef __Data_vectors_tpl_hh
#define __Data_vectors_tpl_hh

// dealii includes
#include <deal.II/lac/vector.h>

// C++ includes
#include <list>

namespace storage {
namespace data {

struct s_vectors {
	std::shared_ptr< dealii::Vector<double> > x;
};

using vectors = struct s_vectors;

}}

/// storage container data type: list for shared_ptr to vectors
using storage_data_vectors = std::list< storage::data::vectors>;

#endif
