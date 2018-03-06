/**
 * @file Grid_DWR_0.tpl.cc
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


// PROJECT includes
#include <Heat/Grid/Grid_DWR_0.tpl.hh>

// DTM++ includes


// DEAL.II includes

// C++ includes

namespace Heat {

/// Generate grid. Throws Exception in base class.
template<int dim, int spacedim>
void
Grid_DWR_0<dim,spacedim>::
generate() {
	const double a(0.);
	const double b(1.);
	
	auto slab(this->slabs.begin());
	auto ends(this->slabs.end());
	
	for (; slab != ends; ++slab) {
		dealii::GridGenerator::hyper_cube(
			*(slab->tria),
			a,b,false
		);
	}
}


/// Set boundary indicators
template<int dim, int spacedim>
void
Grid_DWR_0<dim,spacedim>::
set_boundary_indicators() {
	// set boundary indicators
	auto slab(this->slabs.begin());
	auto ends(this->slabs.end());
	
	for (; slab != ends; ++slab) {
		auto cell(slab->tria->begin_active());
		auto endc(slab->tria->end());
		
		for (; cell != endc; ++cell) {
		if (cell->at_boundary()) {
		for (unsigned int face(0);
			face < dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
			if (cell->face(face)->at_boundary()) {
				cell->face(face)->set_boundary_id(
					static_cast<dealii::types::boundary_id> (
						Heat::types::boundary_id::Dirichlet)
				);
			}
		}}}
	}
}

} // namespaces

#include "Grid_DWR_0.inst.in"
