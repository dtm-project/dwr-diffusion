/**
 * @file Grid_DWR.tpl.hh
 * 
 * @author Uwe Koecher (UK)
 * 
 * @date 2018-07-20, refine_slab_in_time, UK
 * @date 2018-03-07, included in dwr-heat, UK
 * @date 2018-03-05, work on the data structures, UK
 * @date 2017-08-01, Heat/DWR, UK
 * @date 2016-02-10, condiffrea, UK
 * @date 2016-01-14, condiff, UK
 * @date 2016-01-12, UK
 * @date 2015-11-11, UK
 * @date 2015-05-15, DTM++/AcousticWave Module, UK
 * @date (2012-07-26), 2013-08-15, ElasticWave, UK
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

#ifndef __Grid_DWR_tpl_hh
#define __Grid_DWR_tpl_hh

// PROJECT includes
#include <DTM++/types/slabs.tpl.hh>
#include <heat/types/boundary_id.hh>

// DEAL.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>

// #include <deal.II/distributed/tria.h>
// #include <deal.II/distributed/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping.h>

// #include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/numerics/vector_tools.h>

// C++ includes
#include <memory>
#include <algorithm>
#include <list>
#include <iterator>

namespace heat {

template<int dim, int spacedim>
class Grid_DWR {
public:
	Grid_DWR() = default;
	virtual ~Grid_DWR();
	
	virtual void initialize_slabs(
		const unsigned int &p_primal,
		const unsigned int &q_dual,
		const double &t0,
		const double &T,
		const double &tau_n
	);
	
	virtual void refine_slab_in_time(
		typename DTM::types::spacetime::dwr::slabs<dim>::iterator slab
	);
	
	virtual void generate();
	virtual void refine_global(const unsigned int n = 1);
	virtual void set_boundary_indicators();
	
	virtual void distribute();
	
	DTM::types::spacetime::dwr::slabs<dim> slabs;
	
	dealii::GridIn<dim>            grid_in;
	dealii::GridOut                grid_out;
}; // end class Grid_DWR

} // namespace

#endif
