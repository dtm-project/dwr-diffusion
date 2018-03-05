/**
 * @file Grid_DWR.tpl.cc
 * @author Marius Paul Bruchhaeuser (MPB), Uwe Koecher (UK)
 * @date 2017-08-01, Heat/DWR, MPB, UK
 * @date 2016-02-10, condiffrea, UK
 * @date 2016-01-14, condiff, UK
 * @date 2016-01-12, UK
 * @date 2015-11-11, UK
 * @date 2015-05-15, UK
 *
 * @brief Grid_DWR instantiation file
 */

/*  Copyright (C) 2012-2016 by Marius Paul Bruchhaeuser, Uwe Koecher          */
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
#include <Heat/Grid/Grid_DWR.tpl.hh>

// DTM++ includes
#include <DTM++/base/LogStream.hh>

// DEAL.II includes
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/fe/mapping_q.h>

// C++ includes
#include <cmath>
#include <limits>

// class declaration
namespace Heat {

/// Destructor. Clears DoFHandler.
template<int dim, int spacedim>
Grid_DWR<dim,spacedim>::
~Grid_DWR() {
	auto slab(slabs.begin());
	auto ends(slabs.end());
	
	for (; slab != ends; ++slab) {
		slab->primal.dof->clear();
		slab->dual.dof->clear();
	}
}


template<int dim, int spacedim>
void
Grid_DWR<dim,spacedim>::
set_data(
	const unsigned int p_primal,
	const unsigned int p_dual) {
	gdata.p_primal = p_primal;
	gdata.p_dual   = p_dual;
}


/// Initialize list of grids
template<int dim, int spacedim>
void
Grid_DWR<dim,spacedim>::
initialize_grids(
	const double &t0,
	const double &T,
	const double &tau_n) {
	if ( slabs.empty() ) {
		std::cout << "Initialize slabs objects " << std::endl;
		
		// determine initial time intervals
		unsigned int numoftimeintervals;
		numoftimeintervals = static_cast<unsigned int>(std::floor(
			(T-t0)/tau_n
		));
		if (std::abs((numoftimeintervals*tau_n)-(T-t0))
			>= std::numeric_limits< double >::epsilon()*T) {
			numoftimeintervals += 1;
		}
		
		// init spatial "grids" of each slab
		for (unsigned int i{1}; i<= numoftimeintervals; ++i) {
			slabs.emplace_back();
			auto &slab = slabs.back();
			
			slab.tria = std::make_shared< dealii::Triangulation<dim> >(
				typename dealii::Triangulation<dim>::MeshSmoothing(
					dealii::Triangulation<dim>::smoothing_on_refinement
				)
			);
			
			slab.primal.dof = std::make_shared< dealii::DoFHandler<dim> > (*(slab.tria));
			slab.dual.dof = std::make_shared< dealii::DoFHandler<dim> > (*(slab.tria));
			
			slab.primal.constraints = std::make_shared< dealii::ConstraintMatrix > ();
			slab.dual.constraints = std::make_shared< dealii::ConstraintMatrix > ();
			
			slab.primal.sp = std::make_shared< dealii::SparsityPattern >();
			slab.dual.sp = std::make_shared< dealii::SparsityPattern >();
			
			slab.primal.fe = std::make_shared< dealii::FE_Q<dim> > (gdata.p_primal);
			slab.dual.fe = std::make_shared< dealii::FE_Q<dim> > (gdata.p_dual);
			
			slab.primal.mapping = std::make_shared< dealii::MappingQ<dim> > (gdata.p_primal);
			slab.dual.mapping = std::make_shared< dealii::MappingQ<dim> > (gdata.p_dual);
		}
		
		// init temporal "grids" of each slab
		{
			unsigned int n{1};
			for (auto &slab : slabs) {
				slab.t_m = (n-1)*tau_n+t0;
				slab.t_n = n*tau_n + t0;
				++n;
			}
			
			auto &last_slab = slabs.back();
			if ( std::abs(last_slab.t_n - T) >= std::numeric_limits< double >::epsilon()*T) {
				last_slab.t_n = T;
			}
		}
	}
	else {
		Assert(false, dealii::ExcMessage("space-time grid already initialized"));
	}
}


/// Generate grid. Throws Exception in base class.
template<int dim, int spacedim>
void
Grid_DWR<dim,spacedim>::
generate() {
	// TODO: Throws Exception in base class.
	
	// TODO: remove the following code and set an assert
	const double a(0.);
	const double b(1.);
	
	auto slab(slabs.begin());
	auto ends(slabs.end());
	
	for (; slab != ends; ++slab) {
		dealii::GridGenerator::hyper_cube(
			*(slab->tria),
			a,b,false
		);
	}
}


/// Global refinement.
template<int dim, int spacedim>
void
Grid_DWR<dim,spacedim>::
refine_global(const unsigned int n) {
	auto slab(slabs.begin());
	auto ends(slabs.end());
	
	for (; slab != ends; ++slab) {
		slab->tria->refine_global(n);
	}
}


/// Set boundary indicators
template<int dim, int spacedim>
void
Grid_DWR<dim,spacedim>::
set_boundary_indicators() {
	// TODO: remove the following code and set an assert
	
	// set boundary indicators
	auto slab(slabs.begin());
	auto ends(slabs.end());
	
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


/// Distribute.
template<int dim, int spacedim>
void
Grid_DWR<dim,spacedim>::
distribute() {
	// Distribute the degrees of freedom (dofs)
	// Start itererator over all list-elements
	auto slab(slabs.begin());
	auto ends(slabs.end());
	for (; slab != ends; ++slab) {
		////////////////////////////////////////////////////////////////////////////
		// distribute primal dofs, create constraints and sparsity pattern sp
		{
			Assert(slab->primal.dof.use_count(), dealii::ExcNotInitialized());
			Assert(slab->primal.fe.use_count(), dealii::ExcNotInitialized());
			slab->primal.dof->distribute_dofs(*(slab->primal.fe));
			
			DTM::pout
				<< "grid: dof: primal mesh: n_dofs = " << slab->primal.dof->n_dofs()
				<< std::endl;
			
			// setup constraints like boundary values or hanging nodes
			Assert(slab->primal.constraints.use_count(), dealii::ExcNotInitialized());
			slab->primal.constraints->clear();
			slab->primal.constraints->reinit();
			
			dealii::DoFTools::make_hanging_node_constraints(
				*(slab->primal.dof),
				*(slab->primal.constraints)
			);
			
			slab->primal.constraints->close();
			
			// create sparsity pattern
			dealii::DynamicSparsityPattern dsp(
				slab->primal.dof->n_dofs(), slab->primal.dof->n_dofs()
			);
			dealii::DoFTools::make_sparsity_pattern(
				*(slab->primal.dof),
				dsp,
				*(slab->primal.constraints),
				false
			);
			
			Assert(slab->primal.sp.use_count(), dealii::ExcNotInitialized());
			slab->primal.sp->copy_from(dsp);
		}
		
		////////////////////////////////////////////////////////////////////////////
		// distribute dual dofs, create constraints and sparsity pattern sp
		{
			Assert(slab->dual.dof.use_count(), dealii::ExcNotInitialized());
			Assert(slab->dual.fe.use_count(), dealii::ExcNotInitialized());
			slab->dual.dof->distribute_dofs(*(slab->dual.fe));
			
			DTM::pout
				<< "grid: dof: dual mesh: n_dofs = " << slab->dual.dof->n_dofs()
				<< std::endl;
			
			// setup constraints like boundary values or hanging nodes
			Assert(slab->dual.constraints.use_count(), dealii::ExcNotInitialized());
			slab->dual.constraints->clear();
			slab->dual.constraints->reinit();
			
			dealii::DoFTools::make_hanging_node_constraints(
				*(slab->dual.dof),
				*(slab->dual.constraints)
			);
			
			slab->dual.constraints->close();
			
			// create sparsity pattern
			dealii::DynamicSparsityPattern dsp(
				slab->dual.dof->n_dofs(), slab->dual.dof->n_dofs()
			);
			dealii::DoFTools::make_sparsity_pattern(
				*(slab->dual.dof),
				dsp,
				*(slab->dual.constraints),
				false
			);
			
			Assert(slab->dual.sp.use_count(), dealii::ExcNotInitialized());
			slab->dual.sp->copy_from(dsp);
		}
	} // end for-loop slab
}


} // namespaces

#include "Grid_DWR.inst.in"
