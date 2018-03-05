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
	auto Inth(In.begin());
	auto endIn(In.end());
	for (; Inth != endIn; ++Inth) {
		Inth->primal.dof->clear();
		Inth->dual.dof->clear();
	} //end for-loop Inth
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
initialize_grids(const double &t0,
				 const double &T,
				 const double &tau_n) {
	if ( In.empty() ) {
		std::cout << "Initialize In objects " << std::endl;
		
		// determine initial time intervals
		unsigned int numoftimeintervals;
		numoftimeintervals = static_cast<unsigned int>(std::floor(
			(T-t0)/tau_n
		));
		if (std::abs((numoftimeintervals*tau_n)-(T-t0)) >= std::numeric_limits< double >::epsilon()*T) {
			numoftimeintervals += 1;
		}
		
		// init space "grid" on each I_n
		for (unsigned int i{1}; i<= numoftimeintervals; ++i) {
			In.emplace_back();
			auto &element = In.back();
			element.tria = std::make_shared< dealii::Triangulation<dim> >(
				typename dealii::Triangulation<dim>::MeshSmoothing(
					dealii::Triangulation<dim>::smoothing_on_refinement
				)
			);
			
			element.primal.dof = std::make_shared< dealii::DoFHandler<dim> > (*(element.tria));
			element.dual.dof = std::make_shared< dealii::DoFHandler<dim> > (*(element.tria));
			element.primal.constraints = std::make_shared< dealii::ConstraintMatrix > ();
			element.dual.constraints = std::make_shared< dealii::ConstraintMatrix > ();
			element.primal.sp = std::make_shared< dealii::SparsityPattern >();
			element.dual.sp = std::make_shared< dealii::SparsityPattern >();
			element.primal.fe = std::make_shared< dealii::FE_Q<dim> > (gdata.p_primal);
			element.dual.fe = std::make_shared< dealii::FE_Q<dim> > (gdata.p_dual);
			element.primal.mapping = std::make_shared< dealii::MappingQ<dim> > (gdata.p_primal);
			element.dual.mapping = std::make_shared< dealii::MappingQ<dim> > (gdata.p_dual);
		}// end for loop i
		
		{ // init time "grid"
			unsigned int n{1};
			for (auto &element : In) {
				element.t_m = (n-1)*tau_n+t0;
				element.t_n = n*tau_n + t0;
				++n;
			}
			
			auto &last_element = In.back();
			if ( std::abs(last_element.t_n - T) >= std::numeric_limits< double >::epsilon()*T) {
				last_element.t_n = T;
			}
		}
	} // end if (in.empty())
}


/// Generate grid. Throws Exception in base class.
template<int dim, int spacedim>
void
Grid_DWR<dim,spacedim>::
generate() {

	const double a(0.);
	const double b(1.);
	auto Inth(In.begin());
	auto endIn(In.end());
		for (; Inth != endIn; ++Inth) {
			dealii::GridGenerator::hyper_cube(
				*(Inth->tria),
				a,b,false
			);
	} //end for loop i
	
}


/// Global refinement.
template<int dim, int spacedim>
void
Grid_DWR<dim,spacedim>::
refine_global(const unsigned int n) {
	
	auto Inth(In.begin());
	auto endIn(In.end());
	for (; Inth != endIn; ++Inth) {
		Inth->tria->refine_global(n);
	} //end for loop i
}


/// Set boundary indicators
template<int dim, int spacedim>
void
Grid_DWR<dim,spacedim>::
set_boundary_indicators() {
	// set boundary indicators
	
	auto Inth(In.begin());
	auto endIn(In.end());
	for (; Inth != endIn; ++Inth) {
	auto cell(Inth->tria->begin_active());
	auto endc(Inth->tria->end());
	
	for (; cell != endc; ++cell) {
	if (cell->at_boundary()) {
	for (unsigned int face(0); face < dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
		if (cell->face(face)->at_boundary()) {
			cell->face(face)->set_boundary_id(
				static_cast<dealii::types::boundary_id> (
					Heat::types::boundary_id::Dirichlet)
			);
		}
	}}}//end loop cell
	}//end loop Inth
}

template<int dim, int spacedim>
void
Grid_DWR<dim,spacedim>::
output_boundary_id() {
	
	auto Inth(In.begin());
	auto endIn(In.end());
	for (; Inth != endIn; ++Inth) {
	auto cell(Inth->tria->begin_active());
	auto endc(Inth->tria->end());
	
		for (; cell != endc; ++cell) {
		for (unsigned int face_number = 0; face_number < dealii::GeometryInfo<dim>::faces_per_cell; ++face_number) {
			std::cout << "center_x = " << cell->face(face_number)->center()(0) << std::endl;
			std::cout << "center_y = " << cell->face(face_number)->center()(1) << std::endl;
			std::cout << "Einfaerbung = " << cell->face(face_number)->boundary_id() << std::endl;
		}}//end loop cell
		std::cout << "======================================================" << std::endl;
	}
}



/// Distribute.
template<int dim, int spacedim>
void
Grid_DWR<dim,spacedim>::
distribute() {
	// Distribute the degrees of freedom (dofs)
	// Start itererator over all list-elements
	auto Inth(In.begin());
	auto endIn(In.end());
	for (; Inth != endIn; ++Inth) {
	
		////////////////////////////////////////////////////////////////////////////
		// distribute primal dofs, create constraints and sparsity pattern sp
		{
			Assert(Inth->primal.dof.use_count(), dealii::ExcNotInitialized());
			Assert(Inth->primal.fe.use_count(), dealii::ExcNotInitialized());
			Inth->primal.dof->distribute_dofs(*(Inth->primal.fe));
			
			DTM::pout << "grid: dof: primal mesh: n_dofs = " << Inth->primal.dof->n_dofs() << std::endl;
			
			// setup constraints like boundary values or hanging nodes
			Assert(Inth->primal.constraints.use_count(), dealii::ExcNotInitialized());
			Inth->primal.constraints->clear();
			Inth->primal.constraints->reinit();
			
			dealii::DoFTools::make_hanging_node_constraints(*(Inth->primal.dof), *(Inth->primal.constraints));
			
			Inth->primal.constraints->close();
			
			// Now we create a sparsity pattern, which we will use to initialise
			// our system matrix (for the assembly step).
			// See deal.II step-2 tutorial for details.
			dealii::DynamicSparsityPattern dsp(Inth->primal.dof->n_dofs(), Inth->primal.dof->n_dofs());
			dealii::DoFTools::make_sparsity_pattern(*(Inth->primal.dof), dsp, *(Inth->primal.constraints), false);
			
			Assert(Inth->primal.sp.use_count(), dealii::ExcNotInitialized());
			Inth->primal.sp->copy_from(dsp);
		}
		
		////////////////////////////////////////////////////////////////////////////
		// distribute dual dofs, create constraints and sparsity pattern sp
		{
			Assert(Inth->dual.dof.use_count(), dealii::ExcNotInitialized());
			Assert(Inth->dual.fe.use_count(), dealii::ExcNotInitialized());
			Inth->dual.dof->distribute_dofs(*(Inth->dual.fe));
			
			DTM::pout << "grid: dof: dual mesh: n_dofs = " << Inth->dual.dof->n_dofs() << std::endl;
			
			// setup constraints like boundary values or hanging nodes
			Assert(Inth->dual.constraints.use_count(), dealii::ExcNotInitialized());
			Inth->dual.constraints->clear();
			Inth->dual.constraints->reinit();
			
			dealii::DoFTools::make_hanging_node_constraints(*(Inth->dual.dof), *(Inth->dual.constraints));
			
			Inth->dual.constraints->close();
			
			// Now we create a sparsity pattern, which we will use to initialise
			// our system matrix (for the assembly step).
			// See deal.II step-2 tutorial for details.
			dealii::DynamicSparsityPattern dsp(Inth->dual.dof->n_dofs(), Inth->dual.dof->n_dofs());
			dealii::DoFTools::make_sparsity_pattern(*(Inth->dual.dof), dsp, *(Inth->dual.constraints), false);
			
			Assert(Inth->dual.sp.use_count(), dealii::ExcNotInitialized());
			Inth->dual.sp->copy_from(dsp);
		}
	} // end for-loop Inth
}


} // namespaces

#include "Grid_DWR.inst.in"
