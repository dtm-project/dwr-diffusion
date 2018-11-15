/**
 * @file L2_NeumannConstrainedAssembly.tpl.hh
 * @author Uwe Koecher (UK)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 * 
 * @date 2018-11-15, Neumann, UK
 * @date 2018-03-08, included from ewave, UK
 * @date 2017-10-26, auto mode, UK
 * @date 2017-09-13, xwave/ewave, UK
 * @date 2015-05-19, AWAVE/C++.11, UK
 * @date 2012-08-31
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


#ifndef __L2_NeumannConstrainedAssembly_tpl_hh
#define __L2_NeumannConstrainedAssembly_tpl_hh

// PROJECT includes

// DEAL.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/vector.h>

// C++ includes
#include <memory>
#include <vector>


namespace heat {
namespace Assemble {
namespace L2 {
namespace NeumannConstrained {

namespace Assembly {
namespace Scratch {

template<int dim>
struct NeumannConstrainedAssembly {
	NeumannConstrainedAssembly(
		const dealii::FiniteElement<dim> &fe,
		const dealii::Mapping<dim> &mapping,
		const dealii::Quadrature<dim-1> &quad_face,
		const dealii::UpdateFlags &uflags_face
	);
	
	NeumannConstrainedAssembly(const NeumannConstrainedAssembly &scratch);
	
	dealii::FEFaceValues<dim> fe_face_values;
	std::vector<double>       phi;
	double                    JxW;
	double                    u_N;
	
	// other
	unsigned int q;
	unsigned int component;
	unsigned int k;
	unsigned int i;
};

} // namespace Scratch
namespace CopyData {

template<int dim>
struct NeumannConstrainedAssembly {
	NeumannConstrainedAssembly(const dealii::FiniteElement<dim> &fe);
	NeumannConstrainedAssembly(const NeumannConstrainedAssembly &copydata);
	
	dealii::Vector<double> fi_vi_vector;
	std::vector<unsigned int> local_dof_indices;
};

} // namespace CopyData
} // namespace Assembly
////////////////////////////////////////////////////////////////////////////////


/** Assembler.
 *
 */
template<int dim>
class Assembler {
public:
	Assembler(
		std::shared_ptr< dealii::Vector<double> > u_N,
		std::shared_ptr< dealii::DoFHandler<dim> > dof,
		std::shared_ptr< dealii::FiniteElement<dim> > fe,
		std::shared_ptr< dealii::Mapping<dim> > mapping,
		std::shared_ptr< dealii::ConstraintMatrix > constraints
	);
	
	~Assembler() = default;
	
	void set_function(
		std::shared_ptr< dealii::Function<dim> > u_N
	);
	
	/** Assemble vector.
	 *  If @param n_quadrature_points = 0 is given,
	 *  the dynamic default fe.tensor_degree()+1 will be used.
	 */
	void assemble(
		const double time,
		const unsigned int n_quadrature_points = 0,
		const bool quadrature_points_auto_mode = true
	);
	
protected:
	void local_assemble_cell(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		Assembly::Scratch::NeumannConstrainedAssembly<dim> &scratch,
		Assembly::CopyData::NeumannConstrainedAssembly<dim> &copydata
	);

	void copy_local_to_global_cell(
		const Assembly::CopyData::NeumannConstrainedAssembly<dim> &copydata
	);
	
private:
	std::shared_ptr< dealii::Vector<double> > u_N;
	
	std::shared_ptr< dealii::DoFHandler<dim> > dof;
	std::shared_ptr< dealii::FiniteElement<dim> > fe;
	std::shared_ptr< dealii::Mapping<dim> > mapping;
	std::shared_ptr< dealii::ConstraintMatrix > constraints;
	dealii::UpdateFlags uflags_face;
	
	struct {
		std::shared_ptr< dealii::Function<dim> > u_N;
	} function;
};

}}}} // namespaces

#endif
