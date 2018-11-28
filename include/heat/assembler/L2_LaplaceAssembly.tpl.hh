/**
 * @file L2_LaplaceAssembly.tpl.hh
 * @author Uwe Koecher (UK)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 * 
 * @date 2018-03-08, included from ewave, UK
 * @date 2017-10-23, ewave, UK
 * @date 2015-05-18, AWAVE/C++.11, UK
 * @date 2014-04-09, Tensor, UK
 * @date 2012-03-13, UK
 *
 * @brief Purpose: Assemble weak Laplace operator @f$ (\nabla v, \epsilon \nabla u) @f$
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


#ifndef __L2_LaplaceAssembly_tpl_hh
#define __L2_LaplaceAssembly_tpl_hh

// PROJECT includes

// DEAL.II includes
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

// C++ includes
#include <memory>
#include <vector>


namespace heat {
namespace Assemble {
namespace L2 {
namespace Laplace {

namespace Assembly {
namespace Scratch {

/// Struct for scratch on local cell matrix.
template<int dim>
struct LaplaceAssembly {
	LaplaceAssembly(
		const dealii::FiniteElement<dim> &fe,
		const dealii::Mapping<dim> &mapping,
		const dealii::Quadrature<dim> &quad,
		const dealii::UpdateFlags &uflags_cell
	);
	
	LaplaceAssembly(const LaplaceAssembly &scratch);
	
	dealii::FEValues<dim>               fe_values;
	std::vector<dealii::Tensor<1,dim> > grad_phi;
	unsigned int                        dofs_per_cell;
	double                              JxW;
	double                              epsilon;
	
	// other
	unsigned int q;
	unsigned int k;
	unsigned int i;
	unsigned int j;
};

} // namespace Scratch
namespace CopyData {

/// Struct for copydata on local cell matrix.
template<int dim>
struct LaplaceAssembly{
	LaplaceAssembly(const dealii::FiniteElement<dim> &fe);
	LaplaceAssembly(const LaplaceAssembly &copydata);

	dealii::FullMatrix<double> vi_ui_matrix;
	std::vector<dealii::types::global_dof_index> local_dof_indices;
};

} // namespace CopyData
} // namespace Assembly
////////////////////////////////////////////////////////////////////////////////


/**
 * The stiffness matrix \f$ \boldsymbol A \f$ for the corresponding
 * index sets \f$ 1 \le i \le N_u \f$ of test and \f$ 1 \le j \le N_u \f$ 
 * for trial basis function is given by
 * \f[
 * \boldsymbol A = ( a_{i j} )_{i j}\,,\quad
 * a_{i j} = \displaystyle \sum_{K \in \mathcal{T}_h}
 * \displaystyle \sum_{i=1}^{N_u}  \displaystyle \sum_{j=1}^{N_u}
 * \displaystyle \int_K
 * \nabla\varphi^{i}(\boldsymbol x)\, \varepsilon(\boldsymbol x)\,\nabla\varphi^{j}(\boldsymbol x)\,
 * \,\text{d} \boldsymbol x\,,
 * \f]
 * where  \f$ N_u \f$ denotes the degrees of freedom in space for a single 
 * temporal degree of freedem of the fully discrete solution 
 * \f$ u_{\tau, h}^{\text{dG}} \f$.
 */
template<int dim>
class Assembler {
public:
	Assembler(
		std::shared_ptr< dealii::SparseMatrix<double> > A,
		std::shared_ptr< dealii::DoFHandler<dim> > dof,
		std::shared_ptr< dealii::FiniteElement<dim> > fe,
		std::shared_ptr< dealii::Mapping<dim> > mapping,
		std::shared_ptr< dealii::ConstraintMatrix > constraints
	);
	
	~Assembler() = default;
	
	void set_epsilon_function(
		std::shared_ptr< dealii::Function<dim> > epsilon
	);
	
	/** Assemble matrix. Matrix must be initialized before!
	 *  If @param n_quadrature_points = 0 is given,
	 *  the dynamic default fe.tensor_degree()+1 will be used.
	 */
	void assemble(
		const unsigned int n_quadrature_points = 0
	);
	
protected:
	void local_assemble_cell(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		Assembly::Scratch::LaplaceAssembly<dim> &scratch,
		Assembly::CopyData::LaplaceAssembly<dim> &copydata
	);

	void copy_local_to_global_cell(
		const Assembly::CopyData::LaplaceAssembly<dim> &copydata
	);
	
private:
	std::shared_ptr< dealii::SparseMatrix<double> > A;
	
	std::shared_ptr< dealii::DoFHandler<dim> > dof;
	std::shared_ptr< dealii::FiniteElement<dim> > fe;
	std::shared_ptr< dealii::Mapping<dim> > mapping;
	std::shared_ptr< dealii::ConstraintMatrix > constraints;
	dealii::UpdateFlags uflags;
	
	struct {
		std::shared_ptr< dealii::Function<dim> > epsilon;
	} function;
};

}}}} // namespaces


#endif
