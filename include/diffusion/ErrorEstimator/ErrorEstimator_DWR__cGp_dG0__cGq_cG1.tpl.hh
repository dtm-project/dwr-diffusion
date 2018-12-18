/**
 * @file ErrorEstimator_DWR__cGp_dG0__cGq_cG1.tpl.hh
 *
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 *
 * @date 2018-12-18, renamed from heat to diffusion, MPB, UK
 * @date 2018-11-15, add inhomogeneous Neumann, MPB, UK
 * @date 2018-03-16, ErrorEstimator class for heat (final), UK, MPB
 * @date 2018-03-13, new development ErrorEstimator class for heat (begin), UK, MPB
 * @date 2018-03-13, fork from DTM++/dwr-poisson, UK
 *
 * @date 2017-11-08, ErrorEstimator class (Poisson), UK, MPB
 * @date 2016-08-16, ErrorEstimator class (Poisson), UK
 * @date 2016-08-11, Poisson / DWR from deal.II/step-14 and DTM++, UK
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

#ifndef __ErrorEstimator_DWR__cGp_dG0__cGq_cG1_tpl_hh
#define __ErrorEstimator_DWR__cGp_dG0__cGq_cG1_tpl_hh

// PROJECT includes
#include <diffusion/grid/Grid_DWR.tpl.hh>
#include <diffusion/parameters/ParameterSet.hh>

// DTM++ includes
#include <DTM++/types/storage_data_vectors.tpl.hh>

// DEAL.II includes
#include <deal.II/base/function.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

// C++ includes
#include <memory>
#include <vector>

namespace diffusion {
namespace dwr {

namespace cGp_dG0 { // primal
namespace cGq_cG1 { // dual

namespace Assembly {

namespace Scratch {

/// Struct for scratch on local error estimate on cell
template<int dim>
struct ErrorEstimateOnCell {
	ErrorEstimateOnCell(
		const dealii::FiniteElement<dim> &fe,
		const dealii::Mapping<dim> &mapping,
		const dealii::Quadrature<dim> &quad,
		const dealii::UpdateFlags &uflags
	);
	
	ErrorEstimateOnCell(const ErrorEstimateOnCell &scratch);
	
	// data structures of current cell
	dealii::FEValues<dim>               fe_values;
	
	std::vector< dealii::types::global_dof_index > local_dof_indices;
	
	// shape fun scratch:
	std::vector<double>                 phi;
	std::vector<dealii::Tensor<1,dim> > grad_phi;
	dealii::Tensor<2,dim>               hessian_phi;
	std::vector<double>                 laplace_phi;
	
	// local dof scratch:
	std::vector<double>                 local_u0;
	std::vector<double>                 local_z0;
	std::vector<double>                 local_Rz0;
	
	std::vector<double>                 local_um; // u^-(t_m)
	std::vector<double>                 local_up; // u^+(t_m)
	
	std::vector<double>                 local_zm;
	std::vector<double>                 local_Rzm;
	
	// function eval scratch:
	double value_f;
	double value_epsilon;
	dealii::Tensor<1,dim> grad_epsilon;
	double value_density;
	
	double val_R_u_kh_j;
	double val_u_kh_j;
	double val_z_Rz_j;
	
	// other:
	double JxW;
	
	unsigned int q;
	unsigned int d;
	unsigned int j;
};


/// Struct for scratch on local error estimate on face
template<int dim>
struct ErrorEstimateOnFace {
	ErrorEstimateOnFace(
		const dealii::FiniteElement<dim> &fe,
		const dealii::Mapping<dim> &mapping,
		const dealii::Quadrature<dim-1> &quad,
		const dealii::UpdateFlags &uflags
	);
	
	ErrorEstimateOnFace(const ErrorEstimateOnFace &scratch);
	
	// data structures of current face on cell (+)
	dealii::FEFaceValues<dim>            fe_values_face;
	dealii::FESubfaceValues<dim>         fe_values_subface;
	
	std::vector< dealii::types::global_dof_index > local_dof_indices;
	
	// shape fun scratch:
	std::vector<double>                  phi;
	std::vector<dealii::Tensor<1,dim> >  grad_phi;
	dealii::Tensor<1,dim>                normal_vector;
	
	// local dof scratch:
	std::vector<double>                  local_u0;
	std::vector<double>                  local_z0;
	std::vector<double>                  local_Rz0;
	
	// data structures of neighboring face of cell (-)
	dealii::FEFaceValues<dim>            neighbor_fe_values_face;
	
	std::vector< dealii::types::global_dof_index > neighbor_local_dof_indices;
	
	std::vector<dealii::Tensor<1,dim> >  neighbor_grad_phi;
	
	std::vector<double>                  neighbor_local_u0;
	
	// function eval scratch:
	double value_epsilon;
	double value_u_D;
	double value_u_N;
	
	double val_uh;
	double val_grad_uh;
	double val_grad_zh;
	double val_face_jump_grad_u;
	double val_z_Rz_j;
	
	// other
	double JxW;
	
	unsigned int q;
	unsigned int j;
	unsigned int subface_no;
};


template<int dim>
struct ErrorEstimates {
	ErrorEstimates(
		const dealii::FiniteElement<dim> &fe,
		const dealii::Mapping<dim>       &mapping,
		const dealii::Quadrature<dim>    &quad_cell,
		const dealii::Quadrature<dim-1>  &quad_face,
		const dealii::UpdateFlags        &uflags_cell,
		const dealii::UpdateFlags        &uflags_face
	);
	
	ErrorEstimates(const ErrorEstimates &scratch);
	
	ErrorEstimateOnCell<dim> cell;
	ErrorEstimateOnFace<dim> face;
	
	// other
	unsigned int face_no;
};

} // namespace Scratch

namespace CopyData {

/// Struct for copydata on local cell matrix.
template<int dim>
struct ErrorEstimateOnCell{
	ErrorEstimateOnCell() = default;
	ErrorEstimateOnCell(const ErrorEstimateOnCell &copydata);
	
	typename dealii::DoFHandler<dim>::cell_iterator cell;
	double value;
};


/// Struct for copydata on local cell matrix.
template<int dim>
struct ErrorEstimateOnFace{
	ErrorEstimateOnFace() = default;
	ErrorEstimateOnFace(const ErrorEstimateOnFace &copydata);

	typename dealii::DoFHandler<dim>::face_iterator face;
	double value;
};


template<int dim>
struct ErrorEstimates {
	ErrorEstimates() = default;
	ErrorEstimates(const ErrorEstimates &copydata);
	
	ErrorEstimateOnCell<dim> cell;
	ErrorEstimateOnFace<dim> face;
};

} // namespace CopyData

} // namespace Assembly
////////////////////////////////////////////////////////////////////////////////

/**
 * Implements the computation of the a posteriori error estimator \f$ \eta^l \f$
 * given by:
 * \f[
 * \tilde{\eta}^\ell := \displaystyle \sum_{n=1}^{N}\displaystyle \sum_{K \in \mathcal{T}_h}
 * \tilde{\eta}_K^{n,\ell}\,,
 * \f]
 * where \f$ \tilde{\eta}_K^{n,\ell} \f$ is given by
 * \f[
 * \tilde{\eta}_K^{n,\ell}:= \int_{I_n}\left\{\left( R(u_{\tau h}^{n,\ell}),
 * z_{\tau H}^{n,\ell}-I_h z_{\tau H}^{n,\ell}\right)_K
 * - \left( E(u_{\tau h}^{n,\ell}),
 * z_{\tau H}^{n,\ell}-I_h z_{\tau H}^{n,\ell}\right)_{\partial K}
 * - \left( g-g_{\tau h}^{n,\ell},
 * \nabla z_{\tau H}^{n,\ell}\cdot\boldsymbol n\right)_{\partial K \subset \Gamma_D} 
 * \right\}
 * \text{d} t
 * - \left( [u_{\tau h}^{\ell}]_{n-1},
 * z_{\tau H}^{n-1,\ell}-I_h z_{\tau H}^{n-1,\ell}\right)_K
 * \f]
 * where the cell and edge residuals are given by
 * \f[
 * R(u_{\tau h}^{n,l})|_K := f - \partial_t u_{\tau h}^{n,\ell} 
 * + \nabla\cdot(\varepsilon \nabla u_{\tau h}^{n,\ell}) \\
 * E(u_{\tau h}^{n,l})|_F := \begin{cases}
 * \frac{1}{2}\boldsymbol n\cdot[\varepsilon \nabla u_{\tau h}^{n,\ell}] 
 * \quad \text{if } F \subset \partial K\backslash \partial \Omega \,, \\
 * h - \varepsilon \nabla u_{\tau h}^{n,\ell} \cdot\boldsymbol n  
 * \quad \text{if } F \subset \partial \Omega_N \,.
 * \end{cases}
 * \f]
 * Here, \f$ I_h z_{\tau H}^{n,\ell} \f$ is the interpolation of the dual solution
 * \f$ z_{\tau H}^{n,\ell} \f$
 * into the primal finite element space.
 * Furthermore,
 * \f$ [ \, \cdot \, ]_n \f$ denotes a trace operator for the jump
 * of a discontinuous function in \f$ t_n \f$ as given by
 * \f[ 
 * \begin{array}{l@{\quad}l@{\quad}l}
 * \displaystyle [u_{\tau h}^\ell]_n := u_{\tau h}^{n,\ell,+} - u_{\tau h}^{n,\ell,-}\,, &
 * \displaystyle u_{\tau h}^{n,\ell,-} := \lim_{t \to t_n-0} u_{\tau h}^\ell(t)\,, &
 * \displaystyle u_{\tau h}^{n,\ell,+} := \lim_{t \to t_n+0} u_{\tau h}^\ell(t)\,.\\[1.5ex]
 * \end{array}
 * \f]
 * as well as \f$ [\varepsilon \nabla u_{\tau h}^{n,\ell}] \f$ denotes a trace 
 * operator for the jump across an inter-element edge (\f$ d=2 \f$) or
 * face (\f$ d=3 \f$), given by
 * \f[ 
 * [\varepsilon \nabla u_{\tau h}^{n,\ell}] := \left(
 * \varepsilon\nabla u_{\tau h}^{n,\ell}|_{F\cap K} - \varepsilon\nabla u_{\tau h}^{n,\ell}|_{F\cap K'}
 * \right) \cdot \boldsymbol n\,,
 * \f]
 * where \f$ K, K' \in \mathcal{T}_h  \f$ are two neighboring cells with common
 * edge / face \f$ F \f$ and \f$ \boldsymbol n \f$ denotes the
 * normal unit vector pointing from \f$ K \f$ to \f$ K' \f$.
 * The discrete function \f$ g_{\tau h} \f$ is an appropriate finite element 
 * approximation of the extension \f$ \tilde{g} \f$ in the sense that the trace 
 * of \f$ \tilde{g} \f$ equals \f$ g \f$ on \f$ \Gamma_D \f$.
 */
template<int dim>
class ErrorEstimator {
public:
	ErrorEstimator() = default;
	virtual ~ErrorEstimator() = default;
	
	virtual void estimate(
		std::shared_ptr< dealii::Function<dim> > density,
		std::shared_ptr< dealii::Function<dim> > epsilon,
		std::shared_ptr< dealii::Function<dim> > f,
		std::shared_ptr< dealii::Function<dim> > u_D,
		std::shared_ptr< dealii::Function<dim> > u_N,
		std::shared_ptr< dealii::Function<dim> > u_0,
		std::shared_ptr< diffusion::Grid_DWR<dim,1> > grid,
		std::shared_ptr< diffusion::dwr::ParameterSet > parameter_set,
		std::shared_ptr< DTM::types::storage_data_vectors<1> > u,
		std::shared_ptr< DTM::types::storage_data_vectors<2> > z,
		std::shared_ptr< DTM::types::storage_data_vectors<1> > eta
	);
	
protected:
	/// evaluate solution dof vector u^primal(t) on primal solution space
	virtual void primal_get_u_t_on_slab(
		const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<1>::iterator &u,
		const double &t,
		std::shared_ptr< dealii::Vector<double> > &u_result
	);
	
	/// evaluate solution dof vector I^dual( u^primal(t) ) on dual solution space
	// NOTE: this function needs: primal_get_u_t_on_slab
	virtual void dual_get_u_t_on_slab(
		const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<1>::iterator &u,
		const double &t,
		std::shared_ptr< dealii::Vector<double> > &dual_u_result
	);
	
	/// evaluate solution dof vector z^dual(t) on dual solution space
	virtual void dual_get_z_t_on_slab(
		const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<2>::iterator &z,
		const double &t,
		std::shared_ptr< dealii::Vector<double> > &dual_z_result
	);
	
	/// evaluate solution dof vector I^dual( R^primal(z^dual)(t) ) on dual solution space
	virtual void dual_get_z_t_on_slab_after_restriction_to_primal_space(
		const typename DTM::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<2>::iterator &z,
		const double &t,
		std::shared_ptr< dealii::Vector<double> > &dual_z_result_after_restriction
	);
	
	////////////////////////////////////////////////////////////////////////////
	// assemble local functions:
	//
	
	virtual void assemble_local_error(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		Assembly::Scratch::ErrorEstimates<dim> &scratch,
		Assembly::CopyData::ErrorEstimates<dim> &copydata
	);
	
	virtual void assemble_error_on_cell(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		Assembly::Scratch::ErrorEstimateOnCell<dim> &scratch,
		Assembly::CopyData::ErrorEstimateOnCell<dim> &copydata
	);
	
	virtual void assemble_error_on_dirichlet_boundary_face(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		const unsigned int face_no,
		Assembly::Scratch::ErrorEstimateOnFace<dim> &scratch,
		Assembly::CopyData::ErrorEstimateOnFace<dim> &copydata
	);
	
	virtual void assemble_error_on_neumann_boundary_face(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		const unsigned int face_no,
		Assembly::Scratch::ErrorEstimateOnFace<dim> &scratch,
		Assembly::CopyData::ErrorEstimateOnFace<dim> &copydata
	);
	
	virtual void assemble_error_on_regular_face(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		const unsigned int face_no,
		Assembly::Scratch::ErrorEstimateOnFace<dim> &scratch,
		Assembly::CopyData::ErrorEstimateOnFace<dim> &copydata
	);
	
	virtual void assemble_error_on_irregular_face(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		const unsigned int face_no,
		Assembly::Scratch::ErrorEstimateOnFace<dim> &scratch,
		Assembly::CopyData::ErrorEstimateOnFace<dim> &copydata
	);
	
	virtual void copy_local_error(
		const Assembly::CopyData::ErrorEstimates<dim> &copydata
	);
	
	////////////////////////////////////////////////////////////////////////////
	// internal data structures:
	//
	
	std::shared_ptr< diffusion::Grid_DWR<dim,1> > grid;
	
	struct {
		struct {
			std::shared_ptr< DTM::types::storage_data_vectors<1> > u;
		} storage;
	} primal;
	
	struct {
		struct {
			std::shared_ptr< DTM::types::storage_data_vectors<2> > z;
		} storage;
	} dual;
	
	struct {
		struct {
			std::shared_ptr< DTM::types::storage_data_vectors<1> > eta;
		} storage;
	} error_estimator;
	
	struct {
		std::shared_ptr< dealii::Function<dim> > density;
		std::shared_ptr< dealii::Function<dim> > epsilon;
		std::shared_ptr< dealii::Function<dim> > f;
		std::shared_ptr< dealii::Function<dim> > u_D;
		std::shared_ptr< dealii::Function<dim> > u_N;
		std::shared_ptr< dealii::Function<dim> > u_0;
	} function;
	
	// parameter set
	std::shared_ptr< diffusion::dwr::ParameterSet > parameter_set;
	
	double tau_n;
	
	std::shared_ptr< dealii::Vector<double> > dual_um_on_tm;
	
	std::shared_ptr< dealii::Vector<double> > dual_z_on_tm;
	std::shared_ptr< dealii::Vector<double> > dual_Rz_on_tm;
	
	std::shared_ptr< dealii::Vector<double> > dual_z_on_t0;
	std::shared_ptr< dealii::Vector<double> > dual_Rz_on_t0;
	
	std::shared_ptr< dealii::Vector<double> > dual_u_on_t0;
	std::shared_ptr< dealii::Vector<double> > dual_up_on_tm;
	
	std::shared_ptr< dealii::Vector<double> > dual_uD_on_t0;
	
	std::map< typename dealii::DoFHandler<dim>::cell_iterator, double > cell_integrals;
	std::map< typename dealii::DoFHandler<dim>::face_iterator, double > face_integrals;
};

}}}} // namespace

#endif
