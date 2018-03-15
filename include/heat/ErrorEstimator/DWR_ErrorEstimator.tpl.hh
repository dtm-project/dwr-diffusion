/**
 * @file DWR_ErrorEstimator.tpl.hh
 *
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 *
 * @date 2018-03-13, ErrorEstimator class for heat, UK, MPB
 * @date 2017-11-08, ErrorEstimator class, UK, MPB
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

#ifndef __DWR_ErrorEstimator_tpl_hh
#define __DWR_ErrorEstimator_tpl_hh

// PROJECT includes
#include <heat/grid/Grid_DWR.tpl.hh>

// DTM++ includes
#include <DTM++/types/storage_data_vectors.tpl.hh>

// DEAL.II includes
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

// C++ includes
#include <memory>
#include <string>

namespace heat {
namespace dwr {

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
	
	dealii::FEValues<dim> fe_values;
	
	std::vector< dealii::types::global_dof_index > local_dof_indices;
	
	// shape fun scratch:
	std::vector<double>   phi;
	std::vector<dealii::Tensor<1,dim> > grad_phi;
	dealii::Tensor<2,dim> hessian_phi;
	std::vector<double>   laplace_phi;
	
	// local dof scratch:
	std::vector<double> local_u0;
	std::vector<double> local_z0;
	std::vector<double> local_Rz0;
	
	std::vector<double> local_um; // u^-(t_m)
	std::vector<double> local_up; // u^+(t_m)
	
	std::vector<double> local_zm;
	std::vector<double> local_Rzm;
	
	
	// function eval scratch:
	double value_f;
	double value_epsilon;
	dealii::Tensor<1,dim> grad_epsilon;
	
	// other:
	double JxW;
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
	
	std::vector<double>                  neighbor_phi;
	std::vector<dealii::Tensor<1,dim> >  neighbor_grad_phi;
	
	std::vector<double>                  neighbor_local_u0;
	std::vector<double>                  neighbor_local_z0;
	std::vector<double>                  neighbor_local_Rz0;
	
	// data structures of boundary face (-)
// 	std::vector<double> local_uD0; // interpolated boundary values: use neighbor_local_u0
	
	// function eval
	double value_epsilon;
	double value_u_D;
	
	// other
	double JxW;
};


/// Collect 
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
};


} // namespace Scratch
namespace CopyData {


/// Struct for copydata on local cell matrix.
template<int dim>
struct ErrorEstimateOnCell{
	ErrorEstimateOnCell();
	ErrorEstimateOnCell(const ErrorEstimateOnCell &copydata);

	typename dealii::DoFHandler<dim>::cell_iterator cell;
	double value;
};


/// Struct for copydata on local cell matrix.
template<int dim>
struct ErrorEstimateOnFace{
	ErrorEstimateOnFace();
	ErrorEstimateOnFace(const ErrorEstimateOnFace &copydata);

	typename dealii::DoFHandler<dim>::face_iterator face;
	double value;
};


template<int dim>
struct ErrorEstimates {
	ErrorEstimates();
	ErrorEstimates(const ErrorEstimates &copydata);
	
	ErrorEstimateOnCell<dim> cell;
	ErrorEstimateOnFace<dim> face;
};

} // namespace CopyData
} // namespace Assembly
////////////////////////////////////////////////////////////////////////////////



template<int dim>
class ErrorEstimator {
public:
	ErrorEstimator() = default;
	virtual ~ErrorEstimator() = default;
	
	virtual void estimate(
		std::shared_ptr< dealii::Function<dim> > epsilon,
		std::shared_ptr< dealii::Function<dim> > f,
		std::shared_ptr< dealii::Function<dim> > u_D,
		std::shared_ptr< dealii::Function<dim> > u_0,
		std::shared_ptr< heat::Grid_DWR<dim,1> > grid,
		std::shared_ptr< DTM::types::storage_data_vectors<1> > u,
		std::shared_ptr< DTM::types::storage_data_vectors<2> > z,
		std::shared_ptr< DTM::types::storage_data_vectors<1> > eta
	);
	
protected:
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
	
	virtual void assemble_error_on_boundary_face(
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
	
	std::shared_ptr< heat::Grid_DWR<dim,1> > grid;
	
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
		std::shared_ptr< dealii::Function<dim> > epsilon;
		std::shared_ptr< dealii::Function<dim> > f;
		std::shared_ptr< dealii::Function<dim> > u_D;
		std::shared_ptr< dealii::Function<dim> > u_0;
// 		std::shared_ptr< dealii::Function<dim> > density;
	} function;
	
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
	
	dealii::Vector<double> dual_weights;
	dealii::Vector<double> g_interpolated;
};

}} // namespace

#endif