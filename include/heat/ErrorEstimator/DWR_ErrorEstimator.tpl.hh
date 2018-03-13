/**
 * @file DWR_ErrorEstimator.tpl.hh
 * @author Uwe Koecher (UK), Marius Paul Bruchhaeuser (MPB)
 * @date 2017-11-08, ErrorEstimator class, UK, MPB
 */

/*  Copyright (C) 2012-2017 by Uwe Koecher, Marius Paul Bruchhaeuser          */  
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
#include <Poisson/Grid/Grid_DWR.tpl.hh>

// DTM++ includes

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

namespace Poisson {
namespace DWR {

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
	
	std::vector<double> rhs_values;
	std::vector<double> cell_laplacians;
	std::vector<double> cell_values;
	std::vector<double> dual_weights;
	std::vector<double> R_u_h;
	
	std::vector< dealii::Tensor<1,dim> > cell_gradients;
	std::vector< dealii::Tensor<1,dim> > dual_weights_gradients;
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
	
	dealii::FEFaceValues<dim> fe_face_values;
	dealii::FEFaceValues<dim> fe_face_values_neighbor;
	dealii::FESubfaceValues<dim> fe_subface_values;
	
	std::vector<double> boundary_values;
	std::vector<double> g_h;
	std::vector<double> jump_residuals;
	std::vector<double> dual_weights;
	std::vector<double> inhom_dirichlet_difference;
	
	std::vector< dealii::Tensor<1,dim> > dual_solution_gradients;
	std::vector< dealii::Tensor<1,dim> > cell_grads;
	std::vector< dealii::Tensor<1,dim> > neighbor_grads;
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
	
	virtual void set_objects(
		std::shared_ptr< Poisson::Grid_DWR<dim,1> > grid,
		std::shared_ptr< dealii::Function<dim> > epsilon,
		std::shared_ptr< dealii::Function<dim> > BoundaryValues,
		std::shared_ptr< dealii::Function<dim> > f
	);
	
	virtual void estimate(
		std::shared_ptr< dealii::Vector<double> > u, ///< primal problem solution on dual space
		std::shared_ptr< dealii::Vector<double> > z, ///< dual problem solution
		std::shared_ptr< dealii::Vector<double> > error_indicators
	);

protected:
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
	
	struct {
		std::shared_ptr< dealii::Vector<double> > u; ///< primal problem solution on dual space
		std::shared_ptr< dealii::Vector<double> > z; ///< dual problem solution
	} dual;
	
	std::shared_ptr< Poisson::Grid_DWR<dim,1> > grid; ///< grid class
	
	std::shared_ptr< dealii::Function<dim> > BoundaryValues; ///< exact solution/boundary values
	
	std::shared_ptr< dealii::Function<dim> > epsilon; ///< diffusion coefficient
	
	struct {
		std::shared_ptr< dealii::Function<dim> > f; ///< Force function.
	} function;
	
	std::map< typename dealii::DoFHandler<dim>::cell_iterator, double > cell_integrals;
	std::map< typename dealii::DoFHandler<dim>::face_iterator, double > face_integrals;
	
	dealii::Vector<double> primal_z; //TEST
	dealii::Vector<double> primal_z_dual;
	
	dealii::Vector<double> dual_weights;
	dealii::Vector<double> g_interpolated;
};

}} // namespace

#endif
