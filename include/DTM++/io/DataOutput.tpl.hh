/**
 * @file DataOutput.tpl.hh
 * @author Uwe Koecher (UK)
 * @date 2016-08-11, DWR, UK
 * @date 2016-01-12, academic/heat, UK
 * @date 2015-12-04, biot, UK
 * @date 2015-05-22, DTM++.core for DTM++/awave, UK
 * @date 2015-02-25, DTM++.core for DTM++/ccfd, UK
 * @date 2014-06-19, DTM++.core for DTM++ 1.0 MEAT, UK
 * @date 2014-04-29, (2012-08-08), UK
 *
 * @brief This is a template to output a VECTOR as hdf5/xdmf.
 */

/*  Copyright (C) 2012-2016 by Uwe Koecher                                    */
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
/*  along with DTM++.  If not, see <http://www.gnu.org/licenses/>.            */

#ifndef __DataOutput_tpl_hh
#define __DataOutput_tpl_hh

// DEAL.II includes
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>

#include <deal.II/lac/vector.h>

// C++ includes
#include <memory>
#include <string>
#include <vector>

namespace DTM {

using VectorType = dealii::Vector<double>;

template<int dim>
class DataOutput {
public:
	DataOutput();
	
	constexpr DataOutput(DataOutput &copy) :
		mpi_comm(copy.mpi_comm),
		format(copy.format),
		dof(copy.dof),
		data_field_names(copy.data_field_names),
		data_field_names_process_id(copy.data_field_names_process_id),
		dci_field(copy.dci_field),
		data_file_counter(copy.data_file_counter),
		setw_value(copy.setw_value),
		process_id(copy.process_id),
		data_output_patches(copy.data_output_patches),
		xdmf_entries_data(copy.xdmf_entries_data)
	{};
	
	virtual ~DataOutput() = default;
	
	enum class DataFormat {
		invalid,
		HDF5_XDMF
	};
	
	virtual void set_MPI_Comm(MPI_Comm mpi_comm = MPI_COMM_WORLD);
	
	virtual void set_DoF_data(
		std::shared_ptr< dealii::DoFHandler<dim> > dof
	);
	
	virtual void set_output_format(DataFormat format);
	
	virtual void set_data_output_patches(unsigned int data_output_patches = 1);
	
	/// Set output number width.
	virtual void set_setw(const unsigned int setw_value = 4);
	
	virtual void set_data_field_names(
		std::vector<std::string> &data_field_names
	);
	
	virtual void set_data_component_interpretation_field(
		std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> &dci_field
	);
	
	virtual void write_data(
		const std::string &solution_file_name,
		std::shared_ptr< VectorType > solution_vector,
		const double &time
	);
	
	virtual void write_data(
		const std::string &solution_file_name,
		std::vector< std::shared_ptr< VectorType > > &solution_vectors,
		const double &time
	);
	
	virtual void write_data(
		const std::string &solution_file_name,
		std::shared_ptr< VectorType > solution_vector,
		std::shared_ptr< dealii::DataPostprocessor<dim> > data_postprocessor,
		const double &time
	);
	
	virtual void increment_data_file_counter(const unsigned int value = 1) {
		data_file_counter += value;
	}
	
protected:
	MPI_Comm mpi_comm;
	DataFormat format;
	std::shared_ptr< dealii::DoFHandler<dim> > dof;
	
	std::vector<std::string> data_field_names;
	std::vector<std::string> data_field_names_process_id;
	std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> dci_field;
	
	unsigned int data_file_counter;
	unsigned int setw_value;
	unsigned int process_id;
	unsigned int data_output_patches;
	
	std::vector<dealii::XDMFEntry> xdmf_entries_data;
};

} // namespaces

#endif
