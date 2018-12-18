/**
 * @file error_functional.hh
 *
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 *
 * @date 2018-02-08, UK
 */

/*  Copyright (C) 2012-2018 by Uwe Koecher and contributors                   */
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

#ifndef __error_functional_hh
#define __error_functional_hh

namespace diffusion {
namespace types {

enum class error_functional : unsigned int {
	forbidden     = 0,
	
	/**
	 * \f$ J(u) \f$ is a point value \f$ x \in \Omega \text{ for } t=T \f$.
	 */
	point_final_T = 1,
	
	/**
	 * \f$ J(u) = \frac{1}{| \Omega |_d} \int_\Omega u(x,T) \operatorname{d} x \f$.
	 */
	mean_final_T  = 2,
	
	/**
	 * \f$ J(u) = \frac{1}{| \Omega \times I |_{d+1}}
	 * \int_I \int_\Omega u(x,t) \operatorname{d} x \operatorname{d} t \f$.
	 */
	mean_global   = 3,
	
	/**
	 * \f$ J(\varphi) = \displaystyle
	 * \frac{(\varphi(T), \hat e)_{\Omega}}{\| \hat e \|_{L^2(\Omega)}} \f$
	 * with \f$ \hat e \f$ being a sufficiently good approximation of
	 * \f$ e(T) = (u - u_{\tau,h})(T) \f$.
	 */
	L2_final_T    = 4,
	
	/**
	 * \f$ J(\varphi) = \displaystyle
	 * \frac{(\varphi, \hat e)_{\Omega \times I}}
	 * {\| \hat e \|_{L^2(I;L^2(\Omega))}} \f$
	 * with \f$ \hat e \f$ being a sufficiently good approximation of
	 * \f$ e = (u - u_{\tau,h}) \f$.
	 */
	L2_L2_global  = 5,
	
	/**
	 * \f$ J(\varphi) = \displaystyle
	 * \frac{(\varphi, \hat e)_{\Omega_c \times I_c}}
	 * {\| \hat e \|_{L^2(I_c; L^2(\Omega_c))}} \f$
	 * with \f$ \hat e \f$ being a sufficiently good approximation of
	 * \f$ e = (u - u_{\tau,h}) \f$
	 * and a control volume \f$ \Omega_c \times I_c \subset \Omega \times I \f$.
	 */
	L2_L2_local   = 6
};

}}

#endif
