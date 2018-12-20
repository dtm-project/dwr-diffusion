# README #

This README documents whatever steps are necessary to get the application
  DTM++.Project/DWR/DWR-Heat
up and running.

### What is this repository for? ###

* Heat equation problem with DWR

### How do I get set up? ###

* Dependencies
deal.II v9.0.0 at least, installed via candi, cf. https://github.com/dealii/candi

* Configuration
cmake .
make

* Run (single process)
./dwr-heat


### Who do I talk to? ###

* Principial Author
    * Dr.-Ing. Dipl.-Ing. Uwe Koecher (koecher@hsu-hamburg.de, dtmproject@uwe.koecher.cc)
* Contributors
    * Marius P. Bruchhaeuser (bruchhaeuser@hsu-hamburg.de)

Remark. DTM++ is free software.

If you write scientific publication using results obtained by reusing parts
of DTM++, or specifically DTM++/dwr-diffusion, especially by reusing the
datastructures, algorithms and/or supporting parameter/data input/output
classes, you are willing to cite the following two publications:

- U. Koecher, M.P. Bruchhaeuser, M. Bause: "Efficient and scalable
  data structures and algorithms for goal-oriented adaptivity of space-time
  FEM codes", submitted to SoftwareX, p. 1-6, 2018.

- U. Koecher: "Variational space-time methods for the elastic wave equation
  and the diffusion equation", Ph.D. thesis,
  Department of Mechanical Engineering of the Helmut-Schmidt-University,
  University of the German Federal Armed Forces Hamburg, Germany, p. 1-188,
  urn:nbn:de:gbv:705-opus-31129, 2015. Open access via:
  http://edoc.sub.uni-hamburg.de/hsu/volltexte/2015/3112/


### License ###
Copyright (C) 2012-2018 by Uwe Koecher and contributors

This file is part of DTM++.

DTM++ is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version.

DTM++ is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.
 
You should have received a copy of the GNU Lesser General Public License
along with DTM++. If not, see <http://www.gnu.org/licenses/>.
Please see the file
	./LICENSE
for details.
