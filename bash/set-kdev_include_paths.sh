#!/bin/bash

# Copyright (C) 2012-2015 by Uwe Koecher
# 
# This file is part of DTM++.
# 
# DTM++ is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# 
# DTM++ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with DTM++.  If not, see <http://www.gnu.org/licenses/>.


INCLUDE_PATH_FILE=.include_pathes
rm -f $INCLUDE_PATH_FILE

echo "$MPI_HOME/include" >> $INCLUDE_PATH_FILE

echo "$DEAL_II_DIR/include" >> $INCLUDE_PATH_FILE

DTM_PATH=`pwd`
echo "$DTM_PATH/include" >> $INCLUDE_PATH_FILE

find . -type f -name .kdev_include_paths | xargs -i cp -f .include_pathes {}
rm -f $INCLUDE_PATH_FILE
