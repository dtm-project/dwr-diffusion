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

if git rev-parse --git-dir > /dev/null 2>&1; then
    # This is a valid git repository
    find . -type f -name .kdev_include_paths | xargs -n1 git update-index --assume-unchanged

    if [ ! $? = 0 ]; then
        return 0
    fi

else
    # this is not a git repository
    echo "skipping (git command) since this is not a git repository"
fi
