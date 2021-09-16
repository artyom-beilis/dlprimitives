#!/bin/sh

if [ "$1" == "" ] 
then 
    echo "make-docs.sh path/to/build/dir"
    exit 1
fi
BLDDIR=$(readlink -f "$1")
rm -fr docs/doxygen
doxygen
cd docs/doxygen/html
PYTHONPATH=$BLDDIR/python pydoc3 -w $BLDDIR/python


