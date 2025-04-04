#!/bin/bash
brew install boost
sudo ln -s $(brew --prefix boost)/include/boost /usr/local/include/boost
brew link --overwrite boost


export PYTHONPATH=$(pwd)

export DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH"



export BOOST_ROOT=$(brew --prefix boost)
export BOOST_INCLUDEDIR=$(brew --prefix boost)/include
export BOOST_LIBRARYDIR=$(brew --prefix boost)/lib
CXXFLAGS="-I$(brew --prefix boost)/include" LDFLAGS="-L$(brew --prefix boost)/lib" pip install autodock-vina
echo "PYTHONPATH set to $(pwd)"
echo "DGL_SKIP_GRAPHBOLT is enabled."