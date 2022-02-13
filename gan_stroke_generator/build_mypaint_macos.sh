#!/bin/sh
brew install libmypaint pygobject3 gtk+3 lcms2 swig
wget https://github.com/mypaint/mypaint-brushes/releases/download/v1.3.1/mypaint-brushes-1.3.1.tar.xz
tar -xvf mypaint-brushes-1.3.1.tar.xz && mv mypaint-brushes-1.3.1 mypaint-brushes
rm -rf mypaint-brushes-1.3.1.tar.xz
cd mypaint-brushes && ./configure && mv mypaint-brushes-1.0.pc mypaint-brushes-2.0.pc && cd ..
wget https://github.com/mypaint/mypaint/releases/download/v2.0.1/mypaint-2.0.1.tar.xz
tar -xvf mypaint-2.0.1.tar.xz && mv mypaint-2.0.1 mypaint
rm -rf mypaint-2.0.1.tar.xz
cd mypaint
export PKG_CONFIG_PATH=../mypaint-brushes
export CFLAGS="-g -D_DARWIN_C_SOURCE"
python setup.py build
cp build/lib.macosx-*/lib/_mypaintlib.*.so lib/_mypaintlib.so
sudo cp -a /usr/local/Cellar/pygobject3/*/lib/python*/site-packages/ $VIRTUAL_ENV/lib/python*/site-packages/
mv lib ../lib
cd ..
mv mypaint-brushes/brushes brushes
rm -rf mypaint mypaint-brushes
touch cairo.py