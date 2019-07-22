# mxnet-hs
## build
+ LD_LIBRARY_PATH=<path-to-incubator-mxnet-lib> stack build mxnet-base
+ LD_LIBRARY_PATH=<path-to-incubator-mxnet-lib> stack exec  mxnet-op-gen
+ LD_LIBRARY_PATH=<path-to-incubator-mxnet-lib> stack build mxnet-ops

## see also
+ https://github.com/pierric/mxnet-nn
+ https://github.com/pierric/mxnet-dataiter
+ https://github.com/pierric/mxnet-cocoapi
+ https://github.com/pierric/mxnet-examples