# mxnet-hs
## build
+ LD_LIBRARY_PATH=<path-to-incubator-mxnet-lib> stack build fei-base

## generate MXNet operators
+ the repository ships pre-generated operators for various version MXNet.
+ You can generate your own with the utility `mxnet-op-gen`: 
  + `LD_LIBRARY_PATH=<path-to-incubator-mxnet-lib> mxnet-op-gen -o <directory-for-ops-code>`

## see also
+ https://github.com/pierric/fei-nn
+ https://github.com/pierric/fei-dataiter
+ https://github.com/pierric/fei-cocoapi
+ https://github.com/pierric/fei-examples
