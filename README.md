# mxnet-hs
## build
+ update stack.yml, pointing to the actuall MXNet dynamic library folder and include folder.
+ stack build

## generate MXNet operators
+ the repository ships pre-generated operators for various version MXNet.
+ You can generate your own with the utility `mxnet-op-gen`: 
  + `LD_LIBRARY_PATH=<path-to-incubator-mxnet-lib> mxnet-op-gen -o <directory-for-ops-code>`

## see also
+ https://github.com/pierric/fei-nn
+ https://github.com/pierric/fei-cocoapi
+ https://github.com/pierric/fei-examples
