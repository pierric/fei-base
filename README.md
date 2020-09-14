# mxnet-hs
## build
+ Update stack.yml, pointing to the actuall MXNet dynamic library folder and include folder.
+ Build utility **mxnet-op-gen**: `stack build fei-base:mxnet-op-gen`
+ Generate tensor operations: `LD_LIBRARY_PATH=<path-to-mxnet-lib> stack exec mxnet-op-gen -- -o ops/1.6`
+ Rebuild library with tensor operations: `stack build --flag fei-base:-mxnet_1_6`

## see also
+ https://github.com/pierric/fei-nn
+ https://github.com/pierric/fei-cocoapi
+ https://github.com/pierric/fei-datasets
+ https://github.com/pierric/fei-modelzoo
+ https://github.com/pierric/fei-examples
