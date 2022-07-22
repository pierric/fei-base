# fei-base
## build
+ `ln -s <path-to-mxnet-lib> mxnet`
+ Build utility **mxnet-op-gen**: `stack build  --flag fei-base:-mxnet_geq_10900 fei-base:mxnet-op-gen`
+ Generate tensor operations: `LD_LIBRARY_PATH=<path-to-mxnet-lib> stack exec mxnet-op-gen -- -o ops/1.9`
+ Rebuild library with tensor operations: `stack build --flag fei-base:mxnet_geq_10900`

## see also
+ https://github.com/pierric/fei-nn
+ https://github.com/pierric/fei-cocoapi
+ https://github.com/pierric/fei-datasets
+ https://github.com/pierric/fei-modelzoo
+ https://github.com/pierric/fei-examples
