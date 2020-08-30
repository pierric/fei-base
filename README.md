# mxnet-hs
## build
+ update stack.yml, pointing to the actuall MXNet dynamic library folder and include folder.
+ build library without tensor operations
    `stack build --flag fei-base:-MXNET_1_6`
+ generate tensor operations
    `LD_LIBRARY_PATH=<path-to-mxnet-lib> stack exec mxnet-op-gen -- -o ops/1.6`
+ rebuild library with tensor operations
    `stack build`

## see also
+ https://github.com/pierric/fei-nn
+ https://github.com/pierric/fei-cocoapi
+ https://github.com/pierric/fei-datasets
+ https://github.com/pierric/fei-modelzoo
+ https://github.com/pierric/fei-examples
