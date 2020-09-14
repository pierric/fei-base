module MXNet.Base.NDArray where

import qualified MXNet.Base.Raw               as I

newtype NDArray a = NDArray { unNDArray :: I.NDArrayHandle}

