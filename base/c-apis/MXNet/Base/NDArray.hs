module MXNet.Base.NDArray where

import Foreign.Ptr (castPtr)
import Data.Vector.Storable (Vector)
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as VMut
import qualified MXNet.Base.Raw as I

ndshape :: I.NDArrayHandle -> IO [Int]
ndshape = I.mxNDArrayGetShape

ndsize :: I.NDArrayHandle -> IO Int
ndsize arr = product <$> ndshape arr

items :: VMut.Storable a => I.NDArrayHandle -> IO (Vector a)
items arr = do
    nlen <- ndsize arr
    mvec <- VMut.new nlen
    VMut.unsafeWith mvec $ \p -> I.mxNDArraySyncCopyToCPU arr (castPtr p) nlen
    V.unsafeFreeze mvec