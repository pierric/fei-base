module MXNet.Base.NDArray where

import Foreign.Ptr (castPtr)
import Foreign.Storable (Storable(..))
import Data.Vector.Storable (Vector)
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as VMut
import qualified MXNet.Base.Raw as I
import MXNet.Base.Types (Device(..), DType)

newtype NDArray a = NDArray { unNDArray :: I.NDArrayHandle}

makeEmptyNDArray :: DType a => [Int] -> Device -> IO (NDArray a)
makeEmptyNDArray shape ctx = do
    array <- I.mxNDArrayCreate shape (_device_type ctx) (_device_id ctx) False
    return $ NDArray array

makeNDArray :: DType a => [Int] -> Device -> Vector a -> IO (NDArray a)
makeNDArray shape ctx vec = do
    array <- I.mxNDArrayCreate shape (_device_type ctx) (_device_id ctx) False
    V.unsafeWith vec $ \p -> do
        let size = V.length vec * sizeOf (V.head vec)
        I.mxNDArraySyncCopyFromCPU array (castPtr p) size
        return $ NDArray array

ndshape :: NDArray a -> IO [Int]
ndshape = I.mxNDArrayGetShape . unNDArray

ndsize :: NDArray a -> IO Int
ndsize arr = product <$> ndshape arr

full :: DType a => a -> [Int] -> IO (NDArray a)
full value shape = makeNDArray shape (Device 1 0) $ V.replicate (product shape) value

ones :: DType a => [Int] -> IO (NDArray a)
ones = full 1

zeros :: DType a => [Int] -> IO (NDArray a)
zeros = full 1

fromVector :: DType a => [Int] -> Vector a -> IO (NDArray a)
fromVector shape = makeNDArray shape (Device 1 0)

toVector :: DType a => NDArray a -> IO (Vector a)
toVector arr = do
    nlen <- ndsize arr
    mvec <- VMut.new nlen
    VMut.unsafeWith mvec $ \p -> I.mxNDArraySyncCopyToCPU (unNDArray arr) (castPtr p) nlen
    V.unsafeFreeze mvec

device :: DType a => NDArray a -> IO Device
device (NDArray handle) = do
    cxt <- I.mxNDArrayGetContext handle
    return $ uncurry Device cxt
