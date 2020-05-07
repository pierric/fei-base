{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.Base.NDArray where

import RIO hiding (Vector)
import qualified RIO.NonEmpty as RNE
import RIO.Vector.Storable (Vector)
import qualified RIO.Vector.Storable as SV
import qualified RIO.Vector.Storable.Unsafe as SV
import qualified RIO.Vector.Unboxed as UV
import qualified Data.Vector.Storable.Mutable as VMut

import qualified Data.Array.Repa as Repa
import qualified Data.Array.Repa.Eval as Repa
import Foreign.Ptr (castPtr)
import Foreign.Storable (Storable(..))
import GHC.Generics (Generic, Generic1)
import qualified Data.Store as S
import System.IO.Unsafe

import qualified MXNet.Base.Raw as I
import MXNet.Base.Types (Context(..), contextCPU, DType(..), ForeignData(..))

newtype NDArray a = NDArray { unNDArray :: I.NDArrayHandle}
    deriving (Generic, Generic1, Show)
instance ForeignData (NDArray a) where
    touch = I.touchNDArrayHandle . unNDArray

instance NFData (NDArray a)

instance (DType a, S.Store a) => S.Store (NDArray a) where
    size = S.VarSize $ \a ->
                -- not ideal, I should avoid call toVector
                let b = unsafePerformIO $ do
                            shape <- ndshape a
                            vec <- toVector a
                            return (RNE.toList shape, vec)
                    S.VarSize sz = S.size :: S.Size ([Int], SV.Vector a)
                in sz b
    peek = do
        (shape, vec) <- S.peek
        liftIO $ fromVector shape vec
    poke a = do
        shape <- liftIO $ ndshape a
        vec <- liftIO $ toVector a
        S.poke (shape, vec)

makeEmptyNDArray :: forall a. DType a => NonEmpty Int -> Context -> IO (NDArray a)
makeEmptyNDArray shape ctx = do
    array <- I.mxNDArrayCreateEx (RNE.toList shape)
                                 (_device_type ctx)
                                 (_device_id ctx)
                                 False
                                 (flag (undefined :: a))
    return $ NDArray array

makeNDArray :: forall a. DType a => NonEmpty Int -> Context -> Vector a -> IO (NDArray a)
makeNDArray shape ctx vec = do
    array <- I.mxNDArrayCreateEx (RNE.toList shape)
                                 (_device_type ctx)
                                 (_device_id ctx)
                                 False
                                 (flag (undefined :: a))
    SV.unsafeWith vec $ \p -> do
        I.mxNDArraySyncCopyFromCPU array (castPtr p) (SV.length vec)
        return $ NDArray array

makeNDArrayLike :: DType a => NDArray a -> Context -> IO (NDArray a)
makeNDArrayLike src cxt = do
    shape <- ndshape src
    makeEmptyNDArray shape cxt

ndshape :: DType a => NDArray a -> IO (NonEmpty Int)
ndshape arr = do
    shape <- I.mxNDArrayGetShape $ unNDArray arr
    case RNE.nonEmpty shape of
        Just s -> return s
        Nothing -> error "impossible shape"

ndsize :: DType a => NDArray a -> IO Int
ndsize arr = product <$> ndshape arr

full :: DType a => a -> NonEmpty Int -> IO (NDArray a)
full value shape = makeNDArray shape contextCPU $ SV.replicate (product shape) value

ones :: DType a => NonEmpty Int -> IO (NDArray a)
ones = full 1

zeros :: DType a => NonEmpty Int -> IO (NDArray a)
zeros = full 1

fromVector :: DType a => NonEmpty Int -> Vector a -> IO (NDArray a)
fromVector shape = makeNDArray shape contextCPU

copyFromVector :: DType a => NDArray a -> Vector a -> IO ()
copyFromVector arr vec = do
    sz <- ndsize arr
    if (sz /= SV.length vec)
      then error ""
      else do
        SV.unsafeWith vec $ \p -> do
            I.mxNDArraySyncCopyFromCPU (unNDArray arr) (castPtr p) sz

copyFromRepa :: (Repa.Shape sh, DType a, UV.Unbox a) => NDArray a -> Repa.Array Repa.U sh a -> IO ()
copyFromRepa arr repa = do
    let vec = UV.convert $ Repa.toUnboxed repa
    copyFromVector arr vec

toVector :: DType a => NDArray a -> IO (Vector a)
toVector arr = do
    nlen <- ndsize arr
    mvec <- VMut.new nlen
    VMut.unsafeWith mvec $ \p -> I.mxNDArraySyncCopyToCPU (unNDArray arr) (castPtr p) nlen
    SV.unsafeFreeze mvec

toRepa :: (Repa.Shape sh, DType a, UV.Unbox a) => NDArray a -> IO (Repa.Array Repa.U sh a)
toRepa arr = do
    shp <- RNE.toList <$> ndshape arr
    vec <- toVector arr
    return $ Repa.fromUnboxed (Repa.shapeOfList (reverse shp)) (UV.convert vec)

context :: DType a => NDArray a -> IO Context
context (NDArray handle) = do
    cxt <- I.mxNDArrayGetContext handle
    return $ uncurry Context cxt

waitToRead :: DType a => NDArray a -> IO ()
waitToRead (NDArray hdl) = I.mxNDArrayWaitToRead hdl

waitToWrite :: DType a => NDArray a -> IO ()
waitToWrite (NDArray hdl) = I.mxNDArrayWaitToWrite hdl

waitAll = I.mxNDArrayWaitAll

-- All non-update OP return a list of NDArray, but usually
-- it is a single element list. `sing` check and return
-- the first NDArray
sing :: (args -> IO [I.NDArrayHandle]) -> args -> IO I.NDArrayHandle
sing op args = do
    res <- op args
    case res of
      [x] -> return x
      _ -> error "OP result is not a single NDArray."
