{-# LANGUAGE CPP                 #-}
{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.Base.NDArray where

#ifdef USE_REPA
import qualified Data.Array.Repa              as Repa
#endif

import qualified Data.Store                   as S
import qualified Data.Vector.Storable.Mutable as VMut
import           Foreign.Ptr                  (castPtr)
import           GHC.Generics                 (Generic, Generic1)
import           RIO                          hiding (Vector)
import           RIO.Vector.Storable          (Vector)
import qualified RIO.Vector.Storable          as SV
import qualified RIO.Vector.Storable.Unsafe   as SV
import qualified RIO.Vector.Unboxed           as UV
import           System.IO.Unsafe
import           Text.Printf

import           MXNet.Base.Operators.Tensor  (__copyto)
import qualified MXNet.Base.Raw               as I
import           MXNet.Base.Spec.HMap         (HMap (..), (.&))
import           MXNet.Base.Spec.Operator     (ArgOf (..))
import           MXNet.Base.Types             (Context (..), DType (..),
                                               ForeignData (..), contextCPU)

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
                            return (shape, vec)
                    S.VarSize sz = S.size :: S.Size ([Int], SV.Vector a)
                in sz b
    peek = do
        (shape, vec) <- S.peek
        liftIO $ fromVector shape vec
    poke a = do
        shape <- liftIO $ ndshape a
        vec <- liftIO $ toVector a
        S.poke (shape, vec)

makeEmptyNDArray :: forall a. (HasCallStack, DType a) => [Int] -> Context -> IO (NDArray a)
makeEmptyNDArray shape ctx = do
    array <- I.mxNDArrayCreateEx shape
                                 (_device_type ctx)
                                 (_device_id ctx)
                                 False
                                 (flag (undefined :: a))
    return $ NDArray array

makeNDArray :: forall a. (HasCallStack, DType a) => [Int] -> Context -> Vector a -> IO (NDArray a)
makeNDArray shape ctx vec = do
    array <- I.mxNDArrayCreateEx shape
                                 (_device_type ctx)
                                 (_device_id ctx)
                                 False
                                 (flag (undefined :: a))
    SV.unsafeWith vec $ \p -> do
        I.mxNDArraySyncCopyFromCPU array (castPtr p) (SV.length vec)
        return $ NDArray array

makeNDArrayLike :: (HasCallStack, DType a) => NDArray a -> Context -> IO (NDArray a)
makeNDArrayLike src cxt = do
    shape <- ndshape src
    makeEmptyNDArray shape cxt

ndshape :: (HasCallStack, DType a) => NDArray a -> IO [Int]
ndshape arr = I.mxNDArrayGetShape $ unNDArray arr

ndsize :: (HasCallStack, DType a) => NDArray a -> IO Int
ndsize arr = product <$> ndshape arr

full :: (HasCallStack, DType a) => a -> [Int] -> IO (NDArray a)
full value shape = makeNDArray shape contextCPU $ SV.replicate (product shape) value

ones :: (HasCallStack, DType a) => [Int] -> IO (NDArray a)
ones = full 1

zeros :: (HasCallStack, DType a) => [Int] -> IO (NDArray a)
zeros = full 1

fromVector :: (HasCallStack, DType a) => [Int] -> Vector a -> IO (NDArray a)
fromVector shape = makeNDArray shape contextCPU

copyFromVector :: (HasCallStack, DType a) => NDArray a -> Vector a -> IO ()
copyFromVector arr vec = do
    sz <- ndsize arr
    if (sz /= SV.length vec)
      then error $ printf "cannot copy from Vector: size mismatch (%d vs. %d)" sz (SV.length vec)
      else do
        SV.unsafeWith vec $ \p -> do
            I.mxNDArraySyncCopyFromCPU (unNDArray arr) (castPtr p) sz

toVector :: DType a => NDArray a -> IO (Vector a)
toVector arr = do
    nlen <- ndsize arr
    mvec <- VMut.new nlen
    VMut.unsafeWith mvec $ \p -> I.mxNDArraySyncCopyToCPU (unNDArray arr) (castPtr p) nlen
    SV.unsafeFreeze mvec

#ifdef USE_REPA
fromRepa :: (HasCallStack, Repa.Shape sh, DType a, UV.Unbox a) => Repa.Array Repa.U sh a -> IO (NDArray a)
fromRepa arr = do
    let shp = reverse $ Repa.listOfShape $ Repa.extent arr
        vec = UV.convert $ Repa.toUnboxed arr
    makeNDArray shp contextCPU vec


copyFromRepa :: (HasCallStack, Repa.Shape sh, DType a, UV.Unbox a) => NDArray a -> Repa.Array Repa.U sh a -> IO ()
copyFromRepa arr repa = do
    let vec = UV.convert $ Repa.toUnboxed repa
    copyFromVector arr vec

toRepa :: (Repa.Shape sh, DType a, UV.Unbox a) => NDArray a -> IO (Repa.Array Repa.U sh a)
toRepa arr = do
    shp <- ndshape arr
    vec <- toVector arr
    return $ Repa.fromUnboxed (Repa.shapeOfList (reverse shp)) (UV.convert vec)
#endif

context :: (HasCallStack, DType a) => NDArray a -> IO Context
context (NDArray handle) = do
    cxt <- I.mxNDArrayGetContext handle
    return $ uncurry Context cxt

toContext :: forall a. (HasCallStack, DType a) => NDArray a -> Context -> IO (NDArray a)
toContext arr cxt = do
    ncxt <- context arr
    if cxt == ncxt
    then return arr
    else do
        narr <- makeNDArrayLike arr cxt
        void $ (__copyto (#data :≅ arr .& Nil) (Just [narr]) :: IO [NDArray a])
        return narr

toCPU :: (HasCallStack, DType a) => NDArray a -> IO (NDArray a)
toCPU = flip toContext contextCPU

waitToRead :: (HasCallStack, DType a) => NDArray a -> IO ()
waitToRead (NDArray hdl) = I.mxNDArrayWaitToRead hdl

waitToWrite :: (HasCallStack, DType a) => NDArray a -> IO ()
waitToWrite (NDArray hdl) = I.mxNDArrayWaitToWrite hdl

waitAll :: HasCallStack => IO ()
waitAll = I.mxNDArrayWaitAll

emptyCache :: HasCallStack => Context -> IO ()
emptyCache context = I.mxStorageEmptyCache (_device_type context) (_device_id context)
