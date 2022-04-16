{-# LANGUAGE CPP                 #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
{-# OPTIONS_GHC -fplugin=Data.Record.Anon.Plugin #-}
module MXNet.Base.NDArray where

#ifdef USE_REPA
import qualified Data.Array.Repa              as Repa
#endif

import qualified Data.Record.Anon.Simple      as Anon
import qualified Data.Store                   as S
import qualified Data.Vector.Storable.Mutable as VMut
import           Foreign.Ptr                  (castPtr)
import           GHC.Generics                 (Generic, Generic1)
import           RIO                          hiding (Vector)
import           RIO.Vector.Storable          (Vector)
import qualified RIO.Vector.Storable          as SV
import qualified RIO.Vector.Storable.Partial  as SV
import qualified RIO.Vector.Storable.Unsafe   as SV
import qualified RIO.Vector.Unboxed           as UV
import           System.IO.Unsafe
import           Text.Printf

import           MXNet.Base.Operators.Tensor  (__copyto)
import qualified MXNet.Base.Raw               as I
import           MXNet.Base.Types             (Context (..), DType (..),
                                               ForeignData (..), NumericDType,
                                               contextCPU)

newtype NDArray a = NDArray { unNDArray :: I.NDArrayHandle}
    deriving (Generic, Generic1, Show)
instance ForeignData (NDArray a) where
    touch = I.touchNDArrayHandle . unNDArray

instance NFData (NDArray a)

instance (DType a, S.Store a) => S.Store (NDArray a) where
    size = S.VarSize $ \a ->
                let shape = unsafePerformIO $ ndshape a
                    S.ConstSize size_int = S.size :: S.Size Int
                    S.VarSize size_shape = S.size :: S.Size [Int]
                 in case S.size :: S.Size a of
                      S.VarSize _ -> error "Unsupported! element has no constant size."
                      S.ConstSize sizeElm -> size_shape shape + size_int + sizeElm * product shape
    peek = do
        shape <- S.peek :: S.Peek [Int]
        vec   <- S.peek :: S.Peek (SV.Vector a)
        if product shape /= SV.length vec
            then error "mismatched shape and data"
            else liftIO $ fromVector shape vec
    poke a = do
        shape <- liftIO $ ndshape a
        vec   <- liftIO $ toVector a
        S.poke shape
        S.poke vec

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

ndOnes :: (HasCallStack, NumericDType a) => [Int] -> IO (NDArray a)
ndOnes = full 1

ndZeros :: (HasCallStack, NumericDType a) => [Int] -> IO (NDArray a)
ndZeros = full 0

fromVector :: (HasCallStack, DType a) => [Int] -> Vector a -> IO (NDArray a)
fromVector shape = makeNDArray shape contextCPU

copyFromVector :: (HasCallStack, DType a) => NDArray a -> Vector a -> IO ()
copyFromVector arr vec = do
    sz <- ndsize arr
    if sz /= SV.length vec
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

toValue :: DType a => NDArray a -> IO a
toValue arr = do
    vec <- toVector arr
    if SV.length vec /= 1
        then error $ printf "Cannot convert to value for a non-scalar ndarray."
        else return $ SV.head vec

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
        void $ __copyto @NDArray @a ANON{_data = arr} (Just [narr])
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

getGradient :: HasCallStack => NDArray a -> IO (Maybe (NDArray a))
getGradient (NDArray hdl) = do
    grad <- I.mxNDArrayGetGrad hdl
    if I.isNullNDArrayHandle grad then return Nothing else return $ Just $ NDArray grad

detach :: HasCallStack => NDArray a -> IO (NDArray a)
detach (NDArray hdl) = NDArray <$> I.mxNDArrayDetach hdl
