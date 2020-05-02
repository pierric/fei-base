{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.Base.Raw.NDArray where

import RIO
import RIO.List (unzip)
import Foreign.Marshal (alloca, withArray, peekArray)
import Foreign.Storable (Storable(..))
import Foreign.Concurrent (newForeignPtr)
import Foreign.ForeignPtr (touchForeignPtr)
import Foreign.ForeignPtr.Unsafe (unsafeForeignPtrToPtr)
import Foreign.C.Types
import Foreign.C.String (CString)
import Foreign.Ptr
import C2HS.C.Extra.Marshal (withIntegralArray)
import GHC.Generics (Generic)
import Control.DeepSeq (NFData(..), rwhnf)
import Control.Monad ((>=>))

{# import MXNet.Base.Raw.Common #}

#include <mxnet/c_api.h>

{# typedef size_t CSize#}
{# default in `CSize' [size_t] fromIntegral #}
{# typedef mx_uint MX_UINT#}
{# default in `MX_UINT' [mx_uint] id #}

-- NDArray
{#
pointer NDArrayHandle foreign newtype
#}

deriving instance Generic NDArrayHandle
deriving instance Show NDArrayHandle
instance NFData NDArrayHandle where
    rnf = rwhnf

type NDArrayHandlePtr = Ptr NDArrayHandle

touchNDArrayHandle :: NDArrayHandle -> IO ()
touchNDArrayHandle (NDArrayHandle fptr) = touchForeignPtr fptr

newNDArrayHandle :: NDArrayHandlePtr -> IO NDArrayHandle
newNDArrayHandle ptr = do
    hdl <- newForeignPtr ptr (mxNDArrayFree ptr)
    return $ NDArrayHandle hdl

peekNDArrayHandle :: Ptr NDArrayHandlePtr -> IO NDArrayHandle
peekNDArrayHandle = peek >=> newNDArrayHandle

withNDArrayHandleArray :: [NDArrayHandle] -> (Ptr NDArrayHandlePtr -> IO r) -> IO r
withNDArrayHandleArray array io = do
    let unNDArrayHandle (NDArrayHandle fptr) = fptr
    r <- withArray (map (unsafeForeignPtrToPtr . unNDArrayHandle) array) io
    mapM_ (touchForeignPtr . unNDArrayHandle) array
    return r

{#
fun MXNDArrayFree as mxNDArrayFree_
    {
        id `NDArrayHandlePtr'
    } -> `CInt'
#}

mxNDArrayFree :: NDArrayHandlePtr -> IO ()
mxNDArrayFree = checked . mxNDArrayFree_

{#
fun MXNDArrayCreateNone as mxNDArrayCreateNone_
    {
        alloca- `NDArrayHandle' peekNDArrayHandle*
    } -> `CInt'
#}

mxNDArrayCreateNone :: IO NDArrayHandle
mxNDArrayCreateNone = checked mxNDArrayCreateNone_

{#
fun MXNDArrayCreate as mxNDArrayCreate_
    {
        withArray* `[MX_UINT]',
        id `MX_UINT',
        `CInt',
        `CInt',
        `CInt',
        alloca- `NDArrayHandle' peekNDArrayHandle*
    } -> `CInt'
#}

mxNDArrayCreate :: [Int] -> Int -> Int -> Bool -> IO NDArrayHandle
mxNDArrayCreate shape devtype devid delay_alloc = do
    let shape_   = fromIntegral <$> shape
        dim_     = fromIntegral (length shape_)
        devtype_ = fromIntegral devtype
        devid_   = fromIntegral devid
    checked $ mxNDArrayCreate_ shape_ dim_ devtype_ devid_ (if delay_alloc then 1 else 0)

{#
fun MXNDArrayCreateEx as mxNDArrayCreateEx_
    {
        withArray* `[MX_UINT]',
        id `MX_UINT',
        `CInt',
        `CInt',
        `CInt',
        `CInt',
        alloca- `NDArrayHandle' peekNDArrayHandle*
    } -> `CInt'
#}

mxNDArrayCreateEx :: [Int] -> Int -> Int -> Bool -> Int -> IO NDArrayHandle
mxNDArrayCreateEx shape devtype devid delay_alloc dtype = do
    let shape_   = fromIntegral <$> shape
        dim_     = fromIntegral (length shape_)
        devtype_ = fromIntegral devtype
        devid_   = fromIntegral devid
        dtype_   = fromIntegral dtype
    checked $ mxNDArrayCreateEx_ shape_ dim_ devtype_ devid_ (if delay_alloc then 1 else 0) dtype_


{#
fun MXNDArraySyncCopyFromCPU as mxNDArraySyncCopyFromCPU_
    {
        `NDArrayHandle',
        id `Ptr ()',
        `CSize'
    } -> `CInt'
#}

mxNDArraySyncCopyFromCPU :: NDArrayHandle -> Ptr () -> Int -> IO ()
mxNDArraySyncCopyFromCPU array ptr size =
    checked $ mxNDArraySyncCopyFromCPU_ array ptr (fromIntegral size)

{#
fun MXNDArraySyncCopyToCPU as mxNDArraySyncCopyToCPU_
    {
        `NDArrayHandle',
        id `Ptr ()',
        `CSize'
    } -> `CInt'
#}

mxNDArraySyncCopyToCPU :: NDArrayHandle -> Ptr () -> Int -> IO ()
mxNDArraySyncCopyToCPU array ptr size =
    checked $ mxNDArraySyncCopyToCPU_ array ptr (fromIntegral size)

{#
fun MXNDArraySyncCopyFromNDArray as mxNDArraySyncCopyFromNDArray_
    {
        `NDArrayHandle',
        `NDArrayHandle',
        `CInt'
    } -> `CInt'
#}

mxNDArraySyncCopyFromNDArray :: NDArrayHandle -> NDArrayHandle -> Int -> IO ()
mxNDArraySyncCopyFromNDArray array_dst array_src blob =
    checked $ mxNDArraySyncCopyFromNDArray_ array_dst array_src (fromIntegral blob)

{#
fun MXNDArrayWaitToRead as mxNDArrayWaitToRead_
    {
        `NDArrayHandle'
    } -> `CInt'
#}

mxNDArrayWaitToRead :: NDArrayHandle -> IO ()
mxNDArrayWaitToRead = checked . mxNDArrayWaitToRead_

{#
fun MXNDArrayWaitToWrite as mxNDArrayWaitToWrite_
    {
        `NDArrayHandle'
    } -> `CInt'
#}

mxNDArrayWaitToWrite :: NDArrayHandle -> IO ()
mxNDArrayWaitToWrite = checked . mxNDArrayWaitToWrite_

{#
fun MXNDArrayWaitAll as mxNDArrayWaitAll_
    {
    } -> `CInt'
#}

mxNDArrayWaitAll :: IO ()
mxNDArrayWaitAll = checked mxNDArrayWaitAll_

#if MXNet_MAJOR==1 && MXNet_MINOR<6
{#
fun MXNDArraySlice as mxNDArraySlice_
    {
        `NDArrayHandle',
        `MX_UINT',
        `MX_UINT',
        alloca- `NDArrayHandle' peekNDArrayHandle*
    } -> `CInt'
#}
#else
{#
fun MXNDArraySlice as mxNDArraySlice_
    {
        `NDArrayHandle',
        `CUInt',
        `CUInt',
        alloca- `NDArrayHandle' peekNDArrayHandle*
    } -> `CInt'
#}
#endif

mxNDArraySlice :: NDArrayHandle -> Int -> Int -> IO NDArrayHandle
mxNDArraySlice array begin end = do
    let begin_ = fromIntegral begin
        end_   = fromIntegral end
    checked $ mxNDArraySlice_ array begin_ end_


#if MXNet_MAJOR==1 && MXNet_MINOR<6
{#
fun MXNDArrayAt as mxNDArrayAt_
    {
        `NDArrayHandle',
        `MX_UINT',
        alloca- `NDArrayHandle' peekNDArrayHandle*
    } -> `CInt'
#}
#else
{#
fun MXNDArrayAt as mxNDArrayAt_
    {
        `NDArrayHandle',
        `CUInt',
        alloca- `NDArrayHandle' peekNDArrayHandle*
    } -> `CInt'
#}
#endif

mxNDArrayAt :: NDArrayHandle -> Int -> IO NDArrayHandle
mxNDArrayAt array index = do
    checked $ mxNDArrayAt_ array (fromIntegral index)

{#
fun MXNDArrayGetStorageType as mxNDArrayGetStorageType_
    {
        `NDArrayHandle',
        alloca- `CInt' peek*
    } -> `CInt'
#}

mxNDArrayGetStorageType :: NDArrayHandle -> IO Int
mxNDArrayGetStorageType array = do
    storageType <- checked $ mxNDArrayGetStorageType_ array
    return $ fromIntegral storageType

{#
fun MXNDArrayReshape as mxNDArrayReshape_
    {
        `NDArrayHandle',
        `CInt',
        withIntegralArray* `[CInt]',
        alloca- `NDArrayHandle' peekNDArrayHandle*
    } -> `CInt'
#}

mxNDArrayReshape :: NDArrayHandle -> [Int] -> IO NDArrayHandle
mxNDArrayReshape array shape = do
    let shape_ = fromIntegral <$> shape
        num_   = fromIntegral $ length shape_
    checked $ mxNDArrayReshape_ array num_ shape_

{#
fun MXNDArrayGetShape as mxNDArrayGetShape_
    {
        `NDArrayHandle',
        alloca- `MX_UINT' peek*,
        alloca- `Ptr MX_UINT' peek*
    } -> `CInt'
#}

mxNDArrayGetShape :: NDArrayHandle -> IO [Int]
mxNDArrayGetShape array = do
    (size, ptr) <- checked $ mxNDArrayGetShape_ array
    shape <- peekArray (fromIntegral size) ptr
    return $ fromIntegral <$> shape

-- MXImperativeInvoke is hacky.
-- num-outputs
--   0: create new NDArrayHandler in the array-of-NDArrayHandle
--   length of array-of-NDArrayHandle (non-0): reuse NDArrayHandler in the array-of-NDArrayHandle
{#
fun MXImperativeInvoke as mxImperativeInvoke_
    {
        `AtomicSymbolCreator',
        `CInt',
        withNDArrayHandleArray* `[NDArrayHandle]',
        id `Ptr CInt',                    -- num-outputs
        id `Ptr (Ptr NDArrayHandlePtr)',  -- array-of-NDArrayHandle
        `CInt',
        withCStringArrayT* `[Text]',
        withCStringArrayT* `[Text]'
    } -> `CInt'
#}

-- | Invoke a nnvm op and imperative function.
mxImperativeInvoke :: AtomicSymbolCreator
                   -> [NDArrayHandle]
                   -> [(Text, Text)]
                   -> Maybe [NDArrayHandle]
                   -> IO [NDArrayHandle]
mxImperativeInvoke creator inputs params outputs = do
    let (keys, values) = unzip params
        ninput = fromIntegral $ length inputs
        nparam = fromIntegral $ length params
    case outputs of
        Nothing -> alloca $ \(pn :: Ptr CInt) ->
            alloca $ \(pp :: Ptr (Ptr NDArrayHandlePtr)) -> do
                poke pn 0
                poke pp C2HSImp.nullPtr
                checked $ mxImperativeInvoke_ creator ninput inputs pn pp nparam keys values
                n' <- peek pn
                p' <- peek pp
                if n' == 0
                    then return []
                    else do
                        pa <- peekArray (fromIntegral n') (p' :: Ptr NDArrayHandlePtr)
                        mapM newNDArrayHandle pa
        Just out -> alloca $ \(pn :: Ptr CInt) ->
            alloca $ \(pp :: Ptr (Ptr NDArrayHandlePtr)) -> do
                n' <- withNDArrayHandleArray out $ \p' -> do
                    poke pn (fromIntegral $ length out)
                    poke pp p'
                    checked $ mxImperativeInvoke_ creator ninput inputs pn pp nparam keys values
                    peek pn
                return $ take (fromIntegral n') out

{#
fun MXNDArrayGetContext as mxNDArrayGetContext_
    {
        `NDArrayHandle',
        alloca- `CInt' peek*,
        alloca- `CInt' peek*
    } -> `CInt'
#}

mxNDArrayGetContext :: NDArrayHandle -> IO (Int, Int)
mxNDArrayGetContext handle = do
    (devtyp, devidx) <- checked $ mxNDArrayGetContext_ handle
    return (fromIntegral devtyp, fromIntegral devidx)


#if MXNet_MAJOR==1 && MXNet_MINOR<6
{#
fun MXNDArraySave as mxNDArraySave_
    {
        withCStringT* `Text',
        `MX_UINT',
        withNDArrayHandleArray* `[NDArrayHandle]',
        withCStringArrayT* `[Text]'
    } -> `CInt'
#}
#else
{#
fun MXNDArraySave as mxNDArraySave_
    {
        withCStringT* `Text',
        `CUInt',
        withNDArrayHandleArray* `[NDArrayHandle]',
        withCStringArrayT* `[Text]'
    } -> `CInt'
#}
#endif

mxNDArraySave ::  Text -> [(Text, NDArrayHandle)] -> IO ()
mxNDArraySave filename keyvals = do
    let num = length keyvals
        (keys, vals) = unzip keyvals
    checked $ mxNDArraySave_ filename (fromIntegral num) vals keys

{#
fun MXNDArrayLoad as mxNDArrayLoad_
    {
        withCStringT* `Text',
        alloca- `MX_UINT' peek*,
        alloca- `Ptr NDArrayHandlePtr' peek*,
        alloca- `MX_UINT' peek*,
        alloca- `Ptr CString' peek*
    } -> `CInt'
#}

mxNDArrayLoad :: Text -> IO [(Text, NDArrayHandle)]
mxNDArrayLoad path = do
    (numArrays, ptrArrays, numNames, ptrNames) <- checked $ mxNDArrayLoad_ path
    pa <- peekArray (fromIntegral numArrays) ptrArrays
    arrays <- mapM newNDArrayHandle pa
    names <- peekCStringArrayT (fromIntegral numNames) ptrNames
    return $ zip names arrays
