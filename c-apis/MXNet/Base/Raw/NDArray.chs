module MXNet.Base.Raw.NDArray where

import Foreign.Marshal (alloca, withArray)
import Foreign.Storable (peek)
import Foreign.ForeignPtr (newForeignPtr, touchForeignPtr)
import Foreign.ForeignPtr.Unsafe (unsafeForeignPtrToPtr)
import C2HS.C.Extra.Marshal (withIntegralArray)
import GHC.Generics (Generic)
import Control.Monad ((>=>))

{# import MXNet.Base.Raw.Common #}

#include <mxnet/c_api.h>

{# typedef size_t CSize#}
{# default in `CSize' [size_t] fromIntegral #}
{# typedef mx_uint MX_UINT#}
{# default in `MX_UINT' [mx_uint] id #}

-- NDArray
{#
pointer NDArrayHandle foreign finalizer MXNDArrayFree as mxNDArrayFree newtype 
#}

deriving instance Generic NDArrayHandle
type NDArrayHandlePtr = Ptr NDArrayHandle

newNDArrayHandle :: NDArrayHandlePtr -> IO NDArrayHandle
newNDArrayHandle = newForeignPtr mxNDArrayFree >=> return . NDArrayHandle

peekNDArrayHandle :: Ptr NDArrayHandlePtr -> IO NDArrayHandle
peekNDArrayHandle = peek >=> newNDArrayHandle

withNDArrayHandleArray :: [NDArrayHandle] -> (Ptr NDArrayHandlePtr -> IO r) -> IO r
withNDArrayHandleArray array io = do
    let unNDArrayHandle (NDArrayHandle fptr) = fptr
    r <- withArray (map (unsafeForeignPtrToPtr . unNDArrayHandle) array) io
    mapM_ (touchForeignPtr . unNDArrayHandle) array
    return r

{#
fun MXNDArrayCreateNone as mxNDArrayCreateNone
    {
        alloca- `NDArrayHandle' peekNDArrayHandle*
    } -> `CInt'
#}

{#
fun MXNDArrayCreate as mxNDArrayCreate
    {
        withArray* `[MX_UINT]',
        id `MX_UINT',
        `CInt',
        `CInt',
        `CInt',
        alloca- `NDArrayHandle' peekNDArrayHandle*
    } -> `CInt'
#}

{#
fun MXNDArrayCreateEx as mxNDArrayCreateEx
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

{#
fun MXNDArraySyncCopyFromCPU as mxNDArraySyncCopyFromCPU
    {
        `NDArrayHandle',
        id `Ptr ()',
        `CSize'
    } -> `CInt'
#}

{#
fun MXNDArraySyncCopyToCPU as mxNDArraySyncCopyToCPU
    {
        `NDArrayHandle',
        id `Ptr ()',
        `CSize'
    } -> `CInt'
#}

{#
fun MXNDArraySyncCopyFromNDArray as mxNDArraySyncCopyFromNDArray
    {
        `NDArrayHandle',
        `NDArrayHandle',
        `CInt'
    } -> `CInt'
#}

{#
fun MXNDArrayWaitToRead as mxNDArrayWaitToRead
    {
        `NDArrayHandle'
    } -> `CInt'
#}

{#
fun MXNDArrayWaitToWrite as mxNDArrayWaitToWrite
    {
        `NDArrayHandle'
    } -> `CInt'
#}

{#
fun MXNDArrayWaitAll as mxNDArrayWaitAll
    {
    } -> `CInt'
#}

{#
fun MXNDArraySlice as mxNDArraySlice
    {
        `NDArrayHandle',
        `MX_UINT',
        `MX_UINT',
        alloca- `NDArrayHandle' peekNDArrayHandle*
    } -> `CInt'
#}

{#
fun MXNDArrayAt as mxNDArrayAt
    {
        `NDArrayHandle',
        `MX_UINT',
        alloca- `NDArrayHandle' peekNDArrayHandle*
    } -> `CInt'
#}

{#
fun MXNDArrayGetStorageType as mxNDArrayGetStorageType
    {
        `NDArrayHandle',
        alloca- `CInt' peek*
    } -> `CInt'
#}

{#
fun MXNDArrayReshape as mxNDArrayReshape
    {
        `NDArrayHandle',
        `CInt',
        withIntegralArray* `[CInt]',
        alloca- `NDArrayHandle' peekNDArrayHandle*
    } -> `CInt'
#}

{#
fun MXNDArrayGetShape as mxNDArrayGetShape_
    {
        `NDArrayHandle',
        alloca- `MX_UINT' peek*,
        alloca- `Ptr MX_UINT' peek*
    } -> `CInt'
#}
