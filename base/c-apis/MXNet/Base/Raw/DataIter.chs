module MXNet.Base.Raw.DataIter where

import Data.Typeable (Typeable)
import Foreign.Marshal (alloca, peekArray)
import Foreign.Storable (Storable(..))
import Foreign.ForeignPtr (newForeignPtr, finalizeForeignPtr)
import C2HS.C.Extra.Marshal (peekString, peekStringArray, peekIntegralArray)
import GHC.Generics (Generic)
import Control.Monad ((>=>))

{# import MXNet.Base.Raw.Common #}
{# import MXNet.Base.Raw.NDArray #}

#include <mxnet/c_api.h>
#include <nnvm/c_api.h>

{# typedef mx_uint MX_UINT#}
{# default in `MX_UINT' [mx_uint] id #}

{# typedef uint64_t UINT64#}
{# default in `UINT64' [uint64_t] id #}

{#
pointer DataIterCreator newtype
#}

deriving instance Storable DataIterCreator

{#
pointer DataIterHandle foreign finalizer MXDataIterFree as mxDataIterFree newtype 
#}

deriving instance Generic DataIterHandle

type DataIterHandlePtr = Ptr DataIterHandle

newDataIterHandle :: DataIterHandlePtr -> IO DataIterHandle
newDataIterHandle = newForeignPtr mxDataIterFree >=> return . DataIterHandle

peekDataIterHandle :: Ptr DataIterHandlePtr -> IO DataIterHandle
peekDataIterHandle = peek >=> newDataIterHandle

finalizeDataIterHandle :: DataIterHandle -> IO ()
finalizeDataIterHandle (DataIterHandle fptr) = finalizeForeignPtr fptr

{#
fun MXListDataIters as mxListDataIters_
    { 
        alloca- `MX_UINT' peek*, 
        alloca- `Ptr DataIterCreator' peek*
    } -> `CInt' 
#}

mxListDataIters :: IO [DataIterCreator]
mxListDataIters = do
    (cnt, ptr) <- checked mxListDataIters_
    peekArray (fromIntegral cnt :: Int) ptr

{#
fun MXDataIterGetIterInfo as mxDataIterGetIterInfo_
    { 
        `DataIterCreator', 
        alloca- `String' peekString*,
        alloca- `String' peekString*,
        alloca- `MX_UINT' peek*,
        alloca- `Ptr (Ptr CChar)' peek*,
        alloca- `Ptr (Ptr CChar)' peek*,
        alloca- `Ptr (Ptr CChar)' peek*
    } -> `CInt'
#}

mxDataIterGetIterInfo :: DataIterCreator -> IO (String, String, [String], [String], [String])
mxDataIterGetIterInfo dataitercreator = do
    (name, descr, num_args, arg_names, arg_type_infos, arg_descs) <- checked $ mxDataIterGetIterInfo_ dataitercreator
    let num_args_ = fromIntegral num_args
    arg_names_ <- peekStringArray num_args_ arg_names
    arg_type_infos_ <- peekStringArray num_args_ arg_type_infos
    arg_descs_ <- peekStringArray num_args_ arg_descs
    return (name, descr, arg_names_, arg_type_infos_, arg_descs_)

{#
fun MXDataIterCreateIter as mxDataIterCreateIter_
    { 
        `DataIterCreator', 
        `MX_UINT',
        withStringArray* `[String]',
        withStringArray* `[String]',
        alloca- `DataIterHandle' peekDataIterHandle*
    } -> `CInt'
#}

mxDataIterCreateIter :: DataIterCreator -> [String] -> [String] -> IO DataIterHandle
mxDataIterCreateIter dataitercreator keys vals = do
    let num_args = fromIntegral (length keys)
    checked $ mxDataIterCreateIter_ dataitercreator num_args keys vals

{#
fun MXDataIterNext as mxDataIterNext_
    { 
        `DataIterHandle', 
        alloca- `CInt' peek*
    } -> `CInt'
#}

mxDataIterNext :: DataIterHandle -> IO Int
mxDataIterNext dataiter = do
    next <- checked $ mxDataIterNext_ dataiter
    return $ fromIntegral next

{#
fun MXDataIterBeforeFirst as mxDataIterBeforeFirst_
    { 
        `DataIterHandle'
    } -> `CInt'
#}

mxDataIterBeforeFirst :: DataIterHandle -> IO ()
mxDataIterBeforeFirst = checked . mxDataIterBeforeFirst_

{#
fun MXDataIterGetData as mxDataIterGetData_
    { 
        `DataIterHandle',
        alloca- `NDArrayHandle' peekNDArrayHandle*
    } -> `CInt'
#}

mxDataIterGetData :: DataIterHandle -> IO NDArrayHandle
mxDataIterGetData = checked . mxDataIterGetData_

{#
fun MXDataIterGetIndex as mxDataIterGetIndex_
    { 
        `DataIterHandle',
        alloca- `Ptr UINT64' peek*,
        alloca- `UINT64' peek*
    } -> `CInt'
#}

mxDataIterGetIndex :: DataIterHandle -> IO [Integer]
mxDataIterGetIndex dataiter = do
    (ptr, cnt) <- checked $ mxDataIterGetIndex_ dataiter
    peekIntegralArray (fromIntegral cnt) ptr

{#
fun MXDataIterGetPadNum as mxDataIterGetPadNum_
    { 
        `DataIterHandle',
        alloca- `CInt' peek*
    } -> `CInt'
#}

mxDataIterGetPadNum :: DataIterHandle -> IO Int
mxDataIterGetPadNum dataiter = do
    next <- checked $ mxDataIterGetPadNum_ dataiter
    return $ fromIntegral next

{#
fun MXDataIterGetLabel as mxDataIterGetLabel_
    { 
        `DataIterHandle',
        alloca- `NDArrayHandle' peekNDArrayHandle*
    } -> `CInt'
#}

mxDataIterGetLabel :: DataIterHandle -> IO NDArrayHandle
mxDataIterGetLabel = checked . mxDataIterGetLabel_

