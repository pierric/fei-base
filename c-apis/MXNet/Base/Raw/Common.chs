module MXNet.Base.Raw.Common where

import Control.Exception.Base (Exception, throwIO)
import Data.Typeable (Typeable)
import Data.Tuple.Ops (Unconsable, uncons)
import Foreign.Marshal (alloca, peekArray)
import Foreign.Storable (Storable(..))
import C2HS.C.Extra.Marshal (peekString, peekStringArray)
import GHC.Generics (Generic)

type MX_UINT  = C2HSImp.CUInt
type MX_CCHAR = C2HSImp.CChar
type Ptr      = C2HSImp.Ptr
type CSize    = C2HSImp.CSize
type CInt     = C2HSImp.CInt
type CChar    = C2HSImp.CChar

data MXNetError = MXNetError String
    deriving (Typeable, Show)
instance Exception MXNetError

deriving instance Generic C2HSImp.CInt

checked :: Unconsable t CInt r => IO t -> IO r
checked call = do
    (res, ret) <- uncons <$> call
    if res < 0 
      then do err <- mxGetLastError
              throwIO $ MXNetError err
      else return ret

#include <mxnet/c_api.h>
#include <nnvm/c_api.h>

{#
fun MXGetVersion as mxGetVersion_
    {
        alloca- `CInt' peek*
    } -> `CInt'
#}

mxGetVersion :: IO Int
mxGetVersion = fromIntegral <$> checked mxGetVersion_

{#
fun MXListAllOpNames as mxListAllOpNames_
    { 
        alloca- `MX_UINT' peek*, 
        alloca- `Ptr (Ptr MX_CCHAR)' peek*
    } -> `CInt' 
#}

mxListAllOpNames :: IO [String]
mxListAllOpNames = do
    (cnt, ptr) <- checked mxListAllOpNames_
    peekStringArray (fromIntegral cnt :: Int) ptr

{#
fun MXGetLastError as mxGetLastError
    { 
    } -> `String' 
#}

---------------------------------------------------
type NN_UINT  = C2HSImp.CUInt

{# pointer OpHandle newtype #}
deriving instance Storable OpHandle
deriving instance Generic  OpHandle

{# pointer GraphHandle newtype #}
deriving instance Storable GraphHandle
deriving instance Generic  GraphHandle

{#
fun NNListAllOpNames as nnListAllOpNames_
    { 
        alloca- `NN_UINT' peek*, 
        alloca- `Ptr (Ptr MX_CCHAR)' peek*
    } -> `CInt' 
#}

nnListAllOpNames :: IO [String]
nnListAllOpNames = do
    (cnt, ptr) <- checked nnListAllOpNames_
    peekStringArray (fromIntegral cnt :: Int) ptr

{#
fun NNListUniqueOps as nnListUniqueOps_
    { 
        alloca- `NN_UINT' peek*, 
        alloca- `Ptr OpHandle' peek*
    } -> `CInt' 
#}

nnListUniqueOps :: IO [OpHandle]
nnListUniqueOps = do
    (cnt, ptr) <- checked (nnListUniqueOps_ :: IO (CInt, NN_UINT, Ptr OpHandle))
    peekArray (fromIntegral cnt :: Int) (ptr :: Ptr OpHandle)

{#
fun NNGetOpHandle as nnGetOpHandle_
    {
        `String',
        alloca- `OpHandle' peek*
    } -> `CInt'
#}

nnGetOpHandle :: String -> IO OpHandle
nnGetOpHandle = checked . nnGetOpHandle_

{#
fun NNGetOpInfo as nnGetOpInfo_
    {
        `OpHandle',
        alloca- `String' peekString*,
        alloca- `String' peekString*,
        alloca- `NN_UINT' peek*,
        alloca- `Ptr (Ptr CChar)' peek*,
        alloca- `Ptr (Ptr CChar)' peek*,
        alloca- `Ptr (Ptr CChar)' peek*,
        alloca- `String' peekString*
    } -> `CInt'
#}

nnGetOpInfo :: OpHandle -> IO (String, String, [String], [String], [String], String)
nnGetOpInfo op = do
    (name, desc, num_args, ptr_arg_names, ptr_arg_types, ptr_arg_descs, ret_type) <- checked $ nnGetOpInfo_ op
    let num_args_ = fromIntegral num_args
    arg_names <- peekStringArray num_args_ ptr_arg_names
    arg_types <- peekStringArray num_args_ ptr_arg_types
    arg_descs <- peekStringArray num_args_ ptr_arg_descs
    return (name, desc, arg_names, arg_types, arg_descs, ret_type)

{#
fun NNGetLastError as nnGetLastError
    { 
    } -> `String' 
#}
