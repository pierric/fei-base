module MXNet.Base.Raw.Common where

import RIO
import qualified RIO.Text as T
import Data.Tuple.Ops (Unconsable, uncons)
import Foreign.Marshal (alloca, peekArray, withArray)
import Foreign.Storable (Storable(..))
import Foreign.C.Types
import Foreign.C.String (CString, peekCString, withCString)
import Foreign.Ptr
import GHC.Generics (Generic)

type MX_UINT  = C2HSImp.CUInt
type MX_CCHAR = C2HSImp.CChar

{# typedef char CChar #}
{# default in  `Text' [char *] withCStringT* #}
{# default out `Text' [char *] peekCStringT* #}

data MXNetError = MXNetError Text
    deriving Typeable
instance Exception MXNetError

instance Show MXNetError where
    show (MXNetError msg) = "an error occurred in MXNet.\n" ++ T.unpack msg

newtype WrapText = WrapText {unWrapText :: Text}
deriving instance Generic WrapText
deriving instance Generic C2HSImp.CInt
deriving instance Generic C2HSImp.CUInt

checked :: Unconsable t CInt r => IO t -> IO r
checked call = do
    (res, ret) <- uncons <$> call
    if res < 0
      then do err <- mxGetLastError
              throwIO $ MXNetError err
      else return ret

peekCStringT :: CString -> IO Text
peekCStringT = fmap T.pack . peekCString

peekCStringPtrT :: Ptr CString -> IO Text
peekCStringPtrT ptr = do
    cstr <- peek ptr
    if | cstr == nullPtr -> return ""
       | otherwise -> peekCStringT cstr

peekCStringArrayT :: Int -> Ptr CString -> IO [Text]
peekCStringArrayT cnt ptr = peekArray cnt ptr >>= mapM peekCStringT

withCStringT :: Text -> (CString -> IO a) -> IO a
withCStringT str = withCString (T.unpack str)

-- TODO: Does it worth of any opt for the withCStringArrayT?
-- withCStringArrayT :: [Text] -> (Ptr CString -> IO a) -> IO a
-- withCStringArrayT arr act = do
--     let nul = T.singleton '\NUL'
--         whole = T.concat $ map (`T.append` nul) arr
--     T.encodeUtf8 whole

withCStringArrayT :: [Text] -> (Ptr CString -> IO a) -> IO a
withCStringArrayT strs act = go strs []
  where
    go [] all = withArray (reverse all) act
    go (s:ss) all = withCStringT s (go ss . (:all))

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

mxListAllOpNames :: IO [Text]
mxListAllOpNames = do
    (cnt, ptr) <- checked mxListAllOpNames_
    peekCStringArrayT (fromIntegral cnt :: Int) ptr

{#
fun MXGetLastError as mxGetLastError
    {
    } -> `Text' peekCStringT*
#}

{#
pointer AtomicSymbolCreator newtype
#}

deriving instance Storable AtomicSymbolCreator

fromOpHandle :: OpHandle -> AtomicSymbolCreator
fromOpHandle (OpHandle ptr) = AtomicSymbolCreator (C2HSImp.castPtr ptr)

{#
fun MXSymbolListAtomicSymbolCreators as mxSymbolListAtomicSymbolCreators_
    {
        alloca- `MX_UINT' peek*,
        alloca- `Ptr AtomicSymbolCreator' peek*
    } -> `CInt'
#}

mxSymbolListAtomicSymbolCreators :: IO [AtomicSymbolCreator]
mxSymbolListAtomicSymbolCreators = do
    (cnt, ptr) <- checked $ mxSymbolListAtomicSymbolCreators_
    peekArray (fromIntegral cnt) ptr

mxSymbolGetAtomicSymbolCreatorAt :: Int -> IO AtomicSymbolCreator
mxSymbolGetAtomicSymbolCreatorAt idx = do
    (cnt, ptr) <- checked $ mxSymbolListAtomicSymbolCreators_
    peekElemOff ptr idx

{#
fun MXSymbolGetAtomicSymbolName as mxSymbolGetAtomicSymbolName_
    {
        `AtomicSymbolCreator',
        alloca- `Text' peekCStringPtrT*
    } -> `CInt'
#}

mxSymbolGetAtomicSymbolName :: AtomicSymbolCreator -> IO Text
mxSymbolGetAtomicSymbolName = fmap unWrapText . checked . fmap (second WrapText) . mxSymbolGetAtomicSymbolName_

{#
fun MXSymbolGetAtomicSymbolInfo as mxSymbolGetAtomicSymbolInfo_
    {
        `AtomicSymbolCreator',
        alloca- `Text' peekCStringPtrT*,
        alloca- `Text' peekCStringPtrT*,
        alloca- `MX_UINT' peek*,
        alloca- `Ptr CString' peek*,
        alloca- `Ptr CString' peek*,
        alloca- `Ptr CString' peek*,
        alloca- `Text' peekCStringPtrT*,
        alloca- `Text' peekCStringPtrT*
    } -> `CInt'
#}

mxSymbolGetAtomicSymbolInfo :: AtomicSymbolCreator
                            -> IO (Text,
                                   Text,
                                   [Text],
                                   [Text],
                                   [Text],
                                   Text,
                                   Text)
mxSymbolGetAtomicSymbolInfo creator = do
    (name, desc, argcnt, argname, argtype, argdesc, key_var_num_args, rettyp) <- checked $ mxSymbolGetAtomicSymbolInfo_ creator
    let n = fromIntegral argcnt
    argname <- peekCStringArrayT n argname
    argtype <- peekCStringArrayT n argtype
    argdesc <- peekCStringArrayT n argdesc
    return (name, desc, argname, argtype, argdesc, key_var_num_args, rettyp)

{#
fun MXNotifyShutdown as mxNotifyShutdown_
    {
    } -> `CInt'
#}

mxNotifyShutdown :: IO ()
mxNotifyShutdown = checked mxNotifyShutdown_

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

nnListAllOpNames :: IO [Text]
nnListAllOpNames = do
    (cnt, ptr) <- checked nnListAllOpNames_
    peekCStringArrayT (fromIntegral cnt) ptr

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
        withCStringT* `Text',
        alloca- `OpHandle' peek*
    } -> `CInt'
#}

nnGetOpHandle :: Text -> IO OpHandle
nnGetOpHandle = checked . nnGetOpHandle_

{#
fun NNGetOpInfo as nnGetOpInfo_
    {
        `OpHandle',
        alloca- `Text' peekCStringPtrT*,
        alloca- `Text' peekCStringPtrT*,
        alloca- `NN_UINT' peek*,
        alloca- `Ptr CString' peek*,
        alloca- `Ptr CString' peek*,
        alloca- `Ptr CString' peek*,
        alloca- `Text' peekCStringPtrT*
    } -> `CInt'
#}

nnGetOpInfo :: OpHandle -> IO (Text, Text, [Text], [Text], [Text], Text)
nnGetOpInfo op = do
    (name, desc, num_args, ptr_arg_names, ptr_arg_types, ptr_arg_descs, ret_type) <- checked $ nnGetOpInfo_ op
    let num_args_ = fromIntegral num_args
    arg_names <- peekCStringArrayT num_args_ ptr_arg_names
    arg_types <- peekCStringArrayT num_args_ ptr_arg_types
    arg_descs <- peekCStringArrayT num_args_ ptr_arg_descs
    return (name, desc, arg_names, arg_types, arg_descs, ret_type)

{#
fun NNGetLastError as nnGetLastError
    {
    } -> `Text' peekCStringT*
#}
