{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.Base.Raw.Symbol where

import RIO
import qualified RIO.Text as T
import qualified RIO.HashMap as M
import qualified RIO.Vector.Boxed as VB
import qualified RIO.Vector.Boxed.Partial as VB
import Control.Lens ((?~), at, non)
import Foreign.Marshal (alloca, withArray, peekArray, allocaBytesAligned)
import Foreign.Storable (Storable(..))
import Foreign.Ptr (FunPtr)
import Foreign.C.Types
import Foreign.C.String (CString)
import Foreign.Ptr
import Foreign.Concurrent (newForeignPtr)
import Foreign.ForeignPtr (touchForeignPtr)
import Foreign.ForeignPtr.Unsafe (unsafeForeignPtrToPtr)

{# import MXNet.Base.Raw.Common #}
{# import MXNet.Base.Raw.NDArray #}

#include <mxnet/c_api.h>

{# typedef size_t CSize#}
{# default in `CSize' [size_t] fromIntegral #}
{# typedef mx_uint MX_UINT#}
{# default in `MX_UINT' [mx_uint] id #}

{#
pointer SymbolHandle foreign newtype
#}

deriving instance Generic SymbolHandle
deriving instance Show SymbolHandle
type SymbolHandlePtr = Ptr SymbolHandle

touchSymbolHandle :: SymbolHandle -> IO ()
touchSymbolHandle (SymbolHandle fptr) = touchForeignPtr fptr

newSymbolHandle :: SymbolHandlePtr -> IO SymbolHandle
newSymbolHandle ptr = newForeignPtr ptr (mxSymbolFree ptr) >>= return . SymbolHandle

peekSymbolHandle :: Ptr SymbolHandlePtr -> IO SymbolHandle
peekSymbolHandle = peek >=> newSymbolHandle

withSymbolHandleArray :: [SymbolHandle] -> (Ptr SymbolHandlePtr -> IO r) -> IO r
withSymbolHandleArray array io = do
    let unSymbolHandle (SymbolHandle fptr) = fptr
    r <- withArray (map (unsafeForeignPtrToPtr . unSymbolHandle) array) io
    mapM_ (touchForeignPtr . unSymbolHandle) array
    return r

{#
fun MXSymbolFree as mxSymbolFree_
    {
        id `SymbolHandlePtr'
    } -> `CInt'
#}

mxSymbolFree :: HasCallStack => SymbolHandlePtr -> IO ()
mxSymbolFree = checked . mxSymbolFree_

{#
fun MXSymbolCreateAtomicSymbol as mxSymbolCreateAtomicSymbol_
    {
        `AtomicSymbolCreator',
        `CUInt',
        withCStringArrayT* `[Text]',
        withCStringArrayT* `[Text]',
        alloca- `SymbolHandle' peekSymbolHandle*
    } -> `CInt'
#}

mxSymbolCreateAtomicSymbol :: HasCallStack
                           => AtomicSymbolCreator
                           -> [Text]
                           -> [Text]
                           -> IO SymbolHandle
mxSymbolCreateAtomicSymbol creator keys vals = do
    let argcnt = fromIntegral $ length keys
    checked $ mxSymbolCreateAtomicSymbol_ creator argcnt keys vals

{#
fun MXSymbolCreateVariable as mxSymbolCreateVariable_
    {
        withCStringT* `Text',
        alloca- `SymbolHandle' peekSymbolHandle*
    } -> `CInt'
#}

mxSymbolCreateVariable :: HasCallStack => Text -> IO SymbolHandle
mxSymbolCreateVariable = checked . mxSymbolCreateVariable_

{#
fun MXSymbolCreateGroup as mxSymbolCreateGroup_
    {
        `CUInt',
        withSymbolHandleArray* `[SymbolHandle]',
        alloca- `SymbolHandle' peekSymbolHandle*
    } -> `CInt'
#}

mxSymbolCreateGroup :: HasCallStack => [SymbolHandle] -> IO SymbolHandle
mxSymbolCreateGroup syms = checked $ mxSymbolCreateGroup_ (fromIntegral $ length syms) syms

{#
fun MXSymbolCreateFromJSON as mxSymbolCreateFromJSON_
    {
        withCStringT* `Text',
        alloca- `SymbolHandle' peekSymbolHandle*
    } -> `CInt'
#}

mxSymbolCreateFromJSON :: HasCallStack => Text -> IO SymbolHandle
mxSymbolCreateFromJSON = checked . mxSymbolCreateFromJSON_

{#
fun MXSymbolSaveToJSON as mxSymbolSaveToJSON_
    {
        `SymbolHandle',
        alloca- `Text' peekCStringPtrT*
    } -> `CInt'
#}

mxSymbolSaveToJSON :: HasCallStack => SymbolHandle -> IO Text
mxSymbolSaveToJSON = fmap unWrapText . checked . fmap (second WrapText) . mxSymbolSaveToJSON_

{#
fun MXSymbolCopy as mxSymbolCopy_
    {
        `SymbolHandle',
        alloca- `SymbolHandle' peekSymbolHandle*
    } -> `CInt'
#}

mxSymbolCopy :: HasCallStack => SymbolHandle -> IO SymbolHandle
mxSymbolCopy = checked . mxSymbolCopy_

{#
fun MXSymbolPrint as mxSymbolPrint_
    {
        `SymbolHandle',
        alloca- `Text' peekCStringPtrT*
    } -> `CInt'
#}

mxSymbolPrint :: HasCallStack => SymbolHandle -> IO Text
mxSymbolPrint = fmap unWrapText . checked . fmap (second WrapText) . mxSymbolPrint_

{#
fun MXSymbolGetName as mxSymbolGetName_
    {
        `SymbolHandle',
        alloca- `Ptr CString' id,
        alloca- `CInt' peek*
    } -> `CInt'
#}

mxSymbolGetName :: HasCallStack => SymbolHandle -> IO (Maybe Text)
mxSymbolGetName symbol = do
    (out, succ) <- checked $ mxSymbolGetName_ symbol
    if succ == 0 then
        Just <$> peekCStringPtrT out
    else
        return Nothing

{#
fun MXSymbolGetAttr as mxSymbolGetAttr_
    {
        `SymbolHandle',
        withCStringT* `Text',
        alloca- `Ptr CString' id,
        alloca- `CInt' peek*
    } -> `CInt'
#}

mxSymbolGetAttr :: HasCallStack => SymbolHandle
                -> Text
                -> IO (Maybe Text)
mxSymbolGetAttr symbol key = do
    (pstr, succ) <- checked $ mxSymbolGetAttr_ symbol key
    if succ == 0 then
        Just <$> peekCStringPtrT pstr
    else
        return Nothing

{#
fun MXSymbolSetAttr as mxSymbolSetAttr_
    {
        `SymbolHandle',
        withCStringT* `Text',
        withCStringT* `Text'
    } -> `CInt'
#}

mxSymbolSetAttr :: HasCallStack => SymbolHandle
                -> Text
                -> Text
                -> IO ()
mxSymbolSetAttr symbol key val = checked $ mxSymbolSetAttr_ symbol key val

{#
fun MXSymbolListAttr as mxSymbolListAttr_
    {
        `SymbolHandle',
        alloca- `MX_UINT' peek*,
        alloca- `Ptr CString' peek*
    } -> `CInt'
#}

mxSymbolListAttr :: HasCallStack => SymbolHandle
                 -> IO (HashMap Text (HashMap Text Text))
mxSymbolListAttr symbol = do
    (cnt, ptr) <- checked $ mxSymbolListAttr_ symbol
    cnt <- pure $ fromIntegral cnt
    kvs <- VB.fromList <$> peekCStringArrayT (2 * cnt) ptr
    let kv_tuples = map (\i -> let p = i * 2
                                   q = p + 1
                               in (kvs VB.! p, kvs VB.! q))
                        [0..cnt-1]
        upd m (k, v) = case T.split (=='$') k of
                         [k1, k2] -> m & at k1 . non M.empty . at k2 ?~ v
                         _ -> error ("bad attr " ++ T.unpack k)
    return $ foldl' upd M.empty kv_tuples

{#
fun MXSymbolListAttrShallow as mxSymbolListAttrShallow_
    {
        `SymbolHandle',
        alloca- `MX_UINT' peek*,
        alloca- `Ptr CString' peek*
    } -> `CInt'
#}

mxSymbolListAttrShallow :: HasCallStack
                        => SymbolHandle
                        -> IO (HashMap Text Text)
mxSymbolListAttrShallow symbol = do
    (cnt, ptr) <- checked $ mxSymbolListAttrShallow_ symbol
    cnt <- pure $ fromIntegral cnt
    kvs <- VB.fromList <$> peekCStringArrayT (2 * fromIntegral cnt) ptr
    let kv_tuples = map (\i -> let p = i * 2
                                   q = p + 1
                               in (kvs VB.! p, kvs VB.! q))
                        [0..cnt-1]
    return $ M.fromList kv_tuples

{#
fun MXSymbolListArguments as mxSymbolListArguments_
    {
        `SymbolHandle',
        alloca- `MX_UINT' peek*,
        alloca- `Ptr CString' peek*
    } -> `CInt'
#}

mxSymbolListArguments :: HasCallStack
                      => SymbolHandle
                      -> IO [Text]
mxSymbolListArguments symbol = do
    (cnt, ptr) <- checked $ mxSymbolListArguments_ symbol
    peekCStringArrayT (fromIntegral cnt) ptr

{#
fun MXSymbolListOutputs as mxSymbolListOutputs_
    {
        `SymbolHandle',
        alloca- `MX_UINT' peek*,
        alloca- `Ptr CString' peek*
    } -> `CInt'
#}

mxSymbolListOutputs :: HasCallStack
                    => SymbolHandle
                    -> IO [Text]
mxSymbolListOutputs symbol = do
    (cnt, ptr) <- checked $ mxSymbolListOutputs_ symbol
    peekCStringArrayT (fromIntegral cnt) ptr

{#
fun MXSymbolGetInternals as mxSymbolGetInternals_
    {
        `SymbolHandle',
        alloca- `SymbolHandle' peekSymbolHandle*
    } -> `CInt'
#}

mxSymbolGetInternals :: HasCallStack
                     => SymbolHandle -> IO SymbolHandle
mxSymbolGetInternals = checked . mxSymbolGetInternals_

{#
fun MXSymbolGetChildren as mxSymbolGetChildren_
    {
        `SymbolHandle',
        alloca- `SymbolHandle' peekSymbolHandle*
    } -> `CInt'
#}

mxSymbolGetChildren :: HasCallStack
                    => SymbolHandle -> IO SymbolHandle
mxSymbolGetChildren = checked . mxSymbolGetChildren_

{#
fun MXSymbolGetOutput as mxSymbolGetOutput_
    {
        `SymbolHandle',
        `CUInt',
        alloca- `SymbolHandle' peekSymbolHandle*
    } -> `CInt'
#}

mxSymbolGetOutput :: HasCallStack
                  => SymbolHandle
                  -> Int
                  -> IO SymbolHandle
mxSymbolGetOutput symbol ind =
    checked $ mxSymbolGetOutput_ symbol (fromIntegral ind)

{#
fun MXSymbolListAuxiliaryStates as mxSymbolListAuxiliaryStates_
    {
        `SymbolHandle',
        alloca- `MX_UINT' peek*,
        alloca- `Ptr CString' peek*
    } -> `CInt'
#}
mxSymbolListAuxiliaryStates :: HasCallStack => SymbolHandle -> IO [Text]
mxSymbolListAuxiliaryStates symbol = do
    (cnt, ptr) <- checked $ mxSymbolListAuxiliaryStates_ symbol
    peekCStringArrayT (fromIntegral cnt) ptr

{#
fun MXSymbolCompose as mxSymbolCompose_
    {
        `SymbolHandle',
        withCStringT* `Text',
        `CUInt',
        id `Ptr CString',
        withSymbolHandleArray* `[SymbolHandle]'
    } -> `CInt'
#}

mxSymbolCompose :: HasCallStack
                => SymbolHandle
                -> Text
                -> Maybe [Text]
                -> [SymbolHandle]
                -> IO ()
mxSymbolCompose symbol name maybekeys args = do
    let len = fromIntegral $ length args
    case maybekeys of
      Nothing -> checked $ mxSymbolCompose_ symbol name len C2HSImp.nullPtr args
      Just keys -> withCStringArrayT keys $ \pkeys -> checked $ mxSymbolCompose_ symbol name len pkeys args

{#
fun MXSymbolInferShapeEx as mxSymbolInferShape_
    {
        `SymbolHandle',
        `CUInt',
        withCStringArrayT* `[Text]',
        withArray* `[CUInt]',
        withArray* `[CInt]',
        alloca- `CUInt' peek*,
        alloca- `Ptr CInt' peek*,
        alloca- `Ptr (Ptr CInt)' peek*,
        alloca- `CUInt' peek*,
        alloca- `Ptr CInt' peek*,
        alloca- `Ptr (Ptr CInt)' peek*,
        alloca- `CUInt' peek*,
        alloca- `Ptr CInt' peek*,
        alloca- `Ptr (Ptr CInt)' peek*,
        alloca- `CInt' peek*
    } -> `CInt'
#}

mxSymbolInferShape :: HasCallStack
                   => SymbolHandle
                   -> [Text]
                   -> [Int]
                   -> [Int]
                   -> IO ([[Int]], [[Int]], [[Int]], Bool)
mxSymbolInferShape symbol keys arg_ind arg_shape = do
    let num_args' = fromIntegral (length keys)
        arg_ind'  = map fromIntegral arg_ind
        arg_shape'= map fromIntegral arg_shape
    (inshape_size , inshape_ndim , inshape_data ,
     outshape_size, outshape_ndim, outshape_data,
     auxshape_size, auxshape_ndim, auxshape_data,
     complete) <- checked $ mxSymbolInferShape_ symbol num_args' keys arg_ind' arg_shape'
    let inshape_size' = fromIntegral inshape_size
    inshape_ndim'    <- map fromIntegral <$> peekArray inshape_size' inshape_ndim
    inshape_data'    <- peekArray inshape_size' inshape_data
    inshape_data_ret <- mapM (uncurry peekArrayAsIntegral) (zip inshape_ndim' inshape_data')
    let outshape_size'= fromIntegral outshape_size
    outshape_ndim'   <- map fromIntegral <$> peekArray outshape_size' outshape_ndim
    outshape_data'   <- peekArray outshape_size' outshape_data
    outshape_data_ret<- mapM (uncurry peekArrayAsIntegral) (zip outshape_ndim' outshape_data')
    let auxshape_size'= fromIntegral auxshape_size
    auxshape_ndim'   <- map fromIntegral <$> peekArray auxshape_size' auxshape_ndim
    auxshape_data'   <- peekArray auxshape_size' auxshape_data
    auxshape_data_ret<- mapM (uncurry peekArrayAsIntegral) (zip auxshape_ndim' auxshape_data')
    return (inshape_data_ret, outshape_data_ret, auxshape_data_ret, complete == 1)

{#
fun MXSymbolInferShapePartialEx as mxSymbolInferShapePartial_
    {
        `SymbolHandle',
        `CUInt',
        withCStringArrayT* `[Text]',
        withArray* `[CUInt]',
        withArray* `[CInt]',
        alloca- `CUInt' peek*,
        alloca- `Ptr CInt' peek*,
        alloca- `Ptr (Ptr CInt)' peek*,
        alloca- `CUInt' peek*,
        alloca- `Ptr CInt' peek*,
        alloca- `Ptr (Ptr CInt)' peek*,
        alloca- `CUInt' peek*,
        alloca- `Ptr CInt' peek*,
        alloca- `Ptr (Ptr CInt)' peek*,
        alloca- `CInt' peek*
    } -> `CInt'
#}

mxSymbolInferShapePartial :: HasCallStack
                          => SymbolHandle
                          -> [Text]
                          -> [Int]
                          -> [Int]
                          -> IO ([[Int]], [[Int]], [[Int]], Bool)
mxSymbolInferShapePartial symbol keys arg_ind arg_shape = do
    let num_args' = fromIntegral (length keys)
        arg_ind'  = map fromIntegral arg_ind
        arg_shape'= map fromIntegral arg_shape
    (inshape_size , inshape_ndim , inshape_data ,
     outshape_size, outshape_ndim, outshape_data,
     auxshape_size, auxshape_ndim, auxshape_data,
     complete) <- checked $ mxSymbolInferShapePartial_ symbol num_args' keys arg_ind' arg_shape'
    let inshape_size' = fromIntegral inshape_size
    inshape_ndim'    <- map fromIntegral <$> peekArray inshape_size' inshape_ndim
    inshape_data'    <- peekArray inshape_size' inshape_data
    inshape_data_ret <- mapM (uncurry peekArrayAsIntegral) (zip inshape_ndim' inshape_data')
    let outshape_size'= fromIntegral outshape_size
    outshape_ndim'   <- map fromIntegral <$> peekArray outshape_size' outshape_ndim
    outshape_data'   <- peekArray outshape_size' outshape_data
    outshape_data_ret<- mapM (uncurry peekArrayAsIntegral) (zip outshape_ndim' outshape_data')
    let auxshape_size'= fromIntegral auxshape_size
    auxshape_ndim'   <- map fromIntegral <$> peekArray auxshape_size' auxshape_ndim
    auxshape_data'   <- peekArray auxshape_size' auxshape_data
    auxshape_data_ret<- mapM (uncurry peekArrayAsIntegral) (zip auxshape_ndim' auxshape_data')
    return (inshape_data_ret, outshape_data_ret, auxshape_data_ret, complete == 1)

peekArrayAsIntegral cnt ptr = map fromIntegral <$> peekArray (fromIntegral cnt) ptr

{#
fun MXSymbolInferType as mxSymbolInferType_
    {
        `SymbolHandle',
        `CUInt',
        withCStringArrayT* `[Text]',
        withArray* `[CInt]',
        alloca- `MX_UINT' peek*,
        alloca- `Ptr CInt' peek*,
        alloca- `MX_UINT' peek*,
        alloca- `Ptr CInt' peek*,
        alloca- `MX_UINT' peek*,
        alloca- `Ptr CInt' peek*,
        alloca- `CInt' peek*
    } -> `CInt'
#}

mxSymbolInferType :: HasCallStack
                  => SymbolHandle
                  -> [Text]
                  -> [Int]
                  -> IO (Maybe ([Int], [Int], [Int]))
mxSymbolInferType symbol keys arg_type = do
    let num_args = fromIntegral (length keys)
        arg_type'= map fromIntegral arg_type
    (inshape_size , inshape_data ,
     outshape_size, outshape_data,
     auxshape_size, auxshape_data,
     succ) <- checked $ mxSymbolInferType_ symbol num_args keys arg_type'
    if succ == 0 then do
        inshape_data_ret  <- peekArrayAsIntegral (fromIntegral inshape_size)  inshape_data
        outshape_data_ret <- peekArrayAsIntegral (fromIntegral outshape_size) outshape_data
        auxshape_data_ret <- peekArrayAsIntegral (fromIntegral auxshape_size) auxshape_data
        return $ Just (inshape_data_ret, outshape_data_ret, auxshape_data_ret)
    else
        return Nothing

{#
fun MXSymbolSaveToFile as mxSymbolSaveToFile_
    {
        `SymbolHandle',
        withCStringT* `Text'
    } -> `CInt'
#}

mxSymbolSaveToFile :: HasCallStack => Text -> SymbolHandle -> IO ()
mxSymbolSaveToFile filename sym = do
    checked $ mxSymbolSaveToFile_ sym filename

data MXCallbackList = MXCallbackList Int (Ptr (FunPtr (IO CInt))) (Ptr (Ptr ()))

instance Storable MXCallbackList where
    sizeOf _ = {#sizeof MXCallbackList#}
    alignment _ = {#alignof MXCallbackList#}
    peek ptr = do
        n <- {#get MXCallbackList.num_callbacks#} ptr
        callbacks <- {#get MXCallbackList.callbacks#} ptr
        contexts  <- {#get MXCallbackList.contexts#} ptr
        return (MXCallbackList (fromIntegral n) callbacks contexts)
    poke ptr (MXCallbackList n callbacks contexts) = do
        {#set MXCallbackList.num_callbacks#} ptr (fromIntegral n)
        {#set MXCallbackList.callbacks#} ptr callbacks
        {#set MXCallbackList.contexts#}  ptr contexts

type CustomOpPropCreator          = CString -> CInt -> Ptr CString -> Ptr CString -> Ptr () -> IO CInt
type CustomOpFBFunc               = CInt -> Ptr (Ptr ()) -> Ptr CInt -> Ptr CInt -> CInt -> Ptr () -> IO CInt
type CustomOpDelFunc              = Ptr () -> IO CInt
type CustomOpListFunc             = Ptr (Ptr CString) -> Ptr () -> IO CInt
type CustomOpInferShapeFunc       = CInt -> Ptr CInt -> Ptr (Ptr CInt) -> Ptr () -> IO CInt
type CustomOpInferStorageTypeFunc = CInt -> Ptr CInt -> Ptr () -> IO CInt
type CustomOpBackwardInferStorageTypeFunc = CInt -> Ptr CInt -> Ptr CInt -> Ptr () -> IO CInt
type CustomOpInferTypeFunc        = CInt -> Ptr CInt -> Ptr () -> IO CInt
type CustomOpBwdDepFunc           = Ptr CInt -> Ptr CInt -> Ptr CInt -> Ptr CInt -> Ptr (Ptr CInt) -> Ptr () -> IO CInt
type CustomOpCreateFunc           = CString -> CInt -> Ptr (Ptr CUInt) -> Ptr CInt -> Ptr CInt -> Ptr MXCallbackList -> Ptr () -> IO CInt
type CustomFunctionBwdFunc        = CInt -> Ptr (Ptr NDArrayHandle) -> Ptr CInt -> Ptr CInt -> CInt -> Ptr () -> IO CInt
type CustomFunctionDelFunc        = Ptr () -> IO CInt

foreign import ccall "wrapper" mkCustomOpPropCreator    :: CustomOpPropCreator -> IO (FunPtr CustomOpPropCreator)
foreign import ccall "wrapper" mkCustomOpFBFunc         :: CustomOpFBFunc -> IO (FunPtr CustomOpFBFunc)
foreign import ccall "wrapper" mkCustomOpDelFunc        :: CustomOpDelFunc -> IO (FunPtr CustomOpDelFunc)
foreign import ccall "wrapper" mkCustomOpListFunc       :: CustomOpListFunc -> IO (FunPtr CustomOpListFunc)
foreign import ccall "wrapper" mkCustomOpInferShapeFunc :: CustomOpInferShapeFunc -> IO (FunPtr CustomOpInferShapeFunc)
foreign import ccall "wrapper" mkCustomOpInferStorageTypeFunc :: CustomOpInferStorageTypeFunc -> IO (FunPtr CustomOpInferStorageTypeFunc)
foreign import ccall "wrapper" mkCustomOpBackwardInferStorageTypeFunc :: CustomOpBackwardInferStorageTypeFunc -> IO (FunPtr CustomOpBackwardInferStorageTypeFunc)
foreign import ccall "wrapper" mkCustomOpInferTypeFunc  :: CustomOpInferTypeFunc -> IO (FunPtr CustomOpInferTypeFunc)
foreign import ccall "wrapper" mkCustomOpBwdDepFunc     :: CustomOpBwdDepFunc -> IO (FunPtr CustomOpBwdDepFunc)
foreign import ccall "wrapper" mkCustomOpCreateFunc     :: CustomOpCreateFunc -> IO (FunPtr CustomOpCreateFunc)
foreign import ccall "wrapper" mkCustomFunctionBwdFunc  :: CustomFunctionBwdFunc -> IO (FunPtr CustomFunctionBwdFunc)
foreign import ccall "wrapper" mkCustomFunctionDelFunc  :: CustomFunctionDelFunc -> IO (FunPtr CustomFunctionDelFunc)

{#
fun MXCustomOpRegister as mxCustomOpRegister_
    {
        withCStringT* `Text',
        id `FunPtr CustomOpPropCreator'
    } -> `CInt'
#}

mxCustomOpRegister :: Text -> FunPtr CustomOpPropCreator -> IO ()
mxCustomOpRegister op cr = checked $ mxCustomOpRegister_ op cr

-- {#
-- fun MXSymbolGetInputSymbols as mxSymbolGetInputSymbols_
--     {
--         `SymbolHandle',
--         alloca- `Ptr SymbolHandlePtr' peek*,
--         alloca- `CInt' peek*
--     } -> `CInt'
-- #}
--
{#
fun MXSymbolGetNumOutputs as mxSymbolGetNumOutputs_
    {
        `SymbolHandle',
        alloca- `MX_UINT' peek*
    } -> `CInt'
#}

mxSymbolGetNumOutputs :: SymbolHandle -> IO Int
mxSymbolGetNumOutputs symbol = do
    cnt <- checked $ mxSymbolGetNumOutputs_ symbol
    return $ fromIntegral (cnt :: MX_UINT)
--
-- {#
-- fun MXQuantizeSymbol as mxQuantizeSymbol_
--     {
--         `SymbolHandle',
--         alloca- `Ptr SymbolHandlePtr' peekSymbolHandle*,
--         `MX_UINT',
--         withSymbolHandleArray* `[SymbolHandle]',
--         `MX_UINT',
--         withStringArray* `[String]',
--         `String'
--     } -> `CInt'
-- #}
