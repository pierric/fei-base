{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.Base.Raw.Executor where

import Foreign.Marshal (alloca, withArray, peekArray)
import Foreign.Storable (Storable(..))
import Foreign.ForeignPtr (newForeignPtr, newForeignPtr_, touchForeignPtr)
import Foreign.ForeignPtr.Unsafe (unsafeForeignPtrToPtr)
import C2HS.C.Extra.Marshal (withIntegralArray, peekIntegralArray, peekString, peekStringArray)
import GHC.Generics (Generic)
import Control.Monad ((>=>))
import Data.Maybe (fromMaybe)

{# import MXNet.Base.Raw.Common #}
{# import MXNet.Base.Raw.NDArray #}
{# import MXNet.Base.Raw.Symbol #}

#include <mxnet/c_api.h>

{# typedef size_t CSize#}
{# default in `CSize' [size_t] fromIntegral #}
{# typedef mx_uint MX_UINT#}
{# default in `MX_UINT' [mx_uint] id #}

{#
pointer ExecutorHandle foreign finalizer MXExecutorFree as mxExecutorFree newtype 
#}

deriving instance Generic ExecutorHandle

type ExecutorHandlePtr = Ptr ExecutorHandle

newExecutorHandle :: ExecutorHandlePtr -> IO ExecutorHandle
newExecutorHandle = newForeignPtr mxExecutorFree >=> return . ExecutorHandle

peekExecutorHandle :: Ptr ExecutorHandlePtr -> IO ExecutorHandle
peekExecutorHandle = peek >=> newExecutorHandle

withExecutorHandleArray :: [ExecutorHandle] -> (Ptr ExecutorHandlePtr -> IO r) -> IO r
withExecutorHandleArray array io = do
    let unExecutorHandle (ExecutorHandle fptr) = fptr
    r <- withArray (map (unsafeForeignPtrToPtr . unExecutorHandle) array) io
    mapM_ (touchForeignPtr . unExecutorHandle) array
    return r

{#
fun MXExecutorPrint as mxExecutorPrint_
    {
        `ExecutorHandle',
        alloca- `String' peekString*
    } -> `CInt'
#}

mxExecutorPrint :: ExecutorHandle -> IO String
mxExecutorPrint = checked . mxExecutorPrint_

{#
fun MXExecutorForward as mxExecutorForward_
    {
        `ExecutorHandle',
        `CInt'
    } -> `CInt'
#}

mxExecutorForward :: ExecutorHandle -> Bool -> IO ()
mxExecutorForward exec train = checked $ mxExecutorForward_ exec is_train
  where
    is_train = if train then 1 else 0

{#
fun MXExecutorBackward as mxExecutorBackward_
    {
        `ExecutorHandle',
        `MX_UINT',
        withNDArrayHandleArray* `[NDArrayHandle]'
    } -> `CInt'
#}

mxExecutorBackward :: ExecutorHandle -> [NDArrayHandle] -> IO ()
mxExecutorBackward exec head_grads = checked $ mxExecutorBackward_ exec cnt head_grads
  where
    cnt = fromIntegral $ length head_grads

{#
fun MXExecutorBackwardEx as mxExecutorBackwardEx_
    {
        `ExecutorHandle',
        `MX_UINT',
        withNDArrayHandleArray* `[NDArrayHandle]',
        `CInt'
    } -> `CInt'
#}

mxExecutorBackwardEx :: ExecutorHandle -> [NDArrayHandle] -> Bool -> IO ()
mxExecutorBackwardEx exec head_grads train = checked $ mxExecutorBackwardEx_ exec cnt head_grads is_train
  where
    cnt = fromIntegral $ length head_grads
    is_train = if train then 1 else 0

{#
fun MXExecutorOutputs as mxExecutorOutputs_
    {
        `ExecutorHandle',
        alloca- `MX_UINT' peek*,
        alloca- `Ptr NDArrayHandlePtr' peek*
    } -> `CInt'
#}

mxExecutorOutputs :: ExecutorHandle -> IO [NDArrayHandle]
mxExecutorOutputs handle = do
    (cnt, ptr) <- checked $ mxExecutorOutputs_ handle
    handle_ptrs <- peekArray (fromIntegral cnt) ptr
    mapM newNDArrayHandle handle_ptrs

{#
fun MXExecutorBind as mxExecutorBind_
    {
        `SymbolHandle',
        `CInt',
        `CInt',
        `MX_UINT',
        withNDArrayHandleArray* `[NDArrayHandle]',
        withNDArrayHandleArray* `[NDArrayHandle]',
        withArray* `[MX_UINT]',
        `MX_UINT',
        withNDArrayHandleArray* `[NDArrayHandle]',
        alloca- `ExecutorHandle' peekExecutorHandle*
    } -> `CInt'
#}

makeNullNDArrayHandle = NDArrayHandle <$> newForeignPtr_ C2HSImp.nullPtr

mxExecutorBind :: SymbolHandle
               -> Int 
               -> Int
               -> [NDArrayHandle]
               -> [Maybe NDArrayHandle]
               -> [Int]
               -> [NDArrayHandle]
               -> IO ExecutorHandle
mxExecutorBind symbol devtype devid in_args arg_grad_store grad_req_type aux_states = do
    nullNDArrayHandle <- makeNullNDArrayHandle
    let arg_grad_store_ = map (fromMaybe nullNDArrayHandle) arg_grad_store
    checked $ mxExecutorBind_ symbol devtype_ devid_ cnt_args in_args arg_grad_store_ grad_req_type_ cnt_auxs aux_states
  where
    devtype_ = fromIntegral devtype
    devid_   = fromIntegral devid
    cnt_args = fromIntegral $ length arg_grad_store
    cnt_auxs = fromIntegral $ length aux_states
    grad_req_type_ = map fromIntegral grad_req_type

{#
fun MXExecutorBindX as mxExecutorBindX_
    {
        `SymbolHandle',
        `CInt',
        `CInt',
        `MX_UINT',
        withStringArray* `[String]',
        withArray* `[CInt]',
        withArray* `[CInt]',
        `MX_UINT',
        withNDArrayHandleArray* `[NDArrayHandle]',
        withNDArrayHandleArray* `[NDArrayHandle]',
        withArray* `[MX_UINT]',
        `MX_UINT',
        withNDArrayHandleArray* `[NDArrayHandle]',
        alloca- `ExecutorHandle' peekExecutorHandle*
    } -> `CInt'
#}

mxExecutorBindX :: SymbolHandle
                -> Int 
                -> Int
                -> [String]
                -> [Int]
                -> [Int]
                -> [NDArrayHandle]
                -> [Maybe NDArrayHandle]
                -> [Int]
                -> [NDArrayHandle]
                -> IO ExecutorHandle
mxExecutorBindX symbol devtype devid map_keys map_dev_types map_dev_ids in_args arg_grad_store grad_req_type aux_states = do
    nullNDArrayHandle <- makeNullNDArrayHandle
    let arg_grad_store_ = map (fromMaybe nullNDArrayHandle) arg_grad_store
    checked $ mxExecutorBindX_ symbol devtype_ devid_ cnt_maps map_keys map_dev_types_ map_dev_ids_ cnt_args in_args arg_grad_store_ grad_req_type_ cnt_auxs aux_states
  where
    devtype_ = fromIntegral devtype
    devid_   = fromIntegral devid
    cnt_maps = fromIntegral $ length map_keys
    map_dev_types_ = map fromIntegral map_dev_types
    map_dev_ids_ = map fromIntegral map_dev_ids
    cnt_args = fromIntegral $ length arg_grad_store
    cnt_auxs = fromIntegral $ length aux_states
    grad_req_type_ = map fromIntegral grad_req_type

{#
fun MXExecutorBindEX as mxExecutorBindEX_
    {
        `SymbolHandle',
        `CInt',
        `CInt',
        `MX_UINT',
        withStringArray* `[String]',
        withArray* `[CInt]',
        withArray* `[CInt]',
        `MX_UINT',
        withNDArrayHandleArray* `[NDArrayHandle]',
        withNDArrayHandleArray* `[NDArrayHandle]',
        withArray* `[MX_UINT]',
        `MX_UINT',
        withNDArrayHandleArray* `[NDArrayHandle]',
        `ExecutorHandle',
        alloca- `ExecutorHandle' peekExecutorHandle*
    } -> `CInt'
#}

mxExecutorBindEX :: SymbolHandle
                 -> Int 
                 -> Int
                 -> [String]
                 -> [Int]
                 -> [Int]
                 -> [NDArrayHandle]
                 -> [Maybe NDArrayHandle]
                 -> [Int]
                 -> [NDArrayHandle]
                 -> ExecutorHandle
                 -> IO ExecutorHandle
mxExecutorBindEX symbol devtype devid map_keys map_dev_types map_dev_ids in_args arg_grad_store grad_req_type aux_states shared_exec = do
    nullNDArrayHandle <- makeNullNDArrayHandle
    let arg_grad_store_ = map (fromMaybe nullNDArrayHandle) arg_grad_store
    checked $ mxExecutorBindEX_ symbol devtype_ devid_ cnt_maps map_keys map_dev_types_ map_dev_ids_ cnt_args in_args arg_grad_store_ grad_req_type_ cnt_auxs aux_states shared_exec
  where
    devtype_ = fromIntegral devtype
    devid_   = fromIntegral devid
    cnt_maps = fromIntegral $ length map_keys
    map_dev_types_ = map fromIntegral map_dev_types
    map_dev_ids_ = map fromIntegral map_dev_ids
    cnt_args = fromIntegral $ length arg_grad_store
    cnt_auxs = fromIntegral $ length aux_states
    grad_req_type_ = map fromIntegral grad_req_type

{#
fun MXExecutorSimpleBind as mxExecutorSimpleBind_
    {
        `SymbolHandle',
        `CInt',
        `CInt',
        `MX_UINT',
        withStringArray* `[String]',
        withArray* `[CInt]',
        withArray* `[CInt]',
        `MX_UINT',
        withStringArray* `[String]',
        withStringArray* `[String]',
        `MX_UINT',
        withStringArray* `[String]',
        withArray* `[MX_UINT]',
        withArray* `[MX_UINT]',
        `MX_UINT',
        withStringArray* `[String]',
        withArray* `[CInt]',
        `MX_UINT',
        withStringArray* `[String]',
        withArray* `[CInt]',
        `MX_UINT',
        withStringArray* `[String]',
        id `Ptr CInt',                          -- [in/out] shared_buffer_len
        id `Ptr (Ptr CChar)',                   -- [in, optional] shared_buffer_name_list
        id `Ptr NDArrayHandlePtr',              -- [in, optional] shared_buffer_handle_list
        alloca- `Ptr (Ptr CChar)' peek*,        -- [out] updated_shared_buffer_name_list
        alloca- `Ptr NDArrayHandlePtr' peek*,   -- [out] updated_shared_buffer_handle_list
        alloca- `MX_UINT' peek*,
        alloca- `Ptr NDArrayHandlePtr' peek*,
        alloca- `Ptr NDArrayHandlePtr' peek*,
        alloca- `MX_UINT' peek*,
        alloca- `Ptr NDArrayHandlePtr' peek*,
        `ExecutorHandle',
        alloca- `ExecutorHandle' peekExecutorHandle*        
    } -> `CInt'
#}

mxExecutorSimpleBind :: SymbolHandle
                     -> Int -> Int                        -- device
                     -> [String] -> [Int] -> [Int]        -- g2c
                     -> [String] -> [String]              -- provided_grad_req_list
                     -> [String] -> [Int] -> [Int]        -- provided_arg_shapes
                     -> [String] -> [Int]                 -- provided_arg_dtypes
                     -> [String] -> [Int]                 -- provided_arg_stypes
                     -> [String]                          -- shared_arg_names
                     -> Maybe ([String], [NDArrayHandle]) -- shared_buffer
                     -> ExecutorHandle                    -- shared_exec_handle
                     -> IO (Maybe ([String], [NDArrayHandle]), -- updated_shared_buffer
                            [NDArrayHandle],                   -- arg_array
                            [NDArrayHandle],                   -- grad_array
                            [NDArrayHandle],                   -- aux_array
                            ExecutorHandle)
mxExecutorSimpleBind symbol 
                     devtype devid 
                     g2c_keys g2c_dev_types g2c_dev_ids 
                     provided_grad_req_names 
                     provided_grad_req_types 
                     provided_arg_shape_names provided_arg_shape_data provided_arg_shape_idx
                     provided_arg_dtype_names provided_arg_dtypes 
                     provided_arg_stype_names provided_arg_stypes 
                     shared_arg_name_list 
                     shared_buffer 
                     shared_exec_handle =
    case shared_buffer of 

        Nothing -> alloca (\ptr_shared_buffer_len -> do
            poke ptr_shared_buffer_len (-1)
            (_, _, num_in_args, in_args, arg_grads, num_aux_states, aux_states, out) <- checked $ mxExecutorSimpleBind_ 
                        symbol devtype_ devid_ 
                        cnt_g2c g2c_keys g2c_dev_types_ g2c_dev_ids_ 
                        cnt_provided_grad_req_list provided_grad_req_names provided_grad_req_types 
                        cnt_provided_arg_shapes provided_arg_shape_names provided_arg_shape_data_ provided_arg_shape_idx_
                        cnt_provided_arg_dtypes provided_arg_dtype_names provided_arg_dtypes_
                        cnt_provided_arg_stypes provided_arg_stype_names provided_arg_stypes_
                        cnt_shared_arg_names shared_arg_name_list
                        ptr_shared_buffer_len C2HSImp.nullPtr C2HSImp.nullPtr
                        shared_exec_handle
            arg_array  <- peekArray (fromIntegral num_in_args) in_args   >>= mapM newNDArrayHandle
            grad_array <- peekArray (fromIntegral num_in_args) arg_grads >>= mapM newNDArrayHandle
            aux_array  <- peekArray (fromIntegral num_aux_states) aux_states >>= mapM newNDArrayHandle
            return (Nothing, arg_array, grad_array, aux_array, out))

        Just (shared_buffer_name_list, shared_buffer_handle_list) -> alloca (\ptr_shared_buffer_len -> do
            poke ptr_shared_buffer_len (fromIntegral $ length shared_buffer_name_list)
            withStringArray shared_buffer_name_list (\ptr_shared_buffer_name_list -> 
                withNDArrayHandleArray shared_buffer_handle_list (\ptr_shared_buffer_handle_list -> do
                    (ptr_updated_shared_buffer_name_list, ptr_updated_shared_buffer_handle_list, num_in_args, in_args, arg_grads, num_aux_states, aux_states, out) <- checked $ mxExecutorSimpleBind_ 
                                symbol devtype_ devid_ 
                                cnt_g2c g2c_keys g2c_dev_types_ g2c_dev_ids_ 
                                cnt_provided_grad_req_list provided_grad_req_names provided_grad_req_types 
                                cnt_provided_arg_shapes provided_arg_shape_names provided_arg_shape_data_ provided_arg_shape_idx_
                                cnt_provided_arg_dtypes provided_arg_dtype_names provided_arg_dtypes_
                                cnt_provided_arg_stypes provided_arg_stype_names provided_arg_stypes_
                                cnt_shared_arg_names shared_arg_name_list
                                ptr_shared_buffer_len ptr_shared_buffer_name_list ptr_shared_buffer_handle_list
                                shared_exec_handle
                    arg_array  <- peekArray (fromIntegral num_in_args) in_args   >>= mapM newNDArrayHandle
                    grad_array <- peekArray (fromIntegral num_in_args) arg_grads >>= mapM newNDArrayHandle
                    aux_array  <- peekArray (fromIntegral num_aux_states) aux_states >>= mapM newNDArrayHandle
                    update_shared_buffer_len <- fromIntegral <$> peek ptr_shared_buffer_len
                    updated_shared_buffer_name_list <- peekStringArray update_shared_buffer_len ptr_updated_shared_buffer_name_list
                    updated_shared_buffer_handle_list <- peekArray update_shared_buffer_len ptr_updated_shared_buffer_handle_list >>= mapM newNDArrayHandle
                    return (Just (updated_shared_buffer_name_list, updated_shared_buffer_handle_list), arg_array, grad_array, aux_array, out)
                )))

  where
    devtype_ = fromIntegral devtype
    devid_ = fromIntegral devid
    cnt_g2c = fromIntegral $ length g2c_keys
    g2c_dev_types_ = map fromIntegral g2c_dev_types
    g2c_dev_ids_ = map fromIntegral g2c_dev_ids
    cnt_provided_grad_req_list = fromIntegral $ length provided_grad_req_names
    cnt_provided_arg_shapes = fromIntegral $ length provided_arg_shape_names
    provided_arg_shape_data_ = map fromIntegral provided_arg_shape_data
    provided_arg_shape_idx_  = map fromIntegral provided_arg_shape_idx
    cnt_provided_arg_dtypes = fromIntegral $ length provided_arg_dtype_names
    provided_arg_dtypes_ = map fromIntegral provided_arg_dtypes
    cnt_provided_arg_stypes = fromIntegral $ length provided_arg_stype_names
    provided_arg_stypes_ = map fromIntegral $ provided_arg_stypes
    cnt_shared_arg_names = fromIntegral $ length shared_arg_name_list

-- {#
-- fun MXExecutorReshape as mxExecutorReshape_
--     {
--         `CInt',
--         `CInt',
--         `CInt',
--         `CInt',
--         `MX_UINT',
--         withStringArray* `[String]',
--         withArray* `[CInt]',
--         withArray* `[CInt]',
--         `MX_UINT',
--         withStringArray* `[String]',
--         withArray* `[MX_UINT]',
--         withArray* `[MX_UINT]',
--         alloca- `MX_UINT' peek*,
--         alloca- `Ptr NDArrayHandlePtr' peek*,
--         alloca- `Ptr NDArrayHandlePtr' peek*,
--         alloca- `MX_UINT' peek*,
--         alloca- `Ptr NDArrayHandlePtr' peek*,
--         `ExecutorHandle',
--         alloca- `ExecutorHandle' peekExecutorHandle*
--     } -> `CInt'
-- #}

-- {# 
-- fun MXExecutorGetOptimizedSymbol as mxExecutorGetOptimizedSymbol_
--     {
--         `ExecutorHandle',
--         alloca- `SymbolHandle' peekSymbolHandle*
--     } -> `CInt'
-- #}