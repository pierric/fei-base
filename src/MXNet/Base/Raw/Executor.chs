{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.Base.Raw.Executor where

import RIO
import Foreign.Marshal (alloca, withArray, peekArray)
import Foreign.Storable (Storable(..))
import Foreign.ForeignPtr (newForeignPtr_, touchForeignPtr)
import Foreign.ForeignPtr.Unsafe (unsafeForeignPtrToPtr)
import Foreign.Ptr
import Foreign.C.Types
import Control.DeepSeq (NFData(..), rwhnf)

{# import MXNet.Base.Raw.Common #}
{# import MXNet.Base.Raw.NDArray #}
{# import MXNet.Base.Raw.Symbol #}

#include <mxnet/c_api.h>

{# typedef size_t CSize#}
{# default in `CSize' [size_t] fromIntegral #}
{# typedef mx_uint MX_UINT#}
{# default in `MX_UINT' [mx_uint] id #}

{#
pointer ExecutorHandle foreign newtype
#}

deriving instance Generic ExecutorHandle
deriving instance Show ExecutorHandle
instance NFData ExecutorHandle where
    rnf = rwhnf

type ExecutorHandlePtr = Ptr ExecutorHandle

touchExecutorHandle :: ExecutorHandle -> IO ()
touchExecutorHandle (ExecutorHandle fptr) = touchForeignPtr fptr

newExecutorHandle :: ExecutorHandlePtr -> IO ExecutorHandle
newExecutorHandle ptr = newForeignPtr_ ptr >>= return . ExecutorHandle

peekExecutorHandle :: Ptr ExecutorHandlePtr -> IO ExecutorHandle
peekExecutorHandle = peek >=> newExecutorHandle

withExecutorHandleArray :: HasCallStack
                        => [ExecutorHandle] -> (Ptr ExecutorHandlePtr -> IO r) -> IO r
withExecutorHandleArray array io = do
    let unExecutorHandle (ExecutorHandle fptr) = fptr
    r <- withArray (map (unsafeForeignPtrToPtr . unExecutorHandle) array) io
    mapM_ (touchForeignPtr . unExecutorHandle) array
    return r

{#
fun MXExecutorFree as mxExecutorFree_
    {
        id `ExecutorHandlePtr'
    } -> `CInt'
#}

mxExecutorFree :: HasCallStack => ExecutorHandlePtr -> IO ()
mxExecutorFree = checked . mxExecutorFree_

{#
fun MXExecutorPrint as mxExecutorPrint_
    {
        `ExecutorHandle',
        alloca- `Text' peekCStringPtrT*
    } -> `CInt'
#}

mxExecutorPrint :: HasCallStack => ExecutorHandle -> IO Text
mxExecutorPrint = fmap unWrapText . checked . fmap (second WrapText) . mxExecutorPrint_

{#
fun MXExecutorForward as mxExecutorForward_
    {
        `ExecutorHandle',
        `CInt'
    } -> `CInt'
#}

mxExecutorForward :: HasCallStack => ExecutorHandle -> Bool -> IO ()
mxExecutorForward exec train = checked $ mxExecutorForward_ exec is_train
  where
    is_train = if train then 1 else 0

{#
fun MXExecutorBackward as mxExecutorBackward_
    {
        `ExecutorHandle',
        `CUInt',
        withNDArrayHandleArray* `[NDArrayHandle]'
    } -> `CInt'
#}

mxExecutorBackward :: HasCallStack => ExecutorHandle -> [NDArrayHandle] -> IO ()
mxExecutorBackward exec head_grads = checked $ mxExecutorBackward_ exec cnt head_grads
  where
    cnt = fromIntegral $ length head_grads

{#
fun MXExecutorBackwardEx as mxExecutorBackwardEx_
    {
        `ExecutorHandle',
        `CUInt',
        withNDArrayHandleArray* `[NDArrayHandle]',
        `CInt'
    } -> `CInt'
#}

mxExecutorBackwardEx :: HasCallStack => ExecutorHandle -> [NDArrayHandle] -> Bool -> IO ()
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

mxExecutorOutputs :: HasCallStack => ExecutorHandle -> IO [NDArrayHandle]
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
        `CUInt',
        withNDArrayHandleArray* `[NDArrayHandle]',
        withNDArrayHandleArray* `[NDArrayHandle]',
        withArray* `[CUInt]',
        `CUInt',
        withNDArrayHandleArray* `[NDArrayHandle]',
        alloca- `ExecutorHandle' peekExecutorHandle*
    } -> `CInt'
#}

makeNullExecutorHandle = ExecutorHandle <$> newForeignPtr_ C2HSImp.nullPtr

mxExecutorBind :: HasCallStack
               => SymbolHandle
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
        `CUInt',
        withCStringArrayT* `[Text]',
        withArray* `[CInt]',
        withArray* `[CInt]',
        `CUInt',
        withNDArrayHandleArray* `[NDArrayHandle]',
        withNDArrayHandleArray* `[NDArrayHandle]',
        withArray* `[CUInt]',
        `CUInt',
        withNDArrayHandleArray* `[NDArrayHandle]',
        alloca- `ExecutorHandle' peekExecutorHandle*
    } -> `CInt'
#}

mxExecutorBindX :: HasCallStack
                => SymbolHandle
                -> Int
                -> Int
                -> [Text]
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
        `CUInt',
        withCStringArrayT* `[Text]',
        withArray* `[CInt]',
        withArray* `[CInt]',
        `CUInt',
        withNDArrayHandleArray* `[NDArrayHandle]',
        withNDArrayHandleArray* `[NDArrayHandle]',
        withArray* `[CUInt]',
        `CUInt',
        withNDArrayHandleArray* `[NDArrayHandle]',
        `ExecutorHandle',
        alloca- `ExecutorHandle' peekExecutorHandle*
    } -> `CInt'
#}

mxExecutorBindEX :: HasCallStack
                 => SymbolHandle
                 -> Int
                 -> Int
                 -> [Text]
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
fun MXExecutorSimpleBindEx as mxExecutorSimpleBindEx_
    {
        `SymbolHandle',
        `CInt',
        `CInt',
        `CUInt',                        -- num_g2c_keys
        withCStringArrayT* `[Text]',    -- g2c_keys
        withArray* `[CInt]',            -- g2c_dev_types
        withArray* `[CInt]',            -- g2c_dev_ids
        `CUInt',                        -- provided_grad_req_list_len
        withCStringArrayT* `[Text]',    -- provided_grad_req_names
        withCStringArrayT* `[Text]',    -- provided_grad_req_types
        `CUInt',                        -- num_provided_arg_shapes
        withCStringArrayT* `[Text]',    -- provided_arg_shape_names
        withArray* `[CInt]',            -- provided_arg_shape_data
        withArray* `[CUInt]',           -- provided_arg_shape_idx
        `CUInt',                        -- num_provided_arg_dtypes
        withCStringArrayT* `[Text]',    -- provideded_arg_dtype_names
        withArray* `[CInt]',            -- provideded_arg_dtypes
        `CUInt',                        -- num_provided_arg_stypes
        withCStringArrayT* `[Text]',    -- provideded_arg_stype_names
        withArray* `[CInt]',            -- provideded_arg_stypes
        `CUInt',                        -- num_shared_arg_names
        withCStringArrayT* `[Text]',    -- shared_arg_name_list
        id `Ptr CInt',                          -- [in/out] shared_buffer_len
        id `Ptr (Ptr CChar)',                   -- [in, optional] shared_buffer_name_list
        id `Ptr NDArrayHandlePtr',              -- [in, optional] shared_buffer_handle_list
        alloca- `Ptr (Ptr CChar)' peek*,        -- [out] updated_shared_buffer_name_list
        alloca- `Ptr NDArrayHandlePtr' peek*,   -- [out] updated_shared_buffer_handle_list
        alloc0- `CUInt' peek*,                  -- num_in_args
        alloca- `Ptr NDArrayHandlePtr' peek*,   -- in_args
        alloca- `Ptr NDArrayHandlePtr' peek*,   -- arg_grads
        alloc0- `CUInt' peek*,                  -- num_aux_stats
        alloca- `Ptr NDArrayHandlePtr' peek*,   -- aux_stats
        `ExecutorHandle',                       -- shared_exec_handle
        alloca- `ExecutorHandle' peekExecutorHandle*
    } -> `CInt'
#}

mxExecutorSimpleBindEx :: HasCallStack
                       => SymbolHandle
                       -> Int -> Int                        -- device
                       -> [Text] -> [Int] -> [Int]          -- g2c
                       -> [Text] -> [Text]                  -- provided_grad_req_list
                       -> [Text] -> [Int] -> [Int]          -- provided_arg_shapes
                       -> [Text] -> [Int]                   -- provided_arg_dtypes
                       -> [Text] -> [Int]                   -- provided_arg_stypes
                       -> [Text]                            -- shared_arg_names
                       -> Maybe ([Text], [NDArrayHandle])   -- shared_buffer
                       -> Maybe ExecutorHandle              -- shared_exec_handle
                       -> IO (Maybe ([Text], [NDArrayHandle]),   -- updated_shared_buffer
                              [NDArrayHandle],                   -- arg_array
                              [Maybe NDArrayHandle],             -- grad_array
                              [NDArrayHandle],                   -- aux_array
                              ExecutorHandle)
mxExecutorSimpleBindEx symbol
                       devtype devid
                       g2c_keys g2c_dev_types g2c_dev_ids
                       provided_grad_req_names
                       provided_grad_req_types
                       provided_arg_shape_names provided_arg_shape_data provided_arg_shape_idx
                       provided_arg_dtype_names provided_arg_dtypes
                       provided_arg_stype_names provided_arg_stypes
                       shared_arg_name_list
                       shared_buffer
                       shared_exec_handle = do

    nullExecutorHandle <- makeNullExecutorHandle

    case shared_buffer of

        Nothing -> alloca (\ptr_shared_buffer_len -> do
            poke ptr_shared_buffer_len (-1)
            (_, _, num_in_args, in_args, arg_grads, num_aux_states, aux_states, out)
                <- checked $ mxExecutorSimpleBindEx_
                        symbol devtype_ devid_
                        cnt_g2c g2c_keys g2c_dev_types_ g2c_dev_ids_
                        cnt_provided_grad_req_list provided_grad_req_names provided_grad_req_types
                        cnt_provided_arg_shapes provided_arg_shape_names provided_arg_shape_data_ provided_arg_shape_idx_
                        cnt_provided_arg_dtypes provided_arg_dtype_names provided_arg_dtypes_
                        cnt_provided_arg_stypes provided_arg_stype_names provided_arg_stypes_
                        cnt_shared_arg_names shared_arg_name_list
                        ptr_shared_buffer_len C2HSImp.nullPtr C2HSImp.nullPtr
                        (fromMaybe nullExecutorHandle shared_exec_handle)
            arg_array  <- peekArray (fromIntegral num_in_args) in_args   >>= mapM newNDArrayHandle
            grad_array <- peekArray (fromIntegral num_in_args) arg_grads
            grad_array <- forM grad_array $ \ptr ->
                            if ptr == C2HSImp.nullPtr
                            then return Nothing
                            else Just <$> newNDArrayHandle ptr
            aux_array  <- peekArray (fromIntegral num_aux_states) aux_states >>= mapM newNDArrayHandle
            return (Nothing, arg_array, grad_array, aux_array, out))

        Just (shared_buffer_name_list, shared_buffer_handle_list) -> alloca (\ptr_shared_buffer_len -> do
            poke ptr_shared_buffer_len (fromIntegral $ length shared_buffer_name_list)
            withCStringArrayT shared_buffer_name_list (\ptr_shared_buffer_name_list ->
                withNDArrayHandleArray shared_buffer_handle_list (\ptr_shared_buffer_handle_list -> do
                    (ptr_updated_shared_buffer_name_list, ptr_updated_shared_buffer_handle_list,
                     num_in_args, in_args, arg_grads, num_aux_states, aux_states, out)
                        <- checked $ mxExecutorSimpleBindEx_
                                symbol devtype_ devid_
                                cnt_g2c g2c_keys g2c_dev_types_ g2c_dev_ids_
                                cnt_provided_grad_req_list provided_grad_req_names provided_grad_req_types
                                cnt_provided_arg_shapes provided_arg_shape_names provided_arg_shape_data_ provided_arg_shape_idx_
                                cnt_provided_arg_dtypes provided_arg_dtype_names provided_arg_dtypes_
                                cnt_provided_arg_stypes provided_arg_stype_names provided_arg_stypes_
                                cnt_shared_arg_names shared_arg_name_list
                                ptr_shared_buffer_len ptr_shared_buffer_name_list ptr_shared_buffer_handle_list
                                (fromMaybe nullExecutorHandle shared_exec_handle)
                    arg_array  <- peekArray (fromIntegral num_in_args) in_args   >>= mapM newNDArrayHandle
                    grad_array <- peekArray (fromIntegral num_in_args) arg_grads
                    grad_array <- forM grad_array $ \ptr ->
                                    if ptr == C2HSImp.nullPtr
                                    then return Nothing
                                    else Just <$> newNDArrayHandle ptr
                    aux_array  <- peekArray (fromIntegral num_aux_states) aux_states >>= mapM newNDArrayHandle
                    update_shared_buffer_len <- fromIntegral <$> peek ptr_shared_buffer_len
                    updated_shared_buffer_name_list
                        <- peekCStringArrayT update_shared_buffer_len ptr_updated_shared_buffer_name_list
                    updated_shared_buffer_handle_list
                        <- peekArray update_shared_buffer_len ptr_updated_shared_buffer_handle_list >>= mapM newNDArrayHandle
                    return (Just (updated_shared_buffer_name_list, updated_shared_buffer_handle_list),
                            arg_array, grad_array, aux_array, out))))

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

{#
fun MXExecutorReshapeEx as mxExecutorReshapeEx_
    {
        `CInt',
        `CInt',
        `CInt',
        `CInt',
        `CUInt',
        withCStringArrayT* `[Text]',
        withArray* `[CInt]',
        withArray* `[CInt]',
        `CUInt',
        withCStringArrayT* `[Text]',
        withArray* `[CInt]',
        withArray* `[CUInt]',
        alloca0- `CUInt' peek*,
        alloca- `Ptr NDArrayHandlePtr' peek*,
        alloca- `Ptr NDArrayHandlePtr' peek*,
        alloca0- `CUInt' peek*,
        alloca- `Ptr NDArrayHandlePtr' peek*,
        `ExecutorHandle',
        alloca- `ExecutorHandle' peekExecutorHandle*
    } -> `CInt'
#}

mxExecutorReshapeEx :: HasCallStack
                    => Bool -> Bool -> Int -> Int
                    -> [Text] -> [Int] -> [Int]        -- group2ctx
                    -> [Text] -> [Int] -> [Int]        -- provided_arg_shapes
                    -> ExecutorHandle                    -- shared exec
                    -> IO ([NDArrayHandle], [Maybe NDArrayHandle], [NDArrayHandle], ExecutorHandle)
mxExecutorReshapeEx partial_shaping allow_up_sizing
                    devtype devid
                    g2c_keys g2c_dev_types g2c_dev_ids
                    provided_arg_shape_names provided_arg_shape_data provided_arg_shape_idx
                    shared_exec_handler = do
    (num_in, arr_in, arr_grad, num_aux, arr_aux, exec_hdl) <- checked $ mxExecutorReshapeEx_
                    partial_shaping_ allow_up_sizing_
                    devtype_ devid_
                    cnt_g2c g2c_keys g2c_dev_types_ g2c_dev_ids_
                    cnt_provided_arg_shapes provided_arg_shape_names provided_arg_shape_data_ provided_arg_shape_idx_
                    shared_exec_handler

    handle_ptrs <- peekArray (fromIntegral num_in) arr_in
    arr_in   <- mapM newNDArrayHandle handle_ptrs

    handle_ptrs <- peekArray (fromIntegral num_in) arr_grad
    arr_grad <- flip mapM handle_ptrs $ \ptr ->
                    if ptr == nullPtr
                    then return Nothing
                    else Just <$> newNDArrayHandle ptr

    handle_ptrs <- peekArray (fromIntegral num_aux) arr_aux
    arr_aux  <- mapM newNDArrayHandle handle_ptrs

    return (arr_in, arr_grad, arr_aux, exec_hdl)
  where
    partial_shaping_ = if partial_shaping then 1 else 0
    allow_up_sizing_ = if allow_up_sizing then 1 else 0
    devtype_ = fromIntegral devtype
    devid_ = fromIntegral devid

    cnt_g2c = fromIntegral $ length g2c_keys
    g2c_dev_types_ = map fromIntegral g2c_dev_types
    g2c_dev_ids_ = map fromIntegral g2c_dev_ids

    cnt_provided_arg_shapes = fromIntegral $ length provided_arg_shape_names
    provided_arg_shape_data_ = map fromIntegral provided_arg_shape_data
    provided_arg_shape_idx_  = map fromIntegral provided_arg_shape_idx

-- {#
-- fun MXExecutorGetOptimizedSymbol as mxExecutorGetOptimizedSymbol_
--     {
--         `ExecutorHandle',
--         alloca- `SymbolHandle' peekSymbolHandle*
--     } -> `CInt'
-- #}

alloca0 :: (Storable a, Num a) => (Ptr a -> IO b) -> IO b
alloca0 f = alloca (\ptr -> poke ptr 0 >> f ptr)
