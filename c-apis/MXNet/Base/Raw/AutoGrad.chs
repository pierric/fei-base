{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.Base.Raw.AutoGrad where

import RIO
import RIO.List (unzip, unzip3)
import Foreign.C.Types
import Foreign.Ptr
import Foreign.Marshal.Array (peekArray, withArray)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Storable (peek)

{# import MXNet.Base.Raw.Common #}
{# import MXNet.Base.Raw.NDArray #}
{# import MXNet.Base.Raw.Symbol #}

#include <mxnet/c_api.h>

length_ = fromIntegral . length
boolToCInt = fromIntegral . fromEnum

{#
fun MXAutogradSetIsRecording as mxAutogradSetIsRecording_
    {
        `CInt',
        alloca- `CInt' peek*
    } -> `CInt'
#}

mxAutogradSetIsRecording :: HasCallStack => Int -> IO Int
mxAutogradSetIsRecording recording = do
    prev <- checked $ mxAutogradSetIsRecording_ $ fromIntegral recording
    return $ fromIntegral prev

{#
fun MXAutogradSetIsTraining as mxAutogradSetIsTraining_
    {
        `CInt',
        alloca- `CInt' peek*
    } -> `CInt'
#}

mxAutogradSetIsTraining :: HasCallStack => Int -> IO Int
mxAutogradSetIsTraining training = do
    prev <- checked $ mxAutogradSetIsTraining_ $ fromIntegral training
    return $ fromIntegral prev

{#
fun MXAutogradIsRecording as mxAutogradIsRecording_
    {
        alloca- `CUChar' peek*
    } -> `CInt'
#}

mxAutogradIsRecording :: HasCallStack => IO Bool
mxAutogradIsRecording = do
    recording <- checked $ mxAutogradIsRecording_
    return $ recording /= 0

{#
fun MXAutogradIsTraining as mxAutogradIsTraining_
    {
        alloca- `CUChar' peek*
    } -> `CInt'
#}

mxAutogradIsTraining :: HasCallStack => IO Bool
mxAutogradIsTraining = do
    training <- checked $ mxAutogradIsTraining_
    return $ training /= 0

{#
fun MXAutogradMarkVariables as mxAutogradMarkVariables_
    {
        `CUInt',
        withNDArrayHandleArray* `[NDArrayHandle]',
        withArray* `[CUInt]',
        withNDArrayHandleArray* `[NDArrayHandle]'
    } -> `CInt'
#}

mxAutogradMarkVariables :: [(NDArrayHandle, Int, NDArrayHandle)] -> IO ()
mxAutogradMarkVariables variables = do
    let (vars, req_types, grads) = unzip3 variables
    checked $ mxAutogradMarkVariables_
        (length_ variables)
        vars
        (map fromIntegral req_types)
        grads

{#
fun MXAutogradComputeGradient as mxAutogradComputeGradient_
    {
        `CUInt',
        withNDArrayHandleArray* `[NDArrayHandle]'
    } -> `CInt'
#}

mxAutogradComputeGradient :: [NDArrayHandle] -> IO ()
mxAutogradComputeGradient output_arrays = do
    checked $ mxAutogradComputeGradient_
        (length_ output_arrays)
        output_arrays

{#
fun MXAutogradBackwardEx as mxAutogradBackwardEx_
    {
        `CUInt',
        withNDArrayHandleArray* `[NDArrayHandle]',
        id `Ptr NDArrayHandlePtr',
        `CUInt',
        withNDArrayHandleArray* `[NDArrayHandle]',
        `CInt',
        `CInt',
        `CInt',
        id `Ptr (Ptr NDArrayHandlePtr)', -- [out, optional] list of the gradients of the output
        id `Ptr (Ptr CInt)' -- [out, optional] list of storage types of the output
    } -> `CInt'
#}

mxAutogradBackwardEx :: Either [NDArrayHandle] ([(NDArrayHandle, NDArrayHandle)]) -- output arrays
                     -> [NDArrayHandle]  -- variable arrays
                     -> Bool             -- retrain graph
                     -> Bool             -- create graph
                     -> Bool             -- is training
                     -> Bool             -- retrieve gradients of variables
                     -> IO (Maybe [(NDArrayHandle, Int)])
mxAutogradBackwardEx output_arrays variables retrain_graph create_graph is_train ret_grads =
    case output_arrays of
        Left  oa -> _call oa C2HSImp.nullPtr variables
        Right ps -> let (oa, og) = unzip ps
                     in withNDArrayHandleArray og (\ptr_og -> _call oa ptr_og variables)
    where
        _call oa og va
            | not ret_grads = do checked $ mxAutogradBackwardEx_
                                    (length_ oa) oa og (length_ va) va
                                    (boolToCInt retrain_graph)
                                    (boolToCInt create_graph)
                                    (boolToCInt is_train)
                                    C2HSImp.nullPtr C2HSImp.nullPtr
                                 return Nothing

            | otherwise = alloca $ \(pp :: Ptr (Ptr NDArrayHandlePtr)) ->
                          alloca $ \(pt :: Ptr (Ptr CInt)) -> do
                            let num_vars = length va
                            checked $ mxAutogradBackwardEx_
                                (length_ oa) oa og (fromIntegral num_vars) va
                                (boolToCInt retrain_graph)
                                (boolToCInt create_graph)
                                (boolToCInt is_train)
                                pp pt
                            var_grads  <- peek pp >>= peekArray num_vars >>= mapM newNDArrayHandle
                            var_stypes <- peek pt >>= peekArray num_vars >>= return . map fromIntegral
                            return $ Just $ zip var_grads var_stypes

{#
fun MXAutogradGetSymbol as mxAutogradGetSymbol_
    {
        `NDArrayHandle',
        alloca- `SymbolHandle' peekSymbolHandle*
    } -> `CInt'
#}

mxAutogradGetSymbol :: NDArrayHandle -> IO SymbolHandle
mxAutogradGetSymbol = checked . mxAutogradGetSymbol_
