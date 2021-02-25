{-# LANGUAGE MultiParamTypeClasses  #-}
{-# LANGUAGE TypeFamilyDependencies #-}
module MXNet.Base.Tensor.Class where

import                          Data.Bifunctor           (bimap)
import                          Data.Kind                (Constraint)
import                          RIO
import                          RIO.List                 (unzip)

import {-# SOURCE #-}           MXNet.Base.NDArray
import                          MXNet.Base.Raw.Common
import                          MXNet.Base.Raw.NDArray
import                          MXNet.Base.Raw.Symbol
import                          MXNet.Base.Spec.Operator (ArgsHMap)
import                          MXNet.Base.Symbol
import                          MXNet.Base.Types


-- TensorApply is injective
type family TensorApply t = c | c -> t where
    TensorApply NDArrayHandle = Maybe [NDArrayHandle] -> IO [NDArrayHandle]
    TensorApply (NDArray a) = Maybe [NDArray a] -> IO [NDArray a]
    TensorApply SymbolHandle = Text -> IO SymbolHandle

type family TensorMonad t :: * -> *

type instance TensorMonad (NDArray a)   = IO
type instance TensorMonad NDArrayHandle = IO

class TensorOp ti to where
    apply :: HasCallStack => Text -> [(Text, Text)] -> Either [(Text, ti)] [ti] -> TensorApply to


class TensorOp ti to => PrimTensorOp ti to where
    prim      :: HasCallStack => (ArgsHMap s ti a -> TensorApply to) -> ArgsHMap s ti a -> TensorMonad to to
    primMulti :: HasCallStack => (ArgsHMap s ti a -> TensorApply to) -> ArgsHMap s ti a -> TensorMonad to [to]


instance TensorOp NDArrayHandle NDArrayHandle where
    apply opname scalars tensors outputs = do
        let tensors' = case tensors of
                         Left kwargs -> snd $ unzip kwargs
                         Right args  -> args
        op <- nnGetOpHandle opname
        mxImperativeInvoke (fromOpHandle op) tensors' scalars outputs


instance (DType a, DType b) => TensorOp (NDArray a) (NDArray b) where
    apply opname scalars tensors outputs = do
        let tensors' = bimap (map (second unNDArray)) (map unNDArray) tensors
            outputs' = fmap (map unNDArray) outputs :: Maybe [NDArrayHandle]
        ret <- apply opname scalars tensors' outputs' :: IO [NDArrayHandle]
        return $ map NDArray ret


instance TensorOp SymbolHandle SymbolHandle where
    apply opname scalars tensors name = do
        let (scalarkeys, scalarvals) = unzip scalars
            (tensorkeys, tensorvals) = case tensors of
                                         Left kwargs -> first Just $ unzip kwargs
                                         Right args  -> (Nothing, args)
        op <- nnGetOpHandle opname
        sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op)
                                          scalarkeys
                                          scalarvals
        mxSymbolCompose sym name tensorkeys tensorvals
        return sym

instance PrimTensorOp NDArrayHandle NDArrayHandle where
    prim op args = op args Nothing >>= \case
                        [x] -> return x
                        _   -> error "the operation returns multiple ndarrays"
    primMulti op args = op args Nothing

instance (DType a, DType b) => PrimTensorOp (NDArray a) (NDArray b) where
    prim      op args = op args Nothing >>= \case
                        [x] -> return x
                        _   -> error "the operation returns multiple ndarrays"
    primMulti op args = op args Nothing


