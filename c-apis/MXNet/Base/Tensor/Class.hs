{-# LANGUAGE AllowAmbiguousTypes    #-}
{-# LANGUAGE ConstraintKinds        #-}
{-# LANGUAGE MultiParamTypeClasses  #-}
{-# LANGUAGE PolyKinds              #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE TypeOperators          #-}
{-# LANGUAGE UndecidableInstances   #-}
module MXNet.Base.Tensor.Class where

import                          Data.Bifunctor           (bimap)
import                          Data.Kind                (Constraint)
import                qualified GHC.TypeLits             as L
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
    TensorApply (NDArray a)   = Maybe [NDArray a] -> IO [NDArray a]
    TensorApply SymbolHandle  = Text -> IO SymbolHandle
    TensorApply (Symbol a)    = Text -> IO (Symbol a)

type family TensorDType t :: L.Symbol where
    TensorDType (NDArray t) = DTypeName t
    TensorDType (Symbol  t) = DTypeName t
    TensorDType _           = "unknown"

type family TensorWithDType t s where
    TensorWithDType (NDArray t) s = NDArray s
    TensorWithDType (Symbol  t) s = Symbol  s
    TensorWithDType t _ = t

class Tensor (t :: * -> *) where
    type RawTensor t = s | s -> t
    toRaw    :: DType a => t a -> RawTensor t
    fromRaw  :: DType a => RawTensor t -> t a
    applyRaw :: (HasCallStack, DType a)
                 => Text
                 -> [(Text, Text)]
                 -> Either [(Text, RawTensor t)] [RawTensor t]
                 -> TensorApply (t a)

instance Tensor NDArray where
    type RawTensor NDArray = NDArrayHandle
    toRaw (NDArray hdl) = hdl
    fromRaw hdl = NDArray hdl
    applyRaw opname scalars tensors outputs = do
        let tensors' = case tensors of
                         Left kwargs -> snd $ unzip kwargs
                         Right args  -> args
        op  <- nnGetOpHandle opname
        let outputsRaw = map toRaw <$> outputs
        map fromRaw <$> mxImperativeInvoke (fromOpHandle op) tensors' scalars outputsRaw

instance Tensor Symbol where
    type RawTensor Symbol = SymbolHandle
    toRaw (Symbol hdl) = hdl
    fromRaw hdl = Symbol hdl
    applyRaw opname scalars tensors name = do
        let (scalarkeys, scalarvals) = unzip scalars
            (tensorkeys, tensorvals) = case tensors of
                                         Left kwargs -> first Just $ unzip kwargs
                                         Right args  -> (Nothing, args)
        op <- nnGetOpHandle opname
        sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op)
                                          scalarkeys
                                          scalarvals
        mxSymbolCompose sym name tensorkeys tensorvals
        return $ fromRaw sym

type family TensorMonad (t :: * -> *) :: * -> *
type instance TensorMonad NDArray = IO

class (MonadIO (TensorMonad t), Tensor t) => PrimTensorOp t where
    prim      :: (HasCallStack, DType v)
              => (ArgsHMap s (p :: kind) a -> TensorApply (t v)) -> ArgsHMap s p a -> TensorMonad t (t v)
    primMulti :: (HasCallStack, DType v)
              => (ArgsHMap s (p :: kind) a -> TensorApply (t v)) -> ArgsHMap s p a -> TensorMonad t [t v]

instance PrimTensorOp NDArray where
    prim op args = op args Nothing >>= \case
                        [x] -> return x
                        _   -> error "the operation returns multiple ndarrays"
    primMulti op args = op args Nothing

