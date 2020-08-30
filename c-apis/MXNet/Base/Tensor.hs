{-# LANGUAGE TypeFamilyDependencies #-}
module MXNet.Base.Tensor where

import           Data.Bifunctor           (bimap)
import           RIO
import           RIO.List                 (unzip)

import           MXNet.Base.NDArray
import           MXNet.Base.Raw.Common
import           MXNet.Base.Raw.NDArray
import           MXNet.Base.Raw.Symbol
import           MXNet.Base.Spec.Operator (ArgsHMap)
import           MXNet.Base.Symbol
import           MXNet.Base.Types


-- TensorApply is injective
type family TensorApply t = c | c -> t where
    TensorApply NDArrayHandle = Maybe [NDArrayHandle] -> IO [NDArrayHandle]
    TensorApply (NDArray a) = Maybe [NDArray a] -> IO [NDArray a]
    TensorApply SymbolHandle = Text -> IO SymbolHandle

type family TensorM t = m | m -> t

type instance TensorM (NDArray a)   = IO (NDArray a)
type instance TensorM NDArrayHandle = IO NDArrayHandle

class Tensor t where
    apply :: Text -> [(Text, Text)] -> Either [(Text, t)] [t] -> TensorApply t

class Tensor t => PrimTensorOp t where
    prim :: HasCallStack => (ArgsHMap s t a -> TensorApply t) -> ArgsHMap s t a -> TensorM t

instance Tensor NDArrayHandle where
    apply opname scalars tensors outputs = do
        let tensors' = case tensors of
                         Left kwargs -> snd $ unzip kwargs
                         Right args  -> args
        op <- nnGetOpHandle opname
        mxImperativeInvoke (fromOpHandle op) tensors' scalars outputs


instance DType a => Tensor (NDArray a) where
    apply opname scalars tensors outputs = do
        let tensors' = bimap (map (second unNDArray)) (map unNDArray) tensors
            outputs' = fmap (map unNDArray) outputs :: Maybe [NDArrayHandle]
        ret <- apply opname scalars tensors' outputs' :: IO [NDArrayHandle]
        return $ map NDArray ret


instance Tensor SymbolHandle where
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

instance PrimTensorOp NDArrayHandle where
    prim op args = op args Nothing >>= \[x] -> return x

instance DType a => PrimTensorOp (NDArray a) where
    prim op args = op args Nothing >>= \[x] -> return x


