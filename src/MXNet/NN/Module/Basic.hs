{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE ImplicitParams       #-}
{-# LANGUAGE UndecidableInstances #-}
module MXNet.NN.Module.Basic where

import           RIO
import qualified RIO.NonEmpty                as NE
import qualified RIO.NonEmpty.Partial        as NE (fromList)
import qualified RIO.Text                    as T

import           MXNet.Base.AutoGrad
import qualified MXNet.Base.Operators.Tensor as O
import           MXNet.Base.Tensor
import           MXNet.Base.Types
import           MXNet.NN.Initializer
import           MXNet.NN.Module.Class

data Linear args = Linear args (GenericModule (Linear args))

instance (DType t, Satisfying "_FullyConnected" '["data", "bias", "weight"] '(NDArray, t) args) => Module (Linear (ArgsHMap "_FullyConnected" '(NDArray, t) args)) where

    type ModuleDType (Linear (ArgsHMap "_FullyConnected" '(NDArray, t) args)) = t
    type ModuleArgs (Linear (ArgsHMap "_FullyConnected" '(NDArray, t) args)) = ArgsHMap "_FullyConnected" '(NDArray, t) args
    data ModuleParamEnums (Linear (ArgsHMap "_FullyConnected" '(NDArray, t) args)) = LinearWeights | LinearBias
    type ModuleParamTensors (Linear (ArgsHMap "_FullyConnected" '(NDArray, t) args)) = (NDArray t, NDArray t)

    init scope args initializer = do
        let default_initializer LinearWeights = initEmpty
            default_initializer LinearBias    = initZeros
        Linear args <$> initGenericModule scope (fromMaybe default_initializer initializer)

    forward (Linear args generic) inp = do
        (bias, weights) <- getOrInitParams generic $ \initializer -> do
            [batch_size, n_in] <- take 2 <$> ndshape inp
            let num_hidden = args ! #num_hidden
            bias    <- makeEmptyNDArray [num_hidden] ?device
            weights <- makeEmptyNDArray [num_hidden, n_in] ?device
            initializer LinearBias bias
            initializer LinearWeights weights
            attachGradient bias ReqWrite
            attachGradient weights ReqWrite
            return (bias, weights)
        prim O._FullyConnected (#data := inp .& #bias := bias .& #weight := weights .& args)

    parameters (Linear args generic@(GenericModule name _ _)) = do
        (bias, weights) <- getParams generic
        return [ Parameter (scopedName $ NE.fromList ["bias", name]) bias
               , Parameter (scopedName $ NE.fromList ["weights", name]) weights ]

