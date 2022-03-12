{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE ImplicitParams       #-}
{-# LANGUAGE UndecidableInstances #-}
module MXNet.NN.Module.Conv where

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

data Convolution args = Convolution args (GenericModule (Convolution args))

instance (DType t, Satisfying "_Convolution" '["data", "bias", "weight"] '(NDArray, t) args) => Module (Convolution (ArgsHMap "_Convolution" '(NDArray, t) args)) where

    type ModuleDType (Convolution (ArgsHMap "_Convolution" '(NDArray, t) args)) = t
    type ModuleArgs (Convolution (ArgsHMap "_Convolution" '(NDArray, t) args)) = (ArgsHMap "_Convolution" '(NDArray, t) args)
    data ModuleParamEnums (Convolution (ArgsHMap "_Convolution" '(NDArray, t) args)) = ConvWeights | ConvBias
    type ModuleParamTensors (Convolution (ArgsHMap "_Convolution" '(NDArray, t) args)) = (NDArray t, NDArray t)

    init scope args initializer = do
        let default_initializer ConvWeights = initEmpty
            default_initializer ConvBias    = initZeros
        Convolution args <$> initGenericModule scope (fromMaybe default_initializer initializer)

    forward (Convolution args generic) inp = do
        (bias, weights) <- getOrInitParams generic $ \initializer -> do
            [batch_size, n_channels] <- take 2 <$> ndshape inp
            let [kh, kw] = args ! #kernel
                num_filter = args ! #num_filter
            bias    <- makeEmptyNDArray [num_filter] ?device
            weights <- makeEmptyNDArray [num_filter, n_channels, kh, kw] ?device
            initializer ConvBias bias
            initializer ConvWeights weights
            attachGradient bias ReqWrite
            attachGradient weights ReqWrite
            return (bias, weights)
        prim O._Convolution (#data := inp .& #bias := bias .& #weight := weights .& args)

    parameters (Convolution args generic@(GenericModule name _ _)) = do
        (bias, weights) <- getParams generic
        return [ Parameter (scopedName $ NE.fromList ["bias", name]) bias
               , Parameter (scopedName $ NE.fromList ["weights", name]) weights ]
