{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE ImplicitParams       #-}
{-# LANGUAGE RecordWildCards      #-}
{-# LANGUAGE UndecidableInstances #-}
module MXNet.NN.Module.Basic where

import           Control.Lens                (ix, (^?!))
import           Data.Default.Class
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


data LinearArgs = LinearArgs {
    _linear_out_features :: Int,
    _linear_bias         :: Bool
}

instance Default LinearArgs where
    def = LinearArgs undefined True

data Linear t = Linear LinearArgs (GenericModule (Linear t))

data LinearParamEnums = LinearWeights | LinearBias
    deriving (Enum, Show, Eq)

instance DType t => Module (Linear t) where

    type ModuleDType (Linear t)        = t
    type ModuleArgs (Linear t)         = LinearArgs
    type ModuleParamEnums (Linear t)   = LinearParamEnums
    type ModuleParamTensors (Linear t) = (NDArray t, NDArray t)

    init scope args initializer = do
        let default_initializer LinearWeights = initEmpty
            default_initializer LinearBias    = initZeros
        Linear args <$> initGenericModule scope (fromMaybe default_initializer initializer)

    forward (Linear LinearArgs{..} generic) inp = do
        (bias, weights) <- getOrInitParams generic $ \initializer -> do
            [batch_size, n_in] <- take 2 <$> ndshape inp
            bias    <- makeEmptyNDArray [_linear_out_features] ?device
            weights <- makeEmptyNDArray [_linear_out_features, n_in] ?device
            initializer LinearBias bias
            initializer LinearWeights weights
            attachGradient bias ReqWrite
            attachGradient weights ReqWrite
            return (bias, weights)
        prim O._FullyConnected
              (#data := inp
            .& #bias := bias
            .& #weight := weights
            .& #num_hidden := _linear_out_features
            .& #no_bias := not _linear_bias
            .& Nil)

    parameters (Linear args generic@(GenericModule name _ _)) = do
        (bias, weights) <- getParams generic
        return [ Parameter (scopedName $ NE.fromList ["bias", name]) bias
               , Parameter (scopedName $ NE.fromList ["weights", name]) weights ]

data BatchNormArgs = BatchNormArgs {
    _bn_axis             :: Int,
    _bn_momentum         :: Float,
    _bn_eps              :: Double,
    _bn_scale            :: Bool,
    _bn_use_global_stats :: Bool
}

instance Default BatchNormArgs where
    def = BatchNormArgs 1 0.9 1e-5 True False

data BatchNorm t = BatchNorm BatchNormArgs (GenericModule (BatchNorm t))
data BatchNormParamEnums = BatchNormGamma | BatchNormBeta | BatchNormMean | BatchNormVar
    deriving (Eq, Show, Enum)

instance DType t => Module (BatchNorm t) where

    type ModuleDType (BatchNorm t) = t
    type ModuleArgs (BatchNorm t) = BatchNormArgs
    type ModuleParamEnums (BatchNorm t) = BatchNormParamEnums
    type ModuleParamTensors (BatchNorm t) = (NDArray t, NDArray t, NDArray t, NDArray t)

    init scope args initializer = do
        let default_initializer BatchNormGamma = initOnes
            default_initializer BatchNormBeta  = initZeros
            default_initializer BatchNormMean  = initZeros
            default_initializer BatchNormVar   = initOnes
        BatchNorm args <$> initGenericModule scope (fromMaybe default_initializer initializer)

    forward (BatchNorm BatchNormArgs{..} generic) inp = do
        (gamma, beta, mean, var) <- getOrInitParams generic $ \initializer -> do
            shape <- ndshape inp
            let n_channels = shape ^?! ix _bn_axis
            gamma <- makeEmptyNDArray [n_channels] ?device
            beta  <- makeEmptyNDArray [n_channels] ?device
            mean  <- makeEmptyNDArray [n_channels] ?device
            var   <- makeEmptyNDArray [n_channels] ?device
            initializer BatchNormGamma gamma
            initializer BatchNormBeta  beta
            initializer BatchNormMean  mean
            initializer BatchNormVar   var
            attachGradient gamma ReqWrite
            attachGradient beta  ReqWrite
            return (gamma, beta, mean, var)
        prim O._BatchNorm (#data := inp
                        .& #gamma := gamma
                        .& #beta := beta
                        .& #moving_mean := mean
                        .& #moving_var := var
                        .& #eps := _bn_eps
                        .& #momentum := _bn_momentum
                        .& #axis := _bn_axis
                        .& #use_global_stats := _bn_use_global_stats
                        .& #fix_gamma := not _bn_scale
                        .& Nil)

    parameters (BatchNorm args generic@(GenericModule name _ _)) = do
        (gamma, beta, _, _) <- getParams generic
        return [ Parameter (scopedName $ NE.fromList ["gamma", name]) gamma
               , Parameter (scopedName $ NE.fromList ["beta", name]) beta ]
