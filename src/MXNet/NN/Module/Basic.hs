{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE ImplicitParams       #-}
{-# LANGUAGE RecordWildCards      #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fplugin=Data.Record.Anon.Plugin#-}
module MXNet.NN.Module.Basic where

import           Control.Lens                (ix, (^?!))
import           Data.Default.Class
import           Data.Hashable
import qualified Data.HashMap.Strict         as M
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

data LinearParamEnums = LinearWeights | LinearBias
    deriving (Enum, Show, Eq, Typeable)

instance Hashable LinearParamEnums where
    hashWithSalt = hashUsing fromEnum

data Linear t = Linear LinearArgs (GenericModule LinearParamEnums t)

instance DType t => Module (Linear t) where

    type ModuleDType  (Linear t) = t
    type ModuleArgs   (Linear t) = LinearArgs
    type ModuleInput  (Linear t) = NDArray t
    type ModuleOutput (Linear t) = NDArray t

    init scope args initializer = do
        let default_initializer = M.fromList
                [(scope :> LinearWeights, initEmpty),
                 (scope :> LinearBias,    initZeros)]
        Linear args <$> init scope () (M.union initializer default_initializer)

    forward (Linear LinearArgs{..} generic) inp = do
        let scope = _gmodule_scope generic
        pt <- getOrInitParams generic $ \initializer -> do
            [batch_size, n_in] <- take 2 <$> ndshape inp
            bias    <- makeEmptyNDArray [_linear_out_features] ?device
            weights <- makeEmptyNDArray [_linear_out_features, n_in] ?device
            initializer M.! (scope :> LinearBias)    $ bias
            initializer M.! (scope :> LinearWeights) $ weights
            attachGradient bias ReqWrite
            attachGradient weights ReqWrite
            return $ M.fromList [(LinearBias, bias), (LinearWeights, weights)]

        prim O._FullyConnected ANON{
            _data = inp,
            bias = Just $ pt M.! LinearBias,
            weight = Just $ pt M.! LinearWeights,
            num_hidden = _linear_out_features,
            no_bias = Just (not _linear_bias)
        }

    parameters (Linear _ generic) = parameters generic

data BatchNormArgs = BatchNormArgs {
    _bn_axis             :: Int,
    _bn_momentum         :: Float,
    _bn_eps              :: Double,
    _bn_scale            :: Bool,
    _bn_use_global_stats :: Bool
}

instance Default BatchNormArgs where
    def = BatchNormArgs 1 0.9 1e-5 True False

data BatchNormParamEnums = BatchNormGamma | BatchNormBeta | BatchNormMean | BatchNormVar
    deriving (Eq, Show, Enum, Typeable)

instance Hashable BatchNormParamEnums where
    hashWithSalt = hashUsing fromEnum

data BatchNorm t = BatchNorm BatchNormArgs (GenericModule BatchNormParamEnums t)

instance DType t => Module (BatchNorm t) where

    type ModuleDType  (BatchNorm t) = t
    type ModuleArgs   (BatchNorm t) = BatchNormArgs
    type ModuleInput  (BatchNorm t) = NDArray t
    type ModuleOutput (BatchNorm t) = NDArray t

    init scope args initializer = do
        let default_initializer = M.fromList
                [(scope :> BatchNormGamma, initOnes),
                 (scope :> BatchNormBeta,  initZeros),
                 (scope :> BatchNormMean,  initZeros),
                 (scope :> BatchNormVar,   initOnes)]
        BatchNorm args <$> init scope () (M.union initializer default_initializer)

    forward (BatchNorm BatchNormArgs{..} generic) inp = do
        let scope = _gmodule_scope generic
        pt <- getOrInitParams generic $ \initializer -> do
            shape <- ndshape inp
            let n_channels = shape ^?! ix _bn_axis
            gamma <- makeEmptyNDArray [n_channels] ?device
            beta  <- makeEmptyNDArray [n_channels] ?device
            mean  <- makeEmptyNDArray [n_channels] ?device
            var   <- makeEmptyNDArray [n_channels] ?device
            initializer M.! (scope :> BatchNormGamma) $ gamma
            initializer M.! (scope :> BatchNormBeta)  $ beta
            initializer M.! (scope :> BatchNormMean)  $ mean
            initializer M.! (scope :> BatchNormVar)   $ var
            attachGradient gamma ReqWrite
            attachGradient beta  ReqWrite
            return $ M.fromList
                [(BatchNormGamma, gamma),
                 (BatchNormBeta, beta),
                 (BatchNormMean, mean),
                 (BatchNormVar, var)]

        prim O._BatchNorm ANON{
            _data = inp,
            gamma = Just $ pt M.! BatchNormGamma,
            beta = Just $ pt M.! BatchNormBeta,
            moving_mean = Just $ pt M.! BatchNormMean,
            moving_var = Just $ pt M.! BatchNormVar,
            eps = Just _bn_eps,
            momentum = Just _bn_momentum,
            axis = Just _bn_axis,
            use_global_stats = Just _bn_use_global_stats,
            fix_gamma = Just (not _bn_scale)
        }

    parameters (BatchNorm _ generic) = parameters generic
