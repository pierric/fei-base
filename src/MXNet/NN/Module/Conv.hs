{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE ImplicitParams       #-}
{-# LANGUAGE RecordWildCards      #-}
{-# LANGUAGE TypeApplications     #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fplugin=Data.Record.Anon.Plugin#-}
module MXNet.NN.Module.Conv where

import           Data.Default.Class
import           Data.Record.Anon
import           Data.Record.Anon.Simple     (Record)
import qualified Data.Record.Anon.Simple     as Anon
import           GHC.TypeLits
import           RIO
import qualified RIO.NonEmpty                as NE
import qualified RIO.NonEmpty.Partial        as NE (fromList)
import qualified RIO.Text                    as T

import           MXNet.Base.AutoGrad
import           MXNet.Base.Core.Enum
import           MXNet.Base.Core.Spec
import qualified MXNet.Base.Operators.Tensor as O
import           MXNet.Base.Tensor           hiding (Symbol)
import           MXNet.Base.Types
import           MXNet.NN.Initializer
import           MXNet.NN.Module.Class

data ConvBase u = ConvBase (Record (FieldsExc (O.ParameterList_Convolution NDArray u) '["_data", "bias", "weight"])) (GenericModule (ConvBase u))
data ConvParamEnums = ConvWeights | ConvBias

instance DType u => Module (ConvBase u) where

    type ModuleDType (ConvBase u) = u
    type ModuleArgs  (ConvBase u) = Record (FieldsExc (O.ParameterList_Convolution NDArray u) '["_data", "bias", "weight"])
    type ModuleParamEnums   (ConvBase u) = ConvParamEnums
    type ModuleParamTensors (ConvBase u) = (NDArray u, NDArray u)

    init scope args initializer = do
        let default_initializer ConvWeights = initEmpty
            default_initializer ConvBias    = initZeros
        ConvBase args <$> initGenericModule scope (fromMaybe default_initializer initializer)

    forward (ConvBase args generic) inp = do
        (bias, weights) <- getOrInitParams generic $ \initializer -> do
            [batch_size, n_channels] <- take 2 <$> ndshape inp
            let [kh, kw]   = Anon.get #kernel args
                num_filter = Anon.get #num_filter args
            bias    <- makeEmptyNDArray [num_filter] ?device
            weights <- makeEmptyNDArray [num_filter, n_channels, kh, kw] ?device
            initializer ConvBias bias
            initializer ConvWeights weights
            attachGradient bias ReqWrite
            attachGradient weights ReqWrite
            return (bias, weights)
        prim O._Convolution (Anon.insert #_data inp $ Anon.insert #bias (Just bias) $ Anon.insert #weight (Just weights) args)

    parameters (ConvBase args generic@(GenericModule name _ _)) = do
        (bias, weights) <- getParams generic
        return [ Parameter (scopedName $ NE.fromList ["bias", name]) bias
               , Parameter (scopedName $ NE.fromList ["weights", name]) weights ]

-- type FullConvArgsHMap t = BuildArgsHMap "_Convolution" t '[ '("kernel", [Int]), '("num_filter", Int), '("stride", [Int]), '("pad", [Int]), '("dilate", [Int]), '("layout", Maybe (EnumType '["NCDHW", "NCHW", "NCW", "NDHWC", "NHWC"])), '("num_group", Int), '("no_bias", Bool)]


data ConvArgs = ConvArgs {
    _conv_out_channels :: Int,
    _conv_kernel       :: [Int],
    _conv_stride       :: Maybe [Int],
    _conv_padding      :: Maybe [Int],
    _conv_dilation     :: Maybe [Int],
    _conv_bias         :: Bool
}

instance Default ConvArgs where
    def = ConvArgs undefined undefined Nothing Nothing Nothing True

newtype Conv2D u = Conv2D (ConvBase u)

instance DType u => Module (Conv2D u) where
    type ModuleDType        (Conv2D u) = u
    type ModuleArgs         (Conv2D u) = ConvArgs
    type ModuleParamEnums   (Conv2D u) = ModuleParamEnums   (ConvBase u)
    type ModuleParamTensors (Conv2D u) = ModuleParamTensors (ConvBase u)

    init scope ConvArgs{..} initializer =
        let stride   = fromMaybe [1, 1] _conv_stride
            padding  = fromMaybe [0, 0] _conv_padding
            dilation = fromMaybe [1, 1] _conv_dilation
            args' = ANON {
                 kernel = _conv_kernel,
                 stride = Just stride,
                 dilate = Just dilation,
                 pad = Just padding,
                 num_filter = _conv_out_channels,
                 num_group = Just 1,
                 workspace = Nothing,
                 no_bias = Just (not _conv_bias),
                 cudnn_tune = Nothing,
                 cudnn_off = Nothing,
                 layout = Just (Just (EnumType (Proxy @ "NCHW")))
            } :: ModuleArgs (ConvBase u)
         in Conv2D <$> init scope args' initializer

    forward (Conv2D base) inp = forward base inp

    parameters (Conv2D base) = parameters base
