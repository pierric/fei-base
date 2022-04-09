{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE ImplicitParams       #-}
{-# LANGUAGE RecordWildCards      #-}
{-# LANGUAGE UndecidableInstances #-}
module MXNet.NN.Module.Conv where

import           Data.Default.Class
import           GHC.TypeLits
import           RIO
import qualified RIO.NonEmpty                as NE
import qualified RIO.NonEmpty.Partial        as NE (fromList)
import qualified RIO.Text                    as T

import           MXNet.Base.AutoGrad
import qualified MXNet.Base.Operators.Tensor as O
import           MXNet.Base.Tensor           hiding (Symbol)
import           MXNet.Base.Types
import           MXNet.NN.Initializer
import           MXNet.NN.Module.Class

data ConvBase args = ConvBase args (GenericModule (ConvBase args))
data ConvParamEnums = ConvWeights | ConvBias

instance (DType t, Satisfying "_Convolution" '["data", "bias", "weight"] '(NDArray, t) args)
    => Module (ConvBase (ArgsHMap "_Convolution" '(NDArray, t) args)) where

    type ModuleDType (ConvBase (ArgsHMap "_Convolution" '(NDArray, t) args)) = t
    type ModuleArgs (ConvBase (ArgsHMap "_Convolution" '(NDArray, t) args)) = (ArgsHMap "_Convolution" '(NDArray, t) args)
    type ModuleParamEnums (ConvBase (ArgsHMap "_Convolution" '(NDArray, t) args)) = ConvParamEnums
    type ModuleParamTensors (ConvBase (ArgsHMap "_Convolution" '(NDArray, t) args)) = (NDArray t, NDArray t)

    init scope args initializer = do
        let default_initializer ConvWeights = initEmpty
            default_initializer ConvBias    = initZeros
        ConvBase args <$> initGenericModule scope (fromMaybe default_initializer initializer)

    forward (ConvBase args generic) inp = do
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

    parameters (ConvBase args generic@(GenericModule name _ _)) = do
        (bias, weights) <- getParams generic
        return [ Parameter (scopedName $ NE.fromList ["bias", name]) bias
               , Parameter (scopedName $ NE.fromList ["weights", name]) weights ]

type FullConvArgsHMap t = BuildArgsHMap "_Convolution" t '[ '("kernel", [Int]), '("num_filter", Int), '("stride", [Int]), '("pad", [Int]), '("dilate", [Int]), '("layout", Maybe (EnumType '["NCDHW", "NCHW", "NCW", "NDHWC", "NHWC"])), '("num_group", Int), '("no_bias", Bool)]


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

newtype Conv2D t = Conv2D (ConvBase (FullConvArgsHMap t))

instance DType t => Module (Conv2D t) where
    type ModuleDType        (Conv2D t) = t
    type ModuleArgs         (Conv2D t) = ConvArgs
    type ModuleParamEnums   (Conv2D t) = ModuleParamEnums   (ConvBase (FullConvArgsHMap t))
    type ModuleParamTensors (Conv2D t) = ModuleParamTensors (ConvBase (FullConvArgsHMap t))

    init scope ConvArgs{..} initializer =
        let stride   = fromMaybe [1, 1] _conv_stride
            padding  = fromMaybe [0, 0] _conv_padding
            dilation = fromMaybe [1, 1] _conv_dilation
            args' = #kernel := _conv_kernel
                 .& #num_filter := _conv_out_channels
                 .& #stride := stride
                 .& #pad := padding
                 .& #dilate := dilation
                 .& #layout := Just (EnumType (Proxy :: Proxy "NCHW"))
                 .& #num_group := 1
                 .& #no_bias := not _conv_bias
                 .& Nil
         in Conv2D <$> init scope args' initializer

    forward (Conv2D base) inp = forward base inp

    parameters (Conv2D base) = parameters base
