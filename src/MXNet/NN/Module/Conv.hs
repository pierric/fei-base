{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE ImplicitParams       #-}
{-# LANGUAGE RecordWildCards      #-}
{-# LANGUAGE TypeApplications     #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fplugin=Data.Record.Anon.Plugin#-}
module MXNet.NN.Module.Conv where

import           Data.Default.Class
import           Data.Hashable
import           Data.Record.Anon
import           Data.Record.Anon.Simple     (Record)
import qualified Data.Record.Anon.Simple     as Anon
import           GHC.TypeLits
import           RIO
import qualified RIO.HashMap                 as M
import qualified RIO.HashMap.Partial         as M
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

data ConvParamEnums = ConvWeights | ConvBias
    deriving (Eq, Show, Enum, Typeable)

instance Hashable ConvParamEnums where
    hashWithSalt = hashUsing fromEnum

data ConvBase u = ConvBase (Record (FieldsExc (O.ParameterList_Convolution NDArray u) '["_data", "bias", "weight"])) (GenericModule ConvParamEnums u)

instance DType u => Module (ConvBase u) where

    type ModuleDType  (ConvBase u) = u
    type ModuleArgs   (ConvBase u) = Record (FieldsExc (O.ParameterList_Convolution NDArray u) '["_data", "bias", "weight"])
    type ModuleInput  (ConvBase u) = NDArray u
    type ModuleOutput (ConvBase u) = NDArray u

    init scope args initializer = do
        let default_initializer = M.fromList
                [(scope :> ConvWeights, initEmpty),
                 (scope :> ConvBias, initZeros)]
        ConvBase args <$> init scope () (M.union initializer default_initializer)

    forward (ConvBase args generic) inp = do
        let scope = _gmodule_scope generic
        pt <- getOrInitParams generic $ \initializer -> do
            [batch_size, n_channels] <- take 2 <$> ndshape inp
            let [kh, kw]   = Anon.get #kernel args
                num_filter = Anon.get #num_filter args
            bias    <- makeEmptyNDArray [num_filter] ?device
            weights <- makeEmptyNDArray [num_filter, n_channels, kh, kw] ?device
            initializer M.! (scope :> ConvBias)    $ bias
            initializer M.! (scope :> ConvWeights) $ weights
            attachGradient bias ReqWrite
            attachGradient weights ReqWrite
            return $ M.fromList [(ConvBias, bias), (ConvWeights, weights)]

        prim O._Convolution $
            Anon.insert #_data inp $
            Anon.insert #bias   (Just $ pt M.! ConvBias) $
            Anon.insert #weight (Just $ pt M.! ConvWeights)
            args

    parameters (ConvBase _ generic) = parameters generic

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
    type ModuleDType  (Conv2D u) = u
    type ModuleArgs   (Conv2D u) = ConvArgs
    type ModuleInput  (Conv2D u) = NDArray u
    type ModuleOutput (Conv2D u) = NDArray u

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
