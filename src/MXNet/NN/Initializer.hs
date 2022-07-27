{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -fplugin=Data.Record.Anon.Plugin#-}
module MXNet.NN.Initializer where

import qualified Data.Record.Anon.Simple     as Anon
import           RIO

import           MXNet.Base                  (DType, NDArray, ndshape)
import qualified MXNet.Base.Operators.Tensor as I

void' :: IO [NDArray a] -> IO ()
void' = void

type Initializer t = NDArray t -> IO ()

initConstant :: forall t. DType t => Float -> Initializer t
initConstant val arr = void' @t $ I.__set_value ANON{src = Just val} (Just [arr])

initEmpty, initZeros, initOnes :: DType t => Initializer t
initEmpty _ = return ()
initZeros   = initConstant 0
initOnes    = initConstant 1

initNormal, initUniform :: forall t. DType t => Float -> Initializer t
initNormal sigma arr = void' @t $ I.__random_normal ANON{loc = Just 0, scale= Just sigma} (Just [arr])
initUniform sca arr  = void' @t $ I.__random_uniform ANON{low = Just (-sca), high = Just sca} (Just [arr])

data XavierFactor = XavierAvg
    | XavierIn
    | XavierOut
    deriving (Show, Read)

data XavierRandom = XavierUniform
    | XavierGaussian
    deriving (Show, Read)

initXavier :: forall t. DType t => Float -> XavierRandom -> XavierFactor -> Initializer t
initXavier magnitude distr factor arr = do
        shp <- ndshape arr
        if length shp < 2
        then error $ concat ["invalid shape ", show shp, " for xavier initializer"]
        else do
            let ofan : dims = shp
                ifan = product dims
                scale = case factor of
                          XavierIn  -> sqrt (magnitude / fromIntegral ifan)
                          XavierOut -> sqrt (magnitude / fromIntegral ofan)
                          XavierAvg -> sqrt (magnitude * 2.0 / fromIntegral (ifan + ofan))
            case distr of
              XavierUniform  -> initUniform scale arr
              XavierGaussian -> initNormal scale arr
