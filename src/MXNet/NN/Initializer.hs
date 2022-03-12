{-# LANGUAGE TypeApplications #-}
module MXNet.NN.Initializer where

import           RIO

import           MXNet.Base                  (ArgOf (..), DType, HMap (..),
                                              NDArray, ndshape, (.&))
import qualified MXNet.Base.Operators.Tensor as O

void' :: IO [NDArray a] -> IO ()
void' = void

type Initializer p t = p -> NDArray t -> IO ()

initConstant :: forall t. DType t => Float -> NDArray t -> IO ()
initConstant val arr = void' @t $ O.__set_value (#src := val .& Nil) (Just [arr])

initEmpty, initZeros :: DType t => NDArray t -> IO ()
initEmpty _ = return ()
initZeros   = initConstant 0

initNormal, initUniform :: forall t. DType t => Float -> NDArray t -> IO ()
initNormal sigma arr = void' @t $ O.__random_normal (#loc := (0 :: Float) .& #scale  := sigma .& Nil) (Just [arr])
initUniform sca arr  = void' @t $ O.__random_uniform (#low := (-sca) .& #high   := sca .& Nil) (Just [arr])

data XavierFactor = XavierAvg
    | XavierIn
    | XavierOut
    deriving (Show, Read)

data XavierRandom = XavierUniform
    | XavierGaussian
    deriving (Show, Read)

initXavier :: forall t. DType t => Float -> XavierRandom -> XavierFactor -> NDArray t -> IO ()
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
