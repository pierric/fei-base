{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
module MXNet.Base.Types where

import RIO
import RIO.Vector.Unboxed (Unbox)
import Data.Proxy (Proxy(..))
import GHC.TypeLits (Symbol, KnownSymbol)

data Context = Context { _device_type :: Int, _device_id :: Int }
  deriving (Eq, Show)

class (Storable a, Unbox a, Show a, Num a, Floating a, Real a, RealFrac a, KnownSymbol (DTypeName a)) => DType a where
  type DTypeName a :: Symbol
  typename :: a -> Proxy (DTypeName a)
  typename a = Proxy

  flag :: a -> Int

instance DType Float where
  type DTypeName Float = "float32"
  flag _ = 0    -- mshadow::kFloat32

instance DType Double where
  type DTypeName Double = "float64"
  flag _ = 1    -- mshadow::kFloat64

contextCPU :: Context
contextCPU = Context 1 0

contextGPU0 :: Context
contextGPU0 = Context 2 0

class ForeignData a where
    touch :: a -> IO ()
