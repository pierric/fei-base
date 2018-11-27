{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
module MXNet.Base.Types where

import Foreign.Storable (Storable)
import Data.Proxy (Proxy(..))
import GHC.TypeLits (Symbol, KnownSymbol)

data Context = Context { _device_type :: Int, _device_id :: Int }
  deriving (Eq, Show)

class (Storable a, Show a, Num a, Floating a, Real a, KnownSymbol (DTypeName a)) => DType a where
  type DTypeName a :: Symbol
  typename :: a -> Proxy (DTypeName a)
  typename a = Proxy

instance DType Float where
  type DTypeName Float = "float32"

instance DType Double where
  type DTypeName Double = "float64"

contextCPU :: Context
contextCPU = Context 1 0

contextGPU0 :: Context
contextGPU0 = Context 2 0
