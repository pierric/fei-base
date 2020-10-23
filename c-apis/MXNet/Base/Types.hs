{-# LANGUAGE DataKinds    #-}
{-# LANGUAGE TypeFamilies #-}
module MXNet.Base.Types where

import           Data.Proxy         (Proxy (..))
import           GHC.TypeLits       (KnownSymbol, Symbol)
import           RIO
import           RIO.Vector.Unboxed (Unbox)

data Context = Context
    { _device_type :: Int
    , _device_id   :: Int
    }
    deriving (Eq, Show)

class (Storable a,
       Unbox a,
       Show a,
       Eq a,
       Num a,
       Read a,
       KnownSymbol (DTypeName a)) => DType a where
    type DTypeName a :: Symbol
    typename :: a -> Proxy (DTypeName a)
    typename a = Proxy

    flag :: a -> Int


class (RealFloat a, DType a) => FloatDType a

instance DType Float where
    type DTypeName Float = "float32"
    flag _ = 0    -- mshadow::kFloat32

instance DType Double where
    type DTypeName Double = "float64"
    flag _ = 1    -- mshadow::kFloat64

instance FloatDType Float
instance FloatDType Double

instance DType Word8 where
    type DTypeName Word8 = "uint8"
    flag _ = 3    -- mshadow::kUint8

contextCPU :: Context
contextCPU = Context 1 0

contextGPU0 :: Context
contextGPU0 = Context 2 0

class ForeignData a where
    touch :: a -> IO ()
