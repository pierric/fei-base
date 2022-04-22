{-# LANGUAGE CPP                    #-}
{-# LANGUAGE ConstraintKinds        #-}
{-# LANGUAGE DataKinds              #-}
{-# LANGUAGE FlexibleInstances      #-}
{-# LANGUAGE MultiParamTypeClasses  #-}
{-# LANGUAGE PolyKinds              #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE TypeOperators          #-}
{-# LANGUAGE UndecidableInstances   #-}
module MXNet.Base.Types where

import           Data.Proxy           (Proxy (..))
import           GHC.OverloadedLabels
import           GHC.TypeLits         (KnownSymbol, Symbol)
import           RIO
import           RIO.Vector.Unboxed   (Unbox)

import           MXNet.Base.Core.Enum (InEnum, Insert, enumWeaken)

data Context = Context
    { _device_type :: Int
    , _device_id   :: Int
    }
    deriving (Eq, Show)

class (Storable a,
       Unbox a,
       Show a,
       Eq a,
       Read a,
       KnownSymbol (DTypeName a)) => DType a where
    type DTypeName a = (b :: Symbol) | b -> a
    typename :: a -> Proxy (DTypeName a)
    typename a = Proxy

    flag :: a -> Int

instance a ~ b => IsLabel a (Proxy b) where
  fromLabel = Proxy

instance DType Float where
    type DTypeName Float = "float32"
    flag _ = 0    -- mshadow::kFloat32

instance DType Double where
    type DTypeName Double = "float64"
    flag _ = 1    -- mshadow::kFloat64

instance DType Word8 where
    type DTypeName Word8 = "uint8"
    flag _ = 3    -- mshadow::kUint8

instance DType Int32 where
    type DTypeName Int32 = "int32"
    flag _ = 4    -- mshadow::kInt32

instance DType Int64 where
    type DTypeName Int64 = "int64"
    flag _ = 6    -- mshadow::kInt64

instance DType Bool where
    type DTypeName Bool = "bool"
    flag _ = 7    -- mshadow::kBool

type FloatDType   a = (RealFloat a, DType a, InEnum (DTypeName a) FloatDTypes)
type NumericDType a = (Num a, DType a, InEnum (DTypeName a) NumericDTypes )

contextCPU :: Context
contextCPU = Context 1 0

contextPinnedCPU :: Context
contextPinnedCPU = Context 3 0

contextGPU0 :: Context
contextGPU0 = Context 2 0

class ForeignData a where
    touch :: a -> IO ()

type Shape = [Int]

data ReqType = ReqNull
    | ReqWrite
    | ReqInplace
    | ReqAdd
    deriving (Bounded, Enum, Show)

type BasicNumericDTypes = '["float16", "float32", "float64", "int32", "int64", "int8", "uint8"]
type BasicFloatDTypes   = '["float16" ,"float32", "float64"]

#if MXNET_VERSION < 10700
type NumericDTypes = BasicNumericDTypes
type FloatDTypes   = BasicFloatDTypes
type AllDTypes     = Insert "bool" BasicNumericDTypes
#else
type NumericDTypes = Insert "bfloat16" BasicNumericDTypes
type FloatDTypes   = Insert "bfloat16" BasicFloatDTypes
type AllDTypes     = Insert "bfloat16" (Insert "bool" BasicNumericDTypes)
#endif
