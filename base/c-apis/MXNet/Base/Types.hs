{-# LANGUAGE ConstraintKinds #-}
module MXNet.Base.Types where

import Foreign.Storable (Storable)

data Device = Device { _device_type :: Int, _device_id :: Int }
  deriving (Eq, Show)

type DType a = (Storable a, Num a, Floating a, Real a)