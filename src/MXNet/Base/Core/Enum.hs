{-# LANGUAGE AllowAmbiguousTypes   #-}
{-# LANGUAGE ConstraintKinds       #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}
module MXNet.Base.Core.Enum where

import           Data.Constraint
import           Data.Proxy
import           Data.Type.Bool
import           Data.Type.Equality
import           GHC.OverloadedLabels
import           GHC.TypeLits         (CmpSymbol, ErrorMessage (..),
                                       KnownSymbol, Symbol)
import           RIO                  (Bool (..), Ordering (..), ($))
import           Unsafe.Coerce        (unsafeCoerce)

type Enum = [Symbol]

type family Member (s :: Symbol) (ss :: Enum) :: Bool where
    Member s '[] = False
    Member s (s0 ': ss) = If (CmpSymbol s s0 == EQ) True (Member s ss)

type InEnum s0 ss = Member s0 ss ~ True

type Insert (s :: Symbol) (ss :: Enum) = If (Member s ss) ss (s ': ss)

class Subset (ss :: Enum) (ts :: Enum)

instance Subset '[] ts
instance (Subset ss ts, Member s0 ts ~ True) => Subset (s0 ': ss) ts

type family (ss :: Enum) :⊆ (ts :: Enum) :: Bool where
    '[] :⊆ ts = True
    (s0 ': ss) :⊆ ts = Member s0 ts && ss :⊆ ts

-- | InEnum has its weakness. When having a `InEnum s e1`, the compier
--   cannot deduce the fact that `InEnum s e2`, where e2 is an extension
--   of e1. The utility function is a proof of the fact.
--
--   Note: delay to the future, only a fake proof at the moment.
--
enumWeaken :: forall s t x. (InEnum x s, Subset s t) :- InEnum x t
enumWeaken = Sub $ unsafeCoerce (Dict :: Dict ())

type family FormatEnum (e :: Enum) :: ErrorMessage where
  FormatEnum (s ': m ': n) = Text s :<>: Text ", " :<>: FormatEnum (m ': n)
  FormatEnum (s ': '[]) = Text s
  FormatEnum '[] = Text ""

data EnumType (e :: [Symbol]) where
  EnumType :: (KnownSymbol v, InEnum v e) => Proxy v -> EnumType e

instance (KnownSymbol v, InEnum v e) => IsLabel v (EnumType e) where
    fromLabel = EnumType (Proxy :: Proxy v)

