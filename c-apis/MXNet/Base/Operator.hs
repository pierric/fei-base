{-# LANGUAGE OverloadedLabels  #-}
{-# LANGUAGE PolyKinds, DataKinds, TypeFamilies #-}
{-# LANGUAGE FlexibleInstances, MultiParamTypeClasses #-}
{-# LANGUAGE GADTs, TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.Base.Operator where

import GHC.OverloadedLabels
import GHC.TypeLits
import GHC.Exts (Constraint)
import Data.List (intersperse)

data Proxy (s :: Symbol) = Proxy

instance a ~ b => IsLabel a (Proxy b) where
  fromLabel = Proxy

data EnumType (e :: [Symbol]) where
  EnumType :: (KnownSymbol v, HasEnum v e) => Proxy v -> EnumType e

type family HasEnum v e :: Constraint where
  HasEnum v e = IfThenElse (HasElement v e) (() :: Constraint) (TypeError (Text "\"" :<>: Text v :<>: Text "\" is not a valid value for the enum: [" :<>: FormatEnum e :<>: Text "]"))

type family FormatEnum (l :: [Symbol]) :: ErrorMessage where
  FormatEnum (s ': m ': n) = Text s :<>: Text ", " :<>: FormatEnum (m ': n)
  FormatEnum (s ': '[]) = Text s
  FormatEnum ('[]) = Text ""

instance (KnownSymbol v, HasEnum v e) => IsLabel v (EnumType e) where
    fromLabel = EnumType (Proxy :: Proxy v)
  
----
type family ParameterList (s :: Symbol) :: [(Symbol, Attr)]

data Attr where
  AttrReq :: (a :: *) -> Attr
  AttrOpt :: (a :: *) -> Attr

type family ParameterType (a :: Attr) (t :: *) :: Constraint where
  ParameterType (AttrReq a) t = a ~ t
  ParameterType (AttrOpt a) t = a ~ t

type family FilterRequired (pl :: [(k, Attr)]) :: [k] where
  FilterRequired '[] = '[]
  FilterRequired ('(s, AttrReq t) ': pl) = s ': FilterRequired pl
  FilterRequired (_ ': pl) = FilterRequired pl

type family ResolveParameter (s :: Symbol) (k :: Symbol) :: Attr where
  ResolveParameter s k = FindKey k (ParameterList s) (Text "Parameter '" :<>: 
                              Text k :<>:
                              Text " not found")

type family FindKey (s :: Symbol) (l :: [(Symbol, k)]) (e :: ErrorMessage) :: k where
  FindKey s ('(s,i) ': _) _ = i
  FindKey s ('(z,_) ': n) e = FindKey s n e
  FindKey s '[] e = TypeError e

----

data ArgOf s k v where
  (:=) :: (info ~ ResolveParameter s k, ParameterType info v) => Proxy k -> v -> ArgOf s k v

data HMap (s :: Symbol) (kvs :: [*]) where
  Nil :: HMap s '[]
  Cons :: kv -> HMap s kvs -> HMap s (kv ': kvs)

(.&) :: ArgOf s k v -> HMap s kvs -> HMap s (ArgOf s k v ': kvs)
kv .& other = Cons kv other

infixr 8 .& 

----
class Value a where
  showValue :: a -> String

instance Value (EnumType e) where
  showValue (EnumType v) = symbolVal v

instance Value Int where
  showValue = show

instance Value Bool where
  showValue = show

instance Value Float where
  showValue = show

instance Value Double where
  showValue = show

instance Value a => Value (Maybe a) where
  showValue Nothing = "None"
  showValue (Just a) = showValue a
  
instance Value a => Value [a] where
  showValue as = "[" ++ concat (intersperse "," (map showValue as)) ++ "]"

class DumpHMap a where
  dump :: a -> [(String, String)]

instance DumpHMap (HMap s '[]) where
  dump = const []

instance (DumpHMap (HMap s kvs), KnownSymbol k, Value v) => DumpHMap (HMap s (ArgOf s k v ': kvs)) where
  dump (Cons (k := v) kvs) = (symbolVal k, showValue v) : dump kvs

----
type family Subset (s1 :: [Symbol]) (s2 :: [(Symbol, *)]) :: Constraint where
  Subset '[] _ = ()
  Subset (a ': s1) s2 = ( IfThenElse (HasKey a s2) (() :: Constraint) (TypeError (Text "Argument '" :<>: Text a :<>: Text "' is required."))
                        , Subset s1 s2)

type family AsKVs (a :: [*]) :: [(Symbol, *)] where
  AsKVs (ArgOf s k v ': args) = '(k, v) ': AsKVs args
  AsKVs '[] = '[]

type family Fullfilled (s :: Symbol) (args :: [*]) where
  Fullfilled s args = Subset (FilterRequired (ParameterList s)) (AsKVs args)


----
type family HasElement (s :: k) (l :: [k]) :: Bool where
  HasElement s (s ': _) = True
  HasElement s (z ': n) = HasElement s n
  HasElement s '[] = False

type family HasKey (s :: Symbol) (l :: [(Symbol, *)]) :: Bool where
  HasKey s ('(s,_) ': _) = True
  HasKey s ('(z,_) ': n) = HasKey s n
  HasKey s '[] = False

type family IfThenElse (b :: Bool) (t :: k) (f :: k) :: k where
  IfThenElse True  t f = t
  IfThenElse False t f = f

-------------------------------------

type instance ParameterList "fn" = [
  '("a", AttrReq Int), 
  '("b", AttrOpt String), 
  '("c", AttrReq (EnumType '["c1","c2"])),
  '("d", AttrOpt (Maybe (EnumType '["c1","c2"])))
  ]
args1 :: HMap "fn" _
args1 = Nil

args2 :: HMap "fn" _
args2 = #a := 3 .& Nil

args3 :: HMap "fn" _
args3 = #a := 3 .& #b := "Hello" .& Nil

args4 :: HMap "fn" _
args4 = #a := 3 .& #c := #c1 .& Nil

args5 :: HMap "fn" _
args5 = #a := 3 .& #c := #c1 .& #d := Just #c2 .& Nil

fn :: Fullfilled "fn" args => HMap "fn" args -> Bool
fn args = True