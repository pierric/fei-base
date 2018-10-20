{-# LANGUAGE OverloadedLabels  #-}
{-# LANGUAGE PolyKinds, DataKinds, TypeFamilies #-}
{-# LANGUAGE FlexibleInstances, MultiParamTypeClasses #-}
{-# LANGUAGE GADTs, TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE ConstraintKinds #-}
module MXNet.Base.Operator where

import GHC.OverloadedLabels
import GHC.TypeLits
import GHC.Exts (Constraint)

data Proxy (s :: Symbol) = Proxy

instance a ~ b => IsLabel a (Proxy b) where
  fromLabel = Proxy

data EnumType (e :: [Symbol]) = EnumType 

type family HasEnum (s :: Symbol) (l :: [Symbol]) (e :: ErrorMessage) :: Constraint where
  HasEnum s (s ': _) _ = ()
  HasEnum s (z ': n) e = HasEnum s n e
  HasEnum s '[] e = TypeError e

type family FormatEnum (l :: [Symbol]) :: ErrorMessage where
  FormatEnum (s ': m ': n) = Text s :<>: Text ", " :<>: FormatEnum (m ': n)
  FormatEnum (s ': '[]) = Text s
  FormatEnum ('[]) = Text ""

instance HasEnum v e (Text "\"" :<>: Text v :<>: Text "\" is not a valid value for the enum: [" :<>: FormatEnum e :<>: Text "]") 
  => IsLabel v (EnumType e) where
    fromLabel = EnumType
  
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

data Arg (k :: Symbol) v = Arg (Proxy k) v

data HMap (s :: Symbol) (kvs :: [*]) where
  Nil :: HMap s '[]
  Cons :: kv -> HMap s kvs -> HMap s (kv ': kvs)

(.&) :: ArgOf s k v -> HMap s kvs -> HMap s (Arg k v ': kvs)
(k := v) .& other = Cons (Arg k v) other

infixr 8 .& 

----
type family Subset (s1 :: [Symbol]) (s2 :: [(Symbol, *)]) :: Constraint where
  Subset '[] _ = ()
  Subset (a ': s1) s2 = (HasKey a s2 (Text "Argument '" :<>: Text a :<>: Text "' is required."), Subset s1 s2)

type family HasKey (s :: Symbol) (l :: [(Symbol, *)]) (e :: ErrorMessage) :: Constraint where
  HasKey s ('(s,_) ': _) _ = ()
  HasKey s ('(z,_) ': n) e = HasKey s n e
  HasKey s '[] e = TypeError e

type family AsKVs (a :: [*]) :: [(Symbol, *)] where
  AsKVs (Arg k v ': args) = '(k, v) ': AsKVs args
  AsKVs '[] = '[]

type family Fullfilled (s :: Symbol) (args :: [*]) where
  Fullfilled s args = Subset (FilterRequired (ParameterList s)) (AsKVs args)

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
args5 = #a := 3 .& #c := #c1 .& #d := Just #c1 .& Nil

fn :: Fullfilled "fn" args => HMap "fn" args -> Bool
fn args = True