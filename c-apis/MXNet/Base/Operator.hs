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
import Data.Proxy
import Data.List (intersperse)
import MXNet.Base.HMap

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

type family MatchHeadArgOf s k v kvs where
  MatchHeadArgOf s k v (ArgOf s k v ': kvs) = True
  MatchHeadArgOf s k v (_ ': kvs) = False

instance Pair (ArgOf s) where
  key   (k := v) = k
  value (k := v) = v
  type MatchHead (ArgOf s) k v kvs = MatchHeadArgOf s k v kvs

type ArgsHMap s kvs = HMap (ArgOf s) kvs
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

class Dump a where
  dump :: a -> [(String, String)]

instance Dump (ArgsHMap s '[]) where
  dump = const []

instance (Dump (ArgsHMap s kvs), KnownSymbol k, Value v) => Dump (ArgsHMap s (ArgOf s k v ': kvs)) where
  dump (Cons (k := v) kvs) = (symbolVal k, showValue v) : dump kvs

----
type family Subset (s1 :: [(Symbol, *)]) (s2 :: [(Symbol, *)]) :: Constraint where
  Subset '[] _ = ()
  Subset ('(a, t) ': s1) s2 = ( IfThenElse (HasElement '(a,t) s2) 
                                  (() :: Constraint) 
                                  (TypeError (Text "Argument '" :<>: Text a :<>: Text "' is required."))
                              , Subset s1 s2)
  Subset a b = TypeError (Text "xx")

type family AsKVs (a :: [*]) :: [(Symbol, *)] where
  AsKVs (ArgOf s k v ': args) = '(k, v) ': AsKVs args
  AsKVs '[] = '[]

type family GenAccess s kvs (req :: [(Symbol, *)]) :: Constraint where
  GenAccess s kvs '[] = ()
  GenAccess s kvs ('(k, v) ': req) = (Access (MatchHeadArgOf s k v kvs) (ArgOf s) k v kvs, GenAccess s kvs req)

type family GenQuery  s kvs (req :: [(Symbol, *)]) :: Constraint where
  GenQuery  s kvs '[]  = ()
  GenQuery  s kvs ('(k, v) ': req) = (Query  (InHMap (ArgOf s) k kvs)   (ArgOf s) k v kvs, GenQuery  s kvs req)


type family FilterRequired (pl :: [(k, Attr)]) :: [(k, *)] where
  FilterRequired '[] = '[]
  FilterRequired ('(s, AttrReq t) ': pl) = '(s,t) ': FilterRequired pl
  FilterRequired (_ ': pl) = FilterRequired pl


type family AllArgs (pl :: [(k, Attr)]) :: [(k, *)] where
  AllArgs '[] = '[]
  AllArgs ('(s, AttrReq t) ': pl) = '(s,t) ': AllArgs pl
  AllArgs ('(s, AttrOpt t) ': pl) = '(s,t) ': AllArgs pl

type family Fullfilled (s :: Symbol) (args :: [*]) :: Constraint where
  Fullfilled s args = ( Subset ( FilterRequired (ParameterList s)) (AsKVs args)
                      , GenAccess s args (FilterRequired (ParameterList s))
                      , GenQuery  s args (AllArgs (ParameterList s)))

----
type family HasElement (s :: k) (l :: [k]) :: Bool where
  HasElement s (s ': _) = True
  HasElement s (z ': n) = HasElement s n
  HasElement s '[] = False

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
args1 :: ArgsHMap "fn" _
args1 = Nil

args2 :: ArgsHMap "fn" _
args2 = #a := 3 .& Nil

args3 :: ArgsHMap "fn" _
args3 = #a := 3 .& #b := "Hello" .& Nil

args4 :: ArgsHMap "fn" _
args4 = #a := 3 .& #c := #c1 .& Nil

args5 :: ArgsHMap "fn" _
args5 = #a := 3 .& #c := #c1 .& #d := Just #c2 .& Nil

fn :: Fullfilled "fn" args => ArgsHMap "fn" args -> _
fn args = args ! #a
