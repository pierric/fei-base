{-# LANGUAGE ConstraintKinds       #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLabels      #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}
module MXNet.Base.Spec.Operator where

import           Data.Constraint
import           Data.Proxy
import           GHC.Exts             (Constraint)
import           GHC.OverloadedLabels
import           GHC.TypeLits
import           MXNet.Base.Spec.HMap
import           RIO                  hiding (Text)
import           RIO.List             (intersperse)
import qualified RIO.Text             as RT

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

type family ParameterType (a :: Attr) :: * where
  ParameterType (AttrReq a) = a
  ParameterType (AttrOpt a) = a

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
  (:=) :: (info ~ ResolveParameter s k) => Proxy k -> ParameterType info -> ArgOf s k (ParameterType info)
  (:≅) :: Proxy k -> a -> ArgOf s k a

instance Pair (ArgOf s) where
  key   (k := v) = k
  key   (k :≅ v) = k
  value (k := v) = v
  value (k :≅ v) = v

infix 5 !, !?
infix 1 :=, :≅

(!) :: Access (MatchHead (ArgOf s) k v kvs) (ArgOf s) k v kvs
  => ArgsHMap s kvs -> Proxy k -> v
(!) = get

(!?) :: (ParameterType (ResolveParameter s k) ~ v, Query (MatchHead (ArgOf s) k v kvs) (ArgOf s) k v kvs)
  => ArgsHMap s kvs -> Proxy k -> Maybe v
(!?) = query

type ArgsHMap s kvs = HMap (ArgOf s) kvs
----
class Value a where
  showValue :: a -> RT.Text

instance Value (EnumType e) where
  showValue (EnumType v) = RT.pack $ symbolVal v

instance Value Int where
  showValue = tshow

instance Value Bool where
  showValue = tshow

instance Value Float where
  showValue = tshow

instance Value Double where
  showValue = tshow

instance Value RT.Text where
  showValue = id

instance Value a => Value (Maybe a) where
  showValue Nothing  = "None"
  showValue (Just a) = showValue a

instance (Value a, Value b) => Value (a, b) where
  showValue (a, b) = RT.concat $ ["("] ++ [showValue a, ",", showValue b] ++ [")"]

instance ValueList (IsChar a) [a] => Value [a] where
  showValue = showValueList (Proxy :: Proxy (IsChar a))

class ValueList (str :: Bool) as where
  showValueList :: Proxy str -> as -> RT.Text

instance ValueList True String where
  showValueList _ = RT.pack

instance Value a => ValueList False [a] where
  showValueList _ as = RT.concat $ ["["] ++ intersperse "," (map showValue as) ++ ["]"]

type family IsChar a :: Bool where
  IsChar Char = True
  IsChar x = False

class Dump a where
  dump :: a -> [(RT.Text, RT.Text)]

instance Dump (ArgsHMap s '[]) where
  dump = const []

instance (Dump (ArgsHMap s kvs), KnownSymbol k, Value v) => Dump (ArgsHMap s (ArgOf s k v ': kvs)) where
  dump (Cons (k := v) kvs) = (RT.pack $ symbolVal k, showValue v) : dump kvs
  dump (Cons (k :≅ v) kvs) = (RT.pack $ symbolVal k, showValue v) : dump kvs

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
  GenAccess s kvs ('(k, v) ': req) = (Access (MatchHead (ArgOf s) k v kvs) (ArgOf s) k v kvs, GenAccess s kvs req)

type family GenQuery  s kvs (req :: [(Symbol, *)]) :: Constraint where
  GenQuery  s kvs '[]  = ()
  GenQuery  s kvs ('(k, v) ': req) = (Query  (MatchHead (ArgOf s) k v kvs) (ArgOf s) k v kvs, GenQuery  s kvs req)


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

-- type family HasOptArg (s :: Symbol) (args :: [*]) (k :: [Symbol]) :: Constraint where
--   HasOptArg s args '[] = ()
--   HasOptArg s args (k0 ': ks) = ( Query (MatchHead (ArgOf s) k0 (ParameterType (ResolveParameter s k0)) args)
--                                         (ArgOf s)
--                                         k0
--                                         (ParameterType (ResolveParameter s k0))
--                                         args
--                                 , HasOptArg s args ks)

-- type family HasReqArg (s :: Symbol) (args :: [*]) (k :: [Symbol]) :: Constraint where
--   HasReqArg s args '[] = ()
--   HasReqArg s args (k0 ': ks) = ( Access (MatchHead (ArgOf s) k0 (ParameterType (ResolveParameter s k0)) args)
--                                         (ArgOf s)
--                                         k0
--                                         (ParameterType (ResolveParameter s k0))
--                                         args
--                                 , HasElement '(k0, ParameterType (ResolveParameter s k0)) (AsKVs args) ~ True
--                                 , Query  (MatchHead (ArgOf s) k0 (ParameterType (ResolveParameter s k0)) args)
--                                         (ArgOf s)
--                                         k0
--                                         (ParameterType (ResolveParameter s k0))
--                                         args
--                                 , HasReqArg s args ks)

type family HasArgsGen p i k args :: Constraint where
  HasArgsGen p (AttrOpt t) k args = Query (MatchHead p k t args) p k t args
  HasArgsGen p (AttrReq t) k args = (Access (MatchHead p k t args) p k t args
                                    ,HasElement '(k, t) (AsKVs args) ~ True
                                    ,Query (MatchHead p k t args) p k t args)

type family HasArgs (s :: Symbol) (args :: [*]) (k :: [Symbol]) :: Constraint where
  HasArgs s args '[] = ()
  HasArgs s args (k0 ': ks) = (HasArgsGen (ArgOf s) (ResolveParameter s k0) k0 args, HasArgs s args ks)

type family WithoutArgsGen p t k args :: Constraint where
  WithoutArgsGen p t k args = (Query (MatchHead p k t args) p k t args
                              ,HasElement '(k, t) (AsKVs args) ~ False)

type family WithoutArgs (s :: Symbol) (args :: [*]) (k :: [Symbol]) :: Constraint where
  WithoutArgs s args '[] = ()
  WithoutArgs s args (k0 ': ks) = (WithoutArgsGen (ArgOf s) (ParameterType (ResolveParameter s k0)) k0 args, WithoutArgs s args ks)
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

fn1 :: Fullfilled "fn" args => ArgsHMap "fn" args -> _
fn1 args = args !? #b

fn2 :: GenQuery "fn" args '[ '("b", String), '("d", (Maybe (EnumType '["c1","c2"])))]
    => ArgsHMap "fn" args -> _
fn2 args = fn1 (#a := 3 .& #c := #c1 .& args)

fn3 :: (HasArgs "fn" args '["b", "c", "d"]) => ArgsHMap "fn" args -> _
fn3 args = fn1 (#a := 3 .& args)

fn4 :: (HasArgs "fn" args '["c", "b", "d"], WithoutArgs "fn" args '["a"]) => ArgsHMap "fn" args -> _
fn4 args = fn1 (#a := 3 .& args)
