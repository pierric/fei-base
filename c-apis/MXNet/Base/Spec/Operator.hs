{-# LANGUAGE AllowAmbiguousTypes    #-}
{-# LANGUAGE ConstraintKinds        #-}
{-# LANGUAGE DataKinds              #-}
{-# LANGUAGE FlexibleInstances      #-}
{-# LANGUAGE GADTs                  #-}
{-# LANGUAGE MultiParamTypeClasses  #-}
{-# LANGUAGE OverloadedLabels       #-}
{-# LANGUAGE PartialTypeSignatures  #-}
{-# LANGUAGE PolyKinds              #-}
{-# LANGUAGE ScopedTypeVariables    #-}
{-# LANGUAGE TypeApplications       #-}
{-# LANGUAGE TypeFamilies           #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE TypeOperators          #-}
{-# LANGUAGE UndecidableInstances   #-}
module MXNet.Base.Spec.Operator where

import           Data.Constraint
import           Data.Proxy
import           Data.Typeable        (eqT, (:~:) (..))
import           GHC.Exts             (Constraint)
import           GHC.OverloadedLabels
import           GHC.TypeLits
import           RIO                  hiding (Text)
import           RIO.List             (intersperse)
import qualified RIO.Text             as RT
import           Type.Set
import           Unsafe.Coerce        (unsafeCoerce)

import           MXNet.Base.Spec.HMap
import           MXNet.Base.Types

instance a ~ b => IsLabel a (Proxy b) where
  fromLabel = Proxy

data EnumType (e :: [Symbol]) where
  EnumType :: (KnownSymbol v, InEnum v (SetFromList e)) => Proxy v -> EnumType e

type family FormatEnum (l :: [Symbol]) :: ErrorMessage where
  FormatEnum (s ': m ': n) = Text s :<>: Text ", " :<>: FormatEnum (m ': n)
  FormatEnum (s ': '[]) = Text s
  FormatEnum ('[]) = Text ""

instance (KnownSymbol v, InEnum v (SetFromList e)) => IsLabel v (EnumType e) where
    fromLabel = EnumType (Proxy :: Proxy v)

----
type family ParameterList (s :: Symbol) (t :: kind) :: [(Symbol, Attr)]

data Attr where
  AttrReq :: (a :: *) -> Attr
  AttrOpt :: (a :: *) -> Attr

type family ParameterType (a :: Attr) :: * where
  ParameterType (AttrReq a) = a
  ParameterType (AttrOpt a) = a

type family ResolveParameter (s :: Symbol) (t :: kind) (k :: Symbol) :: Attr where
  ResolveParameter s t k = FindKey k (ParameterList s t) (Text "Parameter '" :<>:
                              Text k :<>:
                              Text " not found")

type family FindKey (s :: Symbol) (l :: [(Symbol, kind)]) (e :: ErrorMessage) :: kind where
  FindKey s ('(s,i) ': _) _ = i
  FindKey s ('(z,_) ': n) e = FindKey s n e
  FindKey s '[] e = TypeError e

----

data ArgOf s (t :: kind) k v where
  (:=) :: (info ~ ResolveParameter s t k) => Proxy k -> ParameterType info -> ArgOf s t k (ParameterType info)
  -- | (:≅) is an alternative of (:=) that bypasses the type check
  (:≅) :: Proxy k -> a -> ArgOf s t k a

instance Pair (ArgOf s t) where
  key   (k := v) = k
  key   (k :≅ v) = k
  value (k := v) = v
  value (k :≅ v) = v

infix 5 !, !?
infix 1 :=, :≅

(!) :: Access (MatchHead (ArgOf s t) k v kvs) (ArgOf s t) k v kvs
  => ArgsHMap s t kvs -> Proxy k -> v
(!) = get

(!?) :: (ParameterType (ResolveParameter s t k) ~ v, Query (MatchHead (ArgOf s t) k v kvs) (ArgOf s t) k v kvs)
  => ArgsHMap s t kvs -> Proxy k -> Maybe v
(!?) = query

type ArgsHMap s (t :: kind) kvs = HMap (ArgOf s t) kvs
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

instance Dump (ArgsHMap s t '[]) where
  dump = const []

instance (Dump (ArgsHMap s t kvs), KnownSymbol k, Value v) => Dump (ArgsHMap s t (ArgOf s t k v ': kvs)) where
  dump (Cons (k := v) kvs) = (RT.pack $ symbolVal k, showValue v) : dump kvs
  dump (Cons (k :≅ v) kvs) = (RT.pack $ symbolVal k, showValue v) : dump kvs

----
type family KvSubset (s1 :: [(Symbol, *)]) (s2 :: [(Symbol, *)]) :: Constraint where
  KvSubset '[] _ = ()
  KvSubset ('(a, t) ': s1) s2 = ( IfThenElse (HasElement '(a,t) s2)
                                  (() :: Constraint)
                                  (TypeError (Text "Argument '" :<>: Text a :<>: Text "' is required."))
                              , KvSubset s1 s2)
  KvSubset a b = TypeError (Text "xx")

type family AsKVs (a :: [*]) :: [(Symbol, *)] where
  AsKVs (ArgOf s t k v ': args) = '(k, v) ': AsKVs args
  AsKVs '[] = '[]

type family GenAccess s (t :: kind) kvs (req :: [(Symbol, *)]) :: Constraint where
  GenAccess s t kvs '[] = ()
  GenAccess s t kvs ('(k, v) ': req) = (Access (MatchHead (ArgOf s t) k v kvs) (ArgOf s t) k v kvs, GenAccess s t kvs req)

type family GenQuery  s (t :: kind) kvs (req :: [(Symbol, *)]) :: Constraint where
  GenQuery  s t kvs '[]  = ()
  GenQuery  s t kvs ('(k, v) ': req) = (Query  (MatchHead (ArgOf s t) k v kvs) (ArgOf s t) k v kvs, GenQuery  s t kvs req)


type family FilterRequired (pl :: [(k, Attr)]) :: [(k, *)] where
  FilterRequired '[] = '[]
  FilterRequired ('(s, AttrReq t) ': pl) = '(s,t) ': FilterRequired pl
  FilterRequired (_ ': pl) = FilterRequired pl


type family AllArgs (pl :: [(k, Attr)]) :: [(k, *)] where
  AllArgs '[] = '[]
  AllArgs ('(s, AttrReq t) ': pl) = '(s,t) ': AllArgs pl
  AllArgs ('(s, AttrOpt t) ': pl) = '(s,t) ': AllArgs pl

type family Fullfilled (s :: Symbol) (t :: kind) (args :: [*]) :: Constraint where
  Fullfilled s t args = ( KvSubset ( FilterRequired (ParameterList s t)) (AsKVs args)
                        , GenAccess s t args (FilterRequired (ParameterList s t))
                        , GenQuery  s t args (AllArgs (ParameterList s t)))

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

type family HasArgs (s :: Symbol) t (args :: [*]) (k :: [Symbol]) :: Constraint where
  HasArgs s t args '[] = ()
  HasArgs s t args (k0 ': ks) = (HasArgsGen (ArgOf s t) (ResolveParameter s t k0) k0 args, HasArgs s t args ks)

type family WithoutArgsGen p t k args :: Constraint where
  WithoutArgsGen p t k args = (Query (MatchHead p k t args) p k t args
                              ,HasElement '(k, t) (AsKVs args) ~ False)

type family WithoutArgs (s :: Symbol) t (args :: [*]) (k :: [Symbol]) :: Constraint where
  WithoutArgs s t args '[] = ()
  WithoutArgs s t args (k0 ': ks) = (WithoutArgsGen (ArgOf s t) (ParameterType (ResolveParameter s t k0)) k0 args, WithoutArgs s t args ks)
----
type family HasElement (s :: kind) (l :: [kind]) :: Bool where
  HasElement s (s ': _) = True
  HasElement s (z ': n) = HasElement s n
  HasElement s '[] = False

type family IfThenElse (b :: Bool) (t :: kind) (f :: kind) :: kind where
  IfThenElse True  t f = t
  IfThenElse False t f = f

-------------------------------------
{-

type instance ParameterList "fn" t = [
  '("a", AttrReq Int),
  '("b", AttrOpt String),
  '("c", AttrReq (EnumType '["c1","c2"])),
  '("d", AttrOpt (Maybe (EnumType '["c1","c2"])))
  ]
args1 :: ArgsHMap "fn" _ _
args1 = Nil

args2 :: ArgsHMap "fn" _ _
args2 = #a := 3 .& Nil

args3 :: ArgsHMap "fn" _ _
args3 = #a := 3 .& #b := "Hello" .& Nil

args4 :: ArgsHMap "fn" _ _
args4 = #a := 3 .& #c := #c1 .& Nil

args5 :: ArgsHMap "fn" _ _
args5 = #a := 3 .& #c := #c1 .& #d := Just #c2 .& Nil

fn1 :: Fullfilled "fn" t args => ArgsHMap "fn" t args -> _
fn1 args = args !? #b

fn2 :: GenQuery "fn" t args '[ '("b", String), '("d", (Maybe (EnumType '["c1","c2"])))]
    => ArgsHMap "fn" t args -> _
fn2 args = fn1 (#a := 3 .& #c := #c1 .& args)

fn3 :: (HasArgs "fn" t args '["b", "c", "d"]) => ArgsHMap "fn" t args -> _
fn3 args = fn1 (#a := 3 .& args)

fn4 :: (HasArgs "fn" t args '["c", "b", "d"], WithoutArgs "fn" t args '["a"]) => ArgsHMap "fn" t args -> _
fn4 args = fn1 (#a := 3 .& args)
-}
