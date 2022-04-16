{-# LANGUAGE ConstraintKinds       #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}
{-# OPTIONS_GHC -fplugin=Data.Record.Anon.Plugin #-}
module MXNet.Base.Core.Spec where

import           Data.Record.Anon
import qualified Data.Record.Anon.Advanced as AAnon
import           Data.Record.Anon.Simple   (Record)
import qualified Data.Record.Anon.Simple   as Anon
import           Data.Type.Bool
import           Data.Type.Equality
import           GHC.Exts                  (Constraint)
import           GHC.TypeLits              (CmpSymbol, ErrorMessage ((:<>:)),
                                            Symbol, TypeError, symbolVal)
import qualified GHC.TypeLits              as Lit (ErrorMessage (..))
import           RIO
import           RIO.List                  (intersperse)
import qualified RIO.Text                  as RT

import           MXNet.Base.Core.Enum
import           MXNet.Base.Types          (DType)


data Attr where
    AttrReq :: (a :: *) -> Attr
    AttrOpt :: (a :: *) -> Attr

type family ParameterType (a :: Attr) :: * where
    ParameterType (AttrReq a) = a
    ParameterType (AttrOpt a) = Maybe a

type family ResolveParameter (pl :: [(Symbol, Attr)]) (key :: Symbol) :: Attr where
    ResolveParameter pl key = FindOrFail pl key (Lit.Text "Parameter '" :<>: Lit.Text key :<>: Lit.Text " not found")

type family FindOrFail (pl :: [(Symbol, Attr)]) (key :: Symbol) (e :: ErrorMessage) :: Attr where
  FindOrFail '[] _ e = TypeError e
  FindOrFail ('(s, a) ': pl) key e = If (CmpSymbol s key == EQ) a (FindOrFail pl key e)

-- type family FullfilledWalk pl rec (cc :: Constraint) :: Constraint where
--     FullfilledWalk '[] rec cc = cc
--     FullfilledWalk ('(name, AttrReq pt) ': pl) rec cc = FullfilledWalk pl rec (RowHasField name rec pt, cc)
--     FullfilledWalk ('(name, AttrOpt pt) ': pl) rec cc = FullfilledWalk pl rec (RowHasField name rec (Maybe pt), cc)
--
-- type Fullfilled pl rec = FullfilledWalk pl rec ()

-- type family HasKeysWalk pl rec (keys :: [Symbol]) (cc :: Constraint) :: Constraint where
--     HasKeysWalk pl rec '[] cc = cc
--     HasKeysWalk pl rec (k ': keys) cc = HasKeysWalk pl rec keys (RowHasField k rec (ParameterType (ResolveParameter pl k)), cc)
--
-- type HasKeys pl rec keys = HasKeysWalk pl rec keys ()
--
-- type family NotInRow (rec :: Row k) (key :: Symbol) :: Constraint where
--     NotInRow '[] key = ()
--     NotInRow ((name := _) ': rec) key = If (CmpSymbol name key == EQ) (TypeError (Text "Parameter '" :<>: Text key :<>: Text " should not be specified")) (NotInRow rec key)
--
-- type family NoKeysWalk (rec :: Row *) (keys :: [Symbol]) (cc :: Constraint) :: Constraint where
--     NoKeysWalk rec '[] cc = cc
--     NoKeysWalk rec (k ': keys) cc = NoKeysWalk rec keys (NotInRow rec k, cc)
--
-- type NoKeys rec keys = NoKeysWalk rec keys ()
--
type family FieldsFull (pl :: [(Symbol, Attr)]) :: Row * where
    FieldsFull '[] = '[]
    FieldsFull ('(s, AttrReq t) ': pl) = s := t ': FieldsFull pl
    FieldsFull ('(s, AttrOpt t) ': pl) = s := Maybe t ': FieldsFull pl

type ParamListFull pl = Record (FieldsFull pl)

type family FieldsInc (pl :: [(Symbol, Attr)]) (keys :: [Symbol]) :: Row * where
    FieldsInc pl '[] = '[]
    FieldsInc pl (k ': keys) = k := ParameterType (ResolveParameter pl k) ': FieldsInc pl keys

type family FieldsExc (pl :: [(Symbol, Attr)]) (keys :: [Symbol]) :: Row * where
    FieldsExc '[] keys = '[]
    FieldsExc ('(s, p) ': pl) keys = If (Member s keys) (FieldsExc pl keys) (s := ParameterType p ': FieldsExc pl keys)

type ParamListInc pl keys = Record (FieldsInc pl keys)
type ParamListExc pl keys = Record (FieldsExc pl keys)

type family FieldsMin (pl :: [(Symbol, Attr)]) :: Row * where
    FieldsMin '[] = '[]
    FieldsMin ('(s, AttrReq t) ': pl) = s := t ': FieldsMin pl
    FieldsMin ('(s, AttrOpt t) ': pl) = FieldsMin pl

type FieldsAcc pl rec = (SubRow rec (FieldsMin pl), SubRow (FieldsFull pl) rec)

class Value a where
    showValue :: a -> Text

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

instance ValueList (a == Char) [a] => Value [a] where
    showValue = showValueList (Proxy :: Proxy (a == Char))

class ValueList (str :: Bool) as where
    showValueList :: Proxy str -> as -> RT.Text

instance ValueList True [Char] where
    showValueList _ = RT.pack

instance Value a => ValueList False [a] where
    showValueList _ as = RT.concat $ ["["] ++ intersperse "," (map showValue as) ++ ["]"]

type Dump rec = (KnownFields rec, AllFields rec Value)

dump :: Dump rec => Record rec -> [(Text, Text)]
dump = keyToText . AAnon.toList . AAnon.cmap (Proxy :: Proxy Value) (\(I v) -> K (showValue v)) . Anon.toAdvanced
    where keyToText = map (first RT.pack)

-------------------------------------

type ParamList'fn t u = [
  '("a", AttrReq Int),
  '("b", AttrOpt String),
  '("c", AttrReq (EnumType '["c1","c2"]))
  ]
args1 :: Record '[]
args1 = Anon.empty

args2 :: Record '[ "a" := Int ]
args2 = ANON {a = 3}

args3 :: Record '[ "a" := Int, "b" := Maybe String ]
args3 = ANON {a = 3, b = Just "Hello"}

args4 :: Record '[ "a" := Int, "c" := EnumType '["c1","c2"] ]
args4 = ANON {a = 3, c = #c1}

fn1 :: ParamListFull (ParamList'fn t u) -> _
fn1 = Anon.get #b

fn2 :: ParamListInc (ParamList'fn t u) '["b"] -> Maybe String
fn2 args = fn1 $ Anon.project $ Anon.insert #a 3 $ Anon.insert #c #c1 args

fn3 :: ParamListInc (ParamList'fn t u) '["b", "c"] -> _
fn3 args = fn1 $ Anon.project $ Anon.merge (ANON {a =3, b = Nothing}) args

fn4 :: ParamListExc (ParamList'fn t u) '["a"] -> _
fn4 args = fn1 $ Anon.project $ Anon.merge (ANON {a =3}) args

fn5 :: FieldsAcc (ParamList'fn t u) r => Record r -> _
fn5 args = fn1 $ Anon.inject args (ANON {a = undefined, b = Nothing, c = undefined})

fn6 :: FieldsAcc (ParamList'fn t u) r => Record r -> _
fn6 args = let args' = Anon.inject args (ANON {a = undefined, b = Nothing, c = undefined}) :: ParamListFull (ParamList'fn t u)
            in Anon.get #b args'
