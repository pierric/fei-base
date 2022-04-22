{-# LANGUAGE AllowAmbiguousTypes   #-}
{-# LANGUAGE ConstraintKinds       #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE TypeApplications      #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}
{-# OPTIONS_GHC -fplugin=Data.Record.Anon.Plugin #-}
module MXNet.Base.Core.Spec where

import           Data.Primitive.SmallArray        (smallArrayFromList)
import           Data.Record.Anon
import qualified Data.Record.Anon.Advanced        as AAnon
import           Data.Record.Anon.Simple          (Record)
import qualified Data.Record.Anon.Simple          as Anon
import qualified Data.Record.Generic              as G (Generic (..), Rep (..))
import qualified Data.Record.Generic.Rep.Internal as G (noInlineUnsafeCo)
import           Data.Type.Bool
import           Data.Type.Equality
import           GHC.Exts                         (Any, Constraint)
import           GHC.OverloadedLabels
import           GHC.TypeLits                     (CmpSymbol,
                                                   ErrorMessage ((:<>:)),
                                                   KnownNat, Nat, Symbol,
                                                   TypeError, natVal, symbolVal)
import qualified GHC.TypeLits                     as Lit
import           RIO
import           RIO.List                         (intersperse)
import qualified RIO.Text                         as RT
import           Unsafe.Coerce                    (unsafeCoerce)

import           MXNet.Base.Core.Enum
import           MXNet.Base.Types                 (DType)


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

type family ParamListReq (pl :: [(Symbol, Attr)]) :: [(Symbol, Attr)] where
    ParamListReq '[] = '[]
    ParamListReq ('(s, AttrReq t) ': pl) = '(s, AttrReq t) ': ParamListReq pl
    ParamListReq ('(s, AttrOpt t) ': pl) = ParamListReq pl

type family ParamListOpt (pl :: [(Symbol, Attr)]) :: [(Symbol, Attr)] where
    ParamListOpt '[] = '[]
    ParamListOpt ('(s, AttrReq t) ': pl) = ParamListOpt pl
    ParamListOpt ('(s, AttrOpt t) ': pl) = '(s, AttrOpt t) ': ParamListOpt pl

type FieldsFull pl = ParamListToFields pl
type FieldsReq  pl = ParamListToFields (ParamListReq pl)
type FieldsOpt  pl = ParamListToFields (ParamListOpt pl)

type family FieldsInc (pl :: [(Symbol, Attr)]) (keys :: [Symbol]) :: Row * where
    FieldsInc pl '[] = '[]
    FieldsInc pl (k ': keys) = k := ParameterType (ResolveParameter pl k) ': FieldsInc pl keys

type family FieldsExc (pl :: [(Symbol, Attr)]) (keys :: [Symbol]) :: Row * where
    FieldsExc '[] keys = '[]
    FieldsExc ('(s, p) ': pl) keys = If (Member s keys) (FieldsExc pl keys) (s := ParameterType p ': FieldsExc pl keys)

type family FieldsMin (pl :: [(Symbol, Attr)]) :: Row * where
    FieldsMin '[] = '[]
    FieldsMin ('(s, AttrReq t) ': pl) = s := t ': FieldsMin pl
    FieldsMin ('(s, AttrOpt t) ': pl) = FieldsMin pl

type FieldsAcc pl rec = (SubRow rec (FieldsMin pl), SubRow (FieldsFull pl) rec, KnownFields rec)

class KnownSize (es :: [k]) where
    reifyLen :: Proxy es -> Int

instance KnownSize '[] where
    reifyLen _ = 0

instance KnownSize es => KnownSize (e ': es) where
    reifyLen _ = 1 + (reifyLen $ Proxy @es)

class KnownParamList (pl :: [(Symbol, Attr)]) where
    type ParamListToFields  pl :: Row *
    reifyFields :: Proxy pl -> AAnon.Record (K String) (ParamListToFields pl)

instance KnownParamList '[] where
    type ParamListToFields '[] = '[]
    reifyFields _ = AAnon.empty

instance (KnownSymbol k, KnownHash k, KnownParamList pl) => KnownParamList ('(k, AttrOpt (v :: *)) ': pl) where
    type ParamListToFields ('(k, AttrOpt v) ': pl) = k := Maybe v ': ParamListToFields pl
    reifyFields _ = let key = K $ symbolVal $ Proxy @k :: K String (Maybe v)
                        rem = reifyFields (Proxy @pl) :: AAnon.Record (K String) (ParamListToFields pl)
                     in AAnon.insert (fromLabel @k) key rem

instance (KnownSymbol k, KnownHash k, KnownParamList pl) => KnownParamList ('(k, AttrReq (v :: *)) ': pl) where
    type ParamListToFields ('(k, AttrReq v) ': pl) = k := v ': ParamListToFields pl
    reifyFields _ = let key = K $ symbolVal $ Proxy @k :: K String v
                        rem = reifyFields (Proxy @pl) :: AAnon.Record (K String) (ParamListToFields pl)
                     in AAnon.insert (fromLabel @k) key rem

paramListDefaults :: forall pl. (HasCallStack, KnownSize pl, KnownParamList (ParamListOpt pl))
                  => Proxy pl -> Record (FieldsOpt pl)
paramListDefaults dummy =
    let omitted = I (unsafeCoerce Nothing)
        cnt = reifyLen $ Proxy @pl
     in case AAnon.reflectKnownFields $ reifyFields $ Proxy @(ParamListOpt pl) of
            Reflected -> G.to $ G.Rep $ smallArrayFromList $ replicate cnt omitted

--
-- TODO: Any better implementation?
--
-- This function fills in the optional argument with the default value Nothing. It relies on
-- the internal implementation of the Record. It seems not ideal, because it converts to the
-- internal rep, looking up in the provided args, and converts back to Record.
--
-- Alternative method can be simply merge with the result of paramListDefaults, but it is
-- very hard to prove that:
--
-- |- SubRow r (FieldsMin pl) => SubRow (Merge r (FieldsOpt pl)) (FieldsFull pl)
--
paramListWithDefault :: forall pl r. (HasCallStack, KnownParamList pl, FieldsAcc pl r)
                     => Proxy pl -> Record r -> Record (FieldsFull pl)
paramListWithDefault dummy args =
    let args_kv   = AAnon.toList $ AAnon.map (\(I x) -> K $ I $ (G.noInlineUnsafeCo x :: Any)) $ Anon.toAdvanced args
        opt_def   = I $ G.noInlineUnsafeCo Nothing :: I Any
        full_keys = reifyFields $ Proxy @pl
        full_vals = [fromMaybe opt_def $ lookup k args_kv | k <- AAnon.collapse full_keys]
     in case AAnon.reflectKnownFields full_keys of
            Reflected -> G.to $ G.Rep $ smallArrayFromList full_vals

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

fn1 :: Record (FieldsFull (ParamList'fn t u)) -> _
fn1 = Anon.get #b

fn2 :: Record (FieldsInc (ParamList'fn t u) '["b"]) -> Maybe String
fn2 args = fn1 $ Anon.project $ Anon.insert #a 3 $ Anon.insert #c #c1 args

fn3 :: Record (FieldsInc (ParamList'fn t u) '["b", "c"]) -> _
fn3 args = fn1 $ Anon.project $ Anon.merge (ANON {a =3, b = Nothing}) args

fn4 :: Record (FieldsExc (ParamList'fn t u) '["a"]) -> _
fn4 args = fn1 $ Anon.project $ Anon.merge (ANON {a =3}) args

fn5 :: forall t u r. FieldsAcc (ParamList'fn t u) r => Record r -> _
fn5 args = fn1 $ paramListWithDefault (Proxy @(ParamList'fn t u)) args

fn6 :: forall t u r. FieldsAcc (ParamList'fn t u) r => Record r -> _
fn6 args = let args' = paramListWithDefault (Proxy @(ParamList'fn t u)) args
            in Anon.get #b args'
