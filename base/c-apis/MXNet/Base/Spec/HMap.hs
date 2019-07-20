{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE PolyKinds, DataKinds, TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.Base.Spec.HMap where

import GHC.TypeLits
import Data.Proxy
import Data.Typeable (Typeable)
import Data.Maybe
import Data.Constraint (Dict(..))
import Type.Reflection (someTypeRep)
import Unsafe.Coerce (unsafeCoerce)

class Pair (p :: Symbol -> * -> *) where
    key   :: p k v -> Proxy k
    value :: p k v -> v

data HMap (p :: Symbol -> * -> *) (kvs :: [*]) where
  Nil  :: Pair p => HMap p '[]
  Cons :: Pair p => p k v -> HMap p kvs -> HMap p (p k v ': kvs)

infixr 0 .& 
kv .& other = Cons kv other

class Access (b :: Bool) p k v kvs | k kvs -> v where
  get' :: Proxy b -> HMap p kvs -> Proxy k -> v

instance Access True p k v (p k v ': kvs) where
  get' _ (Cons pair _) _ = value pair

instance Access (MatchHead p k v kvs) p k v kvs => Access False p k v (kv ': kvs) where
  get' _ (Cons _ n) k = get' (Proxy :: Proxy (MatchHead p k v kvs)) n k  

get :: forall p k v kvs. Access (MatchHead p k v kvs) p k v kvs => HMap p kvs -> Proxy k -> v
get = get' (Proxy :: Proxy (MatchHead p k v kvs))

----
type family InHMap p k kvs :: Bool where
    InHMap p k '[] = False
    InHMap p k (p k v ': _) = True
    InHMap p k (_ ': kvs) = InHMap p k kvs

type family MatchHead p k v kvs where
    MatchHead p k v (p k v ': kvs) = True
    MatchHead p k v (_ ': kvs) = False
    MatchHead p k v '[] = False

type family MatchHeadKey p k kvs where
    MatchHeadKey p k (p k v ': kvs) = True
    MatchHeadKey p k (_ ': kvs) = False
    MatchHeadKey p k '[] = False

-- class Query (b :: Bool) p k v kvs where
--     query' :: Proxy b -> HMap p kvs -> Proxy k -> Maybe v

-- instance Access (MatchHead p k v kvs) p k v kvs => Query True p k v kvs where
--     query' _ hmap key = Just $ hmap ! key

-- instance Query False p k v kvs where
--     query' _ hmap key = Nothing

class Query (b :: Bool) p k v kvs where
    query' :: Proxy b -> HMap p kvs -> Proxy k -> Maybe v

instance Query True p k v (p k v ': kvs) where
    query' _ (Cons pair _) key = Just (value pair)

instance Query False p k v '[] where
    query' _ Nil key = Nothing

instance Query (MatchHead p k v kvs) p k v kvs => Query False p k v (kv ': kvs) where
    query' _ (Cons _ n) key = query' (Proxy :: Proxy (MatchHead p k v kvs)) n key

query :: forall p k v kvs. Query (MatchHead p k v kvs) p k v kvs => HMap p kvs -> Proxy k -> Maybe v
query = query' (Proxy :: Proxy (MatchHead p k v kvs))

-- note this definition 'isJust (hmap !? key)' wouldn't work, because the result type cannot
-- be inferenced.
hasKey :: forall p (k :: Symbol) kvs. KnownSymbol k => HMap p kvs -> Proxy k -> Bool
hasKey hmap key = case axiomInHMapTypeable hmap key of
                    Dict -> someTypeRep (Proxy :: Proxy (InHMap p k kvs)) == someTypeRep (Proxy :: Proxy True)

-- pop one key from the HMap
class PopKey (b :: Bool) p k (kvs :: [*]) where
    type PopResult b k kvs :: [*]
    pop' :: Proxy b -> HMap p kvs -> Proxy k -> HMap p (PopResult b k kvs)

instance PopKey False p k '[] where
    type PopResult False k '[] = '[]
    pop' _ Nil _ = Nil

instance PopKey True p k (p k v ': kvs) where
    type PopResult True k (p k v ': kvs) = kvs
    pop' _ (Cons _ rest) _ = rest

instance PopKey (MatchHeadKey p k kvs) p k kvs => PopKey False p k (p k' v' ': kvs) where
    type PopResult False k (p k' v' ': kvs) = p k' v' ': PopResult (MatchHeadKey p k kvs) k kvs
    pop' _ (Cons pair rest) key = Cons pair (pop' (Proxy :: Proxy (MatchHeadKey p k kvs)) rest key)

pop :: forall p (k :: Symbol) kvs. PopKey (MatchHeadKey p k kvs) p k kvs 
    => HMap p kvs -> Proxy k -> HMap p (PopResult (MatchHeadKey p k kvs) k kvs)
pop = pop' (Proxy :: Proxy (MatchHeadKey p k kvs))

axiomInHMapTypeable :: KnownSymbol k => HMap p kvs -> Proxy k -> Dict (Typeable (InHMap p k kvs))
axiomInHMapTypeable _ _ = unsafeCoerce (Dict :: Dict ())
