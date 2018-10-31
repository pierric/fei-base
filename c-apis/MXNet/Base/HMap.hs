{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE PolyKinds, DataKinds, TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.Base.HMap where

import GHC.TypeLits
import Data.Proxy
import Data.Typeable
import Data.Maybe

class Pair (p :: Symbol -> * -> *) where
    key   :: p k v -> Proxy k
    value :: p k v -> v
    type MatchHead p (k :: Symbol) v (kvs :: [*]) :: Bool

data HMap (p :: Symbol -> * -> *) (kvs :: [*]) where
  Nil  :: Pair p => HMap p '[]
  Cons :: Pair p => p k v -> HMap p kvs -> HMap p (p k v ': kvs)

infixr 8 .& 
kv .& other = Cons kv other

class Access (b :: Bool) p k v kvs | k kvs -> v where
  get' :: Proxy b -> HMap p kvs -> Proxy k -> v

instance Access True p k v (p k v ': kvs) where
  get' _ (Cons pair _) _ = value pair

instance Access (MatchHead p k v kvs) p k v kvs => Access False p k v (kv ': kvs) where
  get' _ (Cons _ n) k = get' (Proxy :: Proxy (MatchHead p k v kvs)) n k  

(!) :: forall p k v kvs. Access (MatchHead p k v kvs) p k v kvs => HMap p kvs -> Proxy k -> v
(!) = get' (Proxy :: Proxy (MatchHead p k v kvs))

----
type family InHMap p k kvs :: Bool where
    InHMap p k '[] = False
    InHMap p k (p k v ': _) = True
    InHMap p k (_ ': kvs) = InHMap p k kvs

class Query (b :: Bool) p k v kvs where
    query' :: Proxy b -> HMap p kvs -> Proxy k -> Maybe v

instance Access (MatchHead p k v kvs) p k v kvs => Query True p k v kvs where
    query' _ hmap key = Just $ hmap ! key

instance Query False p k v kvs where
    query' _ hmap key = Nothing

(!?) :: forall p k v kvs. Query (InHMap p k kvs) p k v kvs => HMap p kvs -> Proxy k -> Maybe v
(!?) = query' (Proxy :: Proxy (InHMap p k kvs))

-- note this definition 'isJust (hmap !? key)' wouldn't work, because the result type cannot
-- be inferenced.
hasKey :: forall p (k :: Symbol) kvs. Typeable (InHMap p k kvs) => HMap p kvs -> Proxy k -> Bool
hasKey hmap key = typeRep (Proxy :: Proxy (InHMap p k kvs)) == typeRep (Proxy :: Proxy True)