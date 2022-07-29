{-# LANGUAGE ImplicitParams #-}
module MXNet.NN.Module.Class where

import qualified Data.HashMap.Strict         as M
import           Data.Hashable
import           RIO
import qualified RIO.NonEmpty                as NE
import qualified RIO.NonEmpty.Partial        as NE (fromList)
import qualified RIO.Text                    as T
import           Type.Reflection             (typeOf, eqTypeRep, (:~~:)(..))

import           MXNet.Base.AutoGrad
import qualified MXNet.Base.Operators.Tensor as O
import           MXNet.Base.Tensor
import           MXNet.Base.Types
import           MXNet.NN.Initializer

data ParameterPath where
    (:>) :: (Enum e, Eq e, Typeable e) => NonEmpty Text -> e -> ParameterPath

infix 4 :>

instance Eq ParameterPath where
    (t1 :> e1) == (t2 :> e2)
        | t1 /= t2 = False
        | otherwise = case eqTypeRep (typeOf e1) (typeOf e2) of
                        Just HRefl -> e1 == e2
                        Nothing   -> False

instance Hashable ParameterPath where
    hashWithSalt s (t :> e) = hashUsing fromEnum (hashWithSalt s t) e

type InitializerTable u = HashMap ParameterPath (Initializer u)

type ParameterTable   u = HashMap ParameterPath (NDArray u)

class Module m where
    type ModuleDType  m
    type ModuleArgs   m
    type ModuleInput  m
    type ModuleOutput m
    init        :: (HasCallStack, ?device :: Context)
                => NonEmpty Text
                -> ModuleArgs m
                -> InitializerTable (ModuleDType m)
                -> IO m
    forward     :: (HasCallStack, ?device :: Context)
                => m -> ModuleInput m -> IO (ModuleOutput m)
    parameters  :: HasCallStack => m -> IO (ParameterTable (ModuleDType m))


data GenericModule e u = GenericModule {
    _gmodule_scope       :: NonEmpty Text,
    _gmodule_initializer :: InitializerTable u,
    _gmodule_parameters  :: IORef (Maybe (HashMap e (NDArray u)))
}

instance (Enum e, DType u) => Module (GenericModule e u) where
    type ModuleDType (GenericModule e u) = u
    type ModuleArgs  (GenericModule e u) = ()

    -- make a non-initilized generic module
    init scope args user_init = do
        params <- newIORef Nothing
        return $ GenericModule scope user_init params

    parameters generic = do
        p <- getOrInitParams generic (error "module not initialized yet.")
        return $ M.mapKeys (\e -> _gmodule_scope generic :> fromEnum e) p

    forward _ _ = error "generic module doesn't have forward"


scopedName :: NonEmpty Text -> Text
scopedName scope = foldl' T.append "" $ NE.intersperse scopeSplitter $ NE.reverse scope
    where
    scopeSplitter = "."


getOrInitParams (GenericModule _ initializer ref) make = do
    mparams <- readIORef ref
    let make' = do p <- make initializer
                   writeIORef ref (Just p)
                   return p
    maybe make' return mparams

