{-# LANGUAGE ImplicitParams #-}
module MXNet.NN.Module.Class where

import           RIO
import qualified RIO.NonEmpty                as NE
import qualified RIO.NonEmpty.Partial        as NE (fromList)
import qualified RIO.Text                    as T

import           MXNet.Base.AutoGrad
import qualified MXNet.Base.Operators.Tensor as O
import           MXNet.Base.Tensor
import           MXNet.Base.Types
import           MXNet.NN.Initializer

data Parameter t where
    Parameter :: DType t => Text -> NDArray t -> Parameter t

class Module m where
    type ModuleDType m
    type ModuleArgs  m
    data ModuleParamEnums   m
    type ModuleParamTensors m
    init        :: (HasCallStack, ?device :: Context)
                => NonEmpty Text
                -> ModuleArgs m
                -> Maybe (Initializer (ModuleParamEnums m) (ModuleDType m))
                -> IO m
    forward     :: (HasCallStack, ?device :: Context)
                => m -> NDArray (ModuleDType m) -> IO (NDArray (ModuleDType m))
    parameters  :: HasCallStack => m -> IO [Parameter (ModuleDType m)]


scopedName :: NonEmpty Text -> Text
scopedName scope = foldl' T.append "" $ NE.intersperse scopeSplitter $ NE.reverse scope
    where
    scopeSplitter = "."

data GenericModule m = GenericModule {
    _gmodule_name        :: Text,
    _gmodule_initializer :: Initializer (ModuleParamEnums m) (ModuleDType m),
    _gmodule_tensors     :: IORef (Maybe (ModuleParamTensors m))
}

initGenericModule scope initializer = do
    params <- newIORef Nothing
    return $ GenericModule (scopedName scope) initializer params

getOrInitParams (GenericModule _ initializer ref) make = do
    mparams <- readIORef ref
    let make' = do p <- make initializer
                   writeIORef ref (Just p)
                   return p
    maybe make' return mparams

getParams genric = getOrInitParams genric (error "module not initialized yet.")
