{-# LANGUAGE ConstraintKinds        #-}
{-# LANGUAGE FlexibleInstances      #-}
{-# LANGUAGE MultiParamTypeClasses  #-}
{-# LANGUAGE ScopedTypeVariables    #-}
{-# LANGUAGE TypeApplications       #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances   #-}
{-# OPTIONS_GHC -fplugin=Data.Record.Anon.Plugin#-}
module MXNet.NN.Optimizer where

import           Control.Lens                (use, (.=))
import           Data.Record.Anon
import           Data.Record.Anon.Simple     (Record)
import qualified Data.Record.Anon.Simple     as Anon
import           GHC.Exts                    (Constraint)
import           GHC.TypeLits
import           RIO
import           RIO.State

import           MXNet.Base                  (DType, NDArray)
import           MXNet.Base.Core.Spec
import qualified MXNet.Base.Operators.Tensor as T
import           MXNet.Base.Tensor.Class
import           MXNet.NN.LrScheduler
import           MXNet.NN.Types

-- | Abstract Optimizer type class
class DType dtype => Optimizer (opt :: * -> *) dtype where
    type OptimizerTag opt = (s :: Symbol) | s -> opt
    -- | make the optimizer
    makeOptimizer :: (LrScheduler sch, OptimizerCst opt dtype args, MonadIO m)
                  => Proxy (OptimizerTag opt)
                  -> sch
                  -> Record args
                  -> m (opt dtype)
    -- | run the optimizer with the input & expected tensor
    optimize :: (MonadState Statistics m, MonadIO m)
             => opt dtype                            -- optimizer
             -> Text                                 -- symbol name to optimize
             -> NDArray dtype                        -- parameter
             -> NDArray dtype                        -- gradient
             -> m ()

type family OptimizerCst (opt :: * -> *) dt (args :: Row *) :: Constraint

-- | SGD optimizer
data SGD dtype where
    SGD :: (LrScheduler sch, OptimizerCst SGD dtype rec)
            => sch -> Record rec -> SGD dtype

type instance OptimizerCst SGD dtype rec =
    (SubRow (FieldsInc (T.ParameterList_sgd_update NDArray dtype) '["wd", "rescale_grad", "clip_gradient", "lazy_update"]) rec,
     SubRow (FieldsFull (T.ParameterList_sgd_update NDArray dtype)) rec)

instance DType dtype => Optimizer SGD dtype where
    type OptimizerTag SGD = "SGD"
    makeOptimizer _ sch args = return $ SGD sch args
    optimize (SGD sch args) _ weight gradient = do
        nup <- use stat_num_upd
        let lr = getLR sch nup
        stat_last_lr .= lr
        let defaults = paramListDefaults (Proxy @ (T.ParameterList_sgd_update NDArray dtype))
            args' = Anon.inject (ANON{weight = Just weight, grad = Just gradient, lr = lr}) $ Anon.inject args defaults
        liftIO $ void $ T._sgd_update @NDArray @dtype args' $ Just [weight]
