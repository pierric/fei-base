{-# LANGUAGE ConstraintKinds      #-}
{-# LANGUAGE OverloadedLists      #-}
{-# LANGUAGE UndecidableInstances #-}
module MXNet.NN.Optimizer where

import           Control.Lens                (use, (.=))
import           GHC.Exts                    (Constraint)
import           GHC.TypeLits
import           RIO
import           RIO.State

import           MXNet.Base                  (ArgOf (..), ArgsHMap, DType,
                                              HasArgs, NDArray, (.&))
import qualified MXNet.Base.Operators.Tensor as T
import           MXNet.NN.LrScheduler
import           MXNet.NN.Types

-- | Abstract Optimizer type class
class Optimizer (opt :: * -> *) where
    data OptimizerTag opt :: *
    -- | Specific required arguments
    -- data ReqArgs opt :: *
    -- | Specific optional arguments
    -- type OptArgsList opt :: [KV *]
    -- | make the optimizer
    makeOptimizer :: (DType dtype, LrScheduler sch, OptimizerCst opt dtype args, MonadIO m)
                  => OptimizerTag opt -> sch
                  -> ArgsHMap (OptimizerSym opt) '(NDArray, dtype) args
                  -> m (opt dtype)
    -- | run the optimizer with the input & expected tensor
    optimize :: (DType dtype, MonadState Statistics m, MonadIO m)
             => opt dtype                            -- optimizer
             -> Text                                 -- symbol name to optimize
             -> NDArray dtype                        -- parameter
             -> NDArray dtype                        -- gradient
             -> m ()

type family OptimizerSym (opt :: * -> *) :: Symbol
type family OptimizerCst (opt :: * -> *) dt (args :: [*]) :: Constraint

-- | SGD optimizer
data SGDOpt dtype where
    SGDOpt :: (LrScheduler sch, OptimizerCst SGDOpt dtype args)
            => sch -> ArgsHMap (OptimizerSym SGDOpt) '(NDArray, dtype) args
            -> SGDOpt dtype

type instance OptimizerSym SGDOpt = "_sgd_update"
type instance OptimizerCst SGDOpt dt args =
    HasArgs (OptimizerSym SGDOpt) '(NDArray, dt) args '["wd", "rescale_grad", "clip_gradient", "lazy_update"]

instance Optimizer SGDOpt where
    data OptimizerTag SGDOpt = SGD
    makeOptimizer SGD sch args = return $ SGDOpt sch args
    optimize (SGDOpt sch args) _ weight gradient = do
        nup <- use stat_num_upd
        let lr = getLR sch nup
        stat_last_lr .= lr
        liftIO $ void $ T._sgd_update
            (#weight := weight .& #grad := gradient .& #lr := lr .& args)
            (Just [weight])
