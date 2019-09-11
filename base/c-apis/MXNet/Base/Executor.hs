module MXNet.Base.Executor where

import qualified MXNet.Base.Raw as I
import MXNet.Base.Types (ForeignData(..))


newtype Executor a = Executor { unExecutor :: I.ExecutorHandle }

instance ForeignData (Executor a) where
    touch = I.touchExecutorHandle . unExecutor