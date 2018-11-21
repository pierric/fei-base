module MXNet.Base.Executor where

import qualified MXNet.Base.Raw as I

newtype Executor a = Executor I.ExecutorHandle