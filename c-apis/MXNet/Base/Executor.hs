module MXNet.Base.Executor where

import GHC.Generics (Generic, Generic1)
import Control.DeepSeq (NFData, NFData1, ($!!))

import qualified MXNet.Base.Raw as I
import MXNet.Base.Types (ForeignData(..))
import MXNet.Base.NDArray (NDArray(..))

newtype Executor a = Executor { unExecutor :: I.ExecutorHandle }
    deriving (Generic, Generic1, Show)

instance ForeignData (Executor a) where
    touch = I.touchExecutorHandle . unExecutor

instance NFData (Executor a)
instance NFData1 Executor

execGetOutputs :: Executor a -> IO [NDArray a]
execGetOutputs (Executor hdl) = do
    arrhdls <- I.mxExecutorOutputs hdl
    return $!! map NDArray arrhdls

execForward :: Executor a -> Bool -> IO ()
execForward (Executor hdl) = I.mxExecutorForward hdl

execBackward :: Executor a -> [NDArray a] -> IO ()
execBackward (Executor hdl) arrs = I.mxExecutorBackward hdl (map unNDArray arrs)