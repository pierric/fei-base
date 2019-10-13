{-# Language RecordWildCards #-}
module MXNet.Base.Executor where

import GHC.Generics (Generic, Generic1)
import Control.DeepSeq (NFData, NFData1, ($!!))

import qualified MXNet.Base.Raw as I
import MXNet.Base.Types (ForeignData(..), Context(..))
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

execReshapeEx :: Executor a -> Bool -> Bool -> Context -> [(String, [Int])] -> IO ([NDArray a], [NDArray a], [NDArray a], Executor a)
execReshapeEx (Executor hdl) partial_shaping allow_up_sizing Context{..} input_shapes = do
    let (names, shapes) = unzip input_shapes
        arg_ind = scanl (+) 0 $ map length shapes
        arg_shp = concat shapes
    (new_arg_in, new_arg_grad, new_arg_aux, new_hdl) <- I.mxExecutorReshapeEx
        partial_shaping allow_up_sizing
        _device_type _device_id
        [] [] []
        names arg_ind arg_shp
        hdl
    let new_arg_in'   = map NDArray new_arg_in
        new_arg_grad' = map NDArray new_arg_grad
        new_arg_aux'  = map NDArray new_arg_aux
        new_exec      = Executor new_hdl
    return $!! (new_arg_in', new_arg_grad', new_arg_aux', new_exec)
