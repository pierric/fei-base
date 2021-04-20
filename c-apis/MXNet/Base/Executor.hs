{-# LANGUAGE LambdaCase      #-}
{-# LANGUAGE RecordWildCards #-}
module MXNet.Base.Executor where

import           GHC.Generics       (Generic, Generic1)
import           RIO
import qualified RIO.HashMap        as M
import           RIO.List           (scanl, unzip)

import           MXNet.Base.NDArray (NDArray (..))
import qualified MXNet.Base.Raw     as I
import           MXNet.Base.Symbol  (Symbol (..))
import           MXNet.Base.Types   (Context (..), ForeignData (..),
                                     ReqType (..), Shape)

newtype Executor a = Executor { unExecutor :: I.ExecutorHandle }
    deriving (Generic, Generic1, Show)

instance ForeignData (Executor a) where
    touch = I.touchExecutorHandle . unExecutor

instance NFData (Executor a)

execGetOutputs :: HasCallStack => Executor a -> IO [NDArray a]
execGetOutputs (Executor hdl) = do
    arrhdls <- I.mxExecutorOutputs hdl
    return $!! map NDArray arrhdls

execForward :: HasCallStack => Executor a -> Bool -> IO ()
execForward (Executor hdl) = I.mxExecutorForward hdl

execBackward :: HasCallStack => Executor a -> [NDArray a] -> IO ()
execBackward (Executor hdl) arrs = I.mxExecutorBackward hdl (map unNDArray arrs)

execReshapeEx :: HasCallStack
              => Executor a
              -> Bool
              -> Bool
              -> Context
              -> [(Text, [Int])]
              -> IO ([NDArray a], [Maybe (NDArray a)], [NDArray a], Executor a)
execReshapeEx (Executor hdl) partial_shaping allow_up_sizing Context{..} input_shapes = do
    let (names, shapes) = unzip input_shapes
        arg_ind = scanl (+) 0 $ map length shapes
        arg_shp = concat shapes
    (new_arg_in, new_arg_grad, new_arg_aux, new_hdl) <- I.mxExecutorReshapeEx
        partial_shaping allow_up_sizing
        _device_type _device_id
        [] [] []
        names arg_shp arg_ind
        hdl
    let new_arg_in'   = map NDArray new_arg_in
        new_arg_grad' = map (NDArray <$>) new_arg_grad
        new_arg_aux'  = map NDArray new_arg_aux
        new_exec      = Executor new_hdl
    return $!! (new_arg_in', new_arg_grad', new_arg_aux', new_exec)

execBind :: HasCallStack
         => Symbol a
         -> Context
         -> [NDArray a]
         -> [Maybe (NDArray a, ReqType)]
         -> [NDArray a]
         -> IO (Executor a)
execBind symbol Context{..} arg_in arg_gr_with_req arg_aux = do
    let (arg_gr, arg_gr_req) = unzip $ flip map arg_gr_with_req $ \case
                                 Nothing         -> (Nothing, 0)
                                 Just (arr, req) -> (Just $ unNDArray arr, fromEnum req)
    hdl <- I.mxExecutorBind (unSymbol symbol)
                            _device_type _device_id
                            (map unNDArray arg_in)
                            arg_gr
                            arg_gr_req
                            (map unNDArray arg_aux)
    return $ Executor hdl

execFree :: HasCallStack => Executor a -> IO ()
execFree (Executor hdl) = I.withExecutorHandle hdl I.mxExecutorFree
