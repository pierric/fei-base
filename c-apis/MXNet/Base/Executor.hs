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
import           MXNet.Base.Types   (Context (..), DType, ForeignData (..),
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

execSimpleBindWithShared
    :: (HasCallStack, DType a)
    => Symbol a
    -> Context
    -> HashMap Text ReqType
    -> HashMap Text Shape
    -> HashMap Text Int
    -> HashMap Text Int
    -> [Text]
    -> Maybe (Executor a)
    -> IO ([NDArray a], [Maybe (NDArray a)], [NDArray a], Executor a)
execSimpleBindWithShared
    symbol Context{..} req_types shapes dtypes stypes shared_arg_names shared_exec = do

    let (gtype_names, gtype_vals) = flattenR req_types
        (shape_names, shape_data, shape_idx) = flattenT shapes
        (dtype_names, dtype_vals) = flattenE dtypes
        (stype_names, stype_vals) = flattenE stypes
        -- we shrae the argument via the exec, so passing '[]' as 'shared_buffer'
        -- sharing mechanism is enabled if the size of shared_buffer is >= 0.
        shared_buffer = case shared_exec of
                          Nothing -> Nothing
                          Just _  -> Just ([], [])
    out <- I.mxExecutorSimpleBindEx
                (unSymbol symbol)
                _device_type _device_id
                [] [] []  -- don't support g2c now
                gtype_names gtype_vals
                shape_names shape_data shape_idx
                dtype_names dtype_vals
                stype_names stype_vals
                shared_arg_names shared_buffer (fmap unExecutor shared_exec)
    -- when shared_* is passed, the graph_executor will reuse the NDArray of shared
    -- arguments, and allocate new NDArray for the others. The newly created ones
    -- are also recorded in the 'upd', but we are not interest in it.
    let (upd, args, grads, aux, exec) = out
    return (map NDArray args, map (NDArray <$>) grads, map NDArray aux, Executor exec)
    where
        flattenR :: HashMap Text ReqType -> ([Text], [Text])
        flattenR mapping = let toStr ReqNull    = "null"
                               toStr ReqWrite   = "write"
                               toStr ReqAdd     = "add"
                               toStr ReqInplace = "inplace"
                            in unzip $ M.toList $ M.map toStr mapping

        flattenE :: Enum a => HashMap Text a -> ([Text], [Int])
        flattenE mapping = unzip $ M.toList $ M.map fromEnum mapping

        flattenT :: HashMap Text [Int] -> ([Text], [Int], [Int])
        flattenT mapping = let (k, v) = unzip $ M.toList mapping
                               dat    = concat v
                               idx    = scanl (+) 0 $ map length v
                            in (k, dat, idx)

execSimpleBind :: (HasCallStack, DType a)
               => Symbol a
               -> Context
               -> HashMap Text ReqType
               -> HashMap Text Shape
               -> HashMap Text Int
               -> HashMap Text Int
               -> IO ([NDArray a], [Maybe (NDArray a)], [NDArray a], Executor a)
execSimpleBind symbol context req_types shapes dtypes stypes =
    execSimpleBindWithShared symbol context req_types shapes dtypes stypes [] Nothing

execFree :: HasCallStack => Executor a -> IO ()
execFree (Executor hdl) = I.withExecutorHandle hdl I.mxExecutorFree
