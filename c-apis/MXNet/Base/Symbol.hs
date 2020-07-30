{-# LANGUAGE ViewPatterns #-}
module MXNet.Base.Symbol where

import qualified Data.Vector.Mutable     as VM
import           RIO
import           RIO.List                (headMaybe, scanl, unzip)
import qualified RIO.NonEmpty            as RNE
import           RIO.Partial             (toEnum)
import qualified RIO.Vector.Boxed        as V
import qualified RIO.Vector.Boxed.Unsafe as V

import           Data.Typeable           (Typeable)
import           Foreign.C.String
import           Foreign.C.Types
import           Foreign.Marshal.Alloc
import           Foreign.Marshal.Array
import           Foreign.Ptr
import           Foreign.Storable

import qualified MXNet.Base.Raw          as I
import           MXNet.Base.Types        (ForeignData (..))

newtype Symbol a = Symbol { unSymbol :: I.SymbolHandle }

instance ForeignData (Symbol a) where
    touch = I.touchSymbolHandle . unSymbol

data SymbolException = SymbolIndexOutOfBound Int Int
    | SymbolNameNotFound Text
    | SymbolMultipleOutputs Text
    deriving (Typeable, Show)
instance Exception SymbolException

data FShape = STensor
    { _shape_nonempty :: NonEmpty Int
    }
    | SScalar
    deriving Show

shapeLength (STensor l) = length l
shapeLength SScalar     = 0

shapeToList (STensor l) = RNE.toList l
shapeToList SScalar     = []

shapeCons a (STensor s) = STensor $ a RNE.<| s
shapeCons a SScalar     = STensor $ a RNE.:| []

class SymbolClass s where
    getName             :: (HasCallStack, MonadIO m) => s -> m (Maybe Text)
    listArguments       :: (HasCallStack, MonadIO m) => s -> m [Text]
    listOutputs         :: (HasCallStack, MonadIO m) => s -> m [Text]
    listAuxiliaryStates :: (HasCallStack, MonadIO m) => s -> m [Text]
    numOutputs          :: (HasCallStack, MonadIO m) => s -> m Int
    at                  :: (HasCallStack, MonadIO m) => s -> Int -> m s
    group               :: (HasCallStack, MonadIO m) => [s] -> m s
    internals           :: (HasCallStack, MonadIO m) => s -> m s
    inferShape          :: (HasCallStack, MonadIO m)
                        => s -> [(Text, FShape)]
                        -> m ([(Text, FShape)], [(Text, FShape)], [(Text, FShape)], Bool)

    at'                 :: (HasCallStack, MonadIO m) => s -> Text -> m s
    at' sym name = do
        all_names <- listOutputs sym
        case V.findIndex (== name) $ V.fromList all_names of
            Just idx -> at sym idx
            Nothing  -> throwIO (SymbolNameNotFound name)

instance SymbolClass I.SymbolHandle where
    getName = liftIO . I.mxSymbolGetName
    listArguments       = liftIO . I.mxSymbolListArguments
    listOutputs         = liftIO . I.mxSymbolListOutputs
    listAuxiliaryStates = liftIO . I.mxSymbolListAuxiliaryStates
    numOutputs          = liftIO . I.mxSymbolGetNumOutputs
    at sym index = liftIO $ do
        max <- numOutputs sym
        if index < 0 || index >= max then
            throwIO (SymbolIndexOutOfBound index max)
        else
            I.mxSymbolGetOutput sym index
    group = liftIO . I.mxSymbolCreateGroup
    internals = liftIO . I.mxSymbolGetInternals

    inferShape sym known = liftIO $ do
        let (names, shapes) = unzip known
            arg_ind = scanl (+) 0 $ map shapeLength shapes
            arg_shp = concatMap shapeToList shapes
        (inp_shp, out_shp, aux_shp, complete) <- I.mxSymbolInferShapePartial sym names arg_ind arg_shp
        inps <- listArguments sym
        outs <- listOutputs sym
        auxs <- listAuxiliaryStates sym
        return (pair inps inp_shp, pair outs out_shp, pair auxs aux_shp, complete)
      where
        build name (RNE.nonEmpty -> Just s) = (name, STensor s)
        build name _                        = (name, SScalar)
        pair names shapes = zipWith build names shapes

instance SymbolClass (Symbol a) where
    getName             = getName . unSymbol
    listArguments       = listArguments . unSymbol
    listOutputs         = listOutputs . unSymbol
    listAuxiliaryStates = listAuxiliaryStates . unSymbol
    numOutputs          = numOutputs . unSymbol
    at (Symbol s)       = (Symbol <$>) . at s
    group = (Symbol <$>) . group . map unSymbol
    internals           = (Symbol <$>) . internals . unSymbol
    inferShape          = inferShape  . unSymbol

listInternals :: (SymbolClass sym, MonadIO m) => sym -> m [Text]
listInternals sym = internals sym >>= listOutputs

getInternalByName :: (SymbolClass sym, MonadIO m) => sym -> Text -> m (Maybe sym)
getInternalByName sym name = liftIO $ handle on_exc $ do
    layers <- internals sym
    target <- at' layers name
    return $ Just target
  where
      on_exc :: SymbolException -> IO (Maybe a)
      on_exc _ = return Nothing

inferOutputShape :: (SymbolClass sym, MonadIO m) => sym  -> [(Text, FShape)] -> m FShape
inferOutputShape sym input_shapes = do
    (_, out, _, _) <- inferShape sym input_shapes
    case out of
      [(_, shp)] -> return shp
      _ -> throwIO (SymbolMultipleOutputs "use `inferShape` to get list of output shapes.")

data StorageType = StorageTypeUndefined
    | StorageTypeDefault
    | StorageTypeRowSparse
    | StorageTypeCSR
    deriving (Eq, Bounded, Enum, Show)

toStorageType :: Integral a => a -> StorageType
toStorageType = toEnum . (+1) . fromIntegral

fromStorageType :: Integral a => StorageType -> a
fromStorageType = fromIntegral . (subtract 1) . fromEnum

-- MXDType values: -1, 0, 1, 2, 3, 4, 5, 6
data MXDType = TyNone
    | TyFloat32
    | TyFloat64
    | TyFloat16
    | TyUInt8
    | TyInt32
    | TyInt8
    | TyInt64
    deriving (Eq, Bounded, Enum, Show)

toMXDType :: Integral a => a -> MXDType
toMXDType = toEnum . (+1) . fromIntegral

fromMXDType :: Integral a => MXDType -> a
fromMXDType = fromIntegral . (subtract 1) . fromEnum

-- TODO: change String to Text
class CustomOperation (Operation prop) => CustomOperationProp prop where
    -- list the names of inputs
    prop_list_arguments        :: prop -> [String]
    -- list the names of outputs
    prop_list_outputs          :: prop -> [String]
    -- list the names of axiliary states
    prop_list_auxiliary_states :: prop -> [String]
    -- infer the shape.
    -- params: shapes of each inputs
    -- return: shapes of inputs, outputs, auxiliary states
    prop_infer_shape :: prop -> [FShape] -> ([FShape], [FShape], [FShape])
    -- declare the dependency of symbols
    -- params: unique indices of inputs, outputs, auxiliary states
    -- return: dependant indices.
    prop_declare_backward_dependency :: prop -> [Int] -> [Int] -> [Int] -> [Int]
    prop_infer_storage_type          :: prop -> [StorageType] -> ([StorageType], [StorageType], [StorageType])
    prop_infer_storage_type prop in_stype =
        let out_stype = replicate (length (prop_list_outputs prop)) StorageTypeDefault
            aux_stype = replicate (length (prop_list_auxiliary_states prop)) StorageTypeDefault
            withAssert = assert (all (==StorageTypeDefault) in_stype)
        in withAssert (in_stype, out_stype, aux_stype)
    prop_infer_storage_type_backward :: prop ->
                                        [StorageType] ->
                                        [StorageType] ->
                                        [StorageType] ->
                                        [StorageType] ->
                                        [StorageType] ->
                                        ([StorageType], [StorageType], [StorageType], [StorageType], [StorageType])
    prop_infer_storage_type_backward prop ograd_stype in_stype out_stype igrad_stype aux_stype =
        let ograd_stype' = replicate (length ograd_stype) StorageTypeDefault
            in_stype'    = replicate (length in_stype)    StorageTypeDefault
            out_stype'   = replicate (length out_stype)   StorageTypeDefault
            igrad_stype' = replicate (length igrad_stype) StorageTypeDefault
            aux_stype'   = replicate (length aux_stype)   StorageTypeDefault
            withAssert   = let allDefault   = all (==StorageTypeDefault)
                           in assert (allDefault ograd_stype && allDefault igrad_stype)
        in withAssert (ograd_stype', in_stype', out_stype', igrad_stype', aux_stype')
    prop_infer_type :: prop -> [MXDType] -> ([MXDType],[MXDType], [MXDType])
    prop_infer_type prop in_type =
        case headMaybe in_type of
            Just t0 -> let out_type = replicate (length (prop_list_outputs prop)) t0
                           aux_type = replicate (length (prop_list_auxiliary_states prop)) t0
                       in (in_type, out_type, aux_type)

    data Operation prop :: *
    prop_create_operator :: prop -> [[Int]] -> [MXDType] -> IO (Operation prop)

class CustomOperation op where
    forward  :: op -> [ReqEnum]
             -> [I.NDArrayHandle]
             -> [I.NDArrayHandle]
             -> [I.NDArrayHandle]
             -> Bool
             -> IO ()
    backward :: op -> [ReqEnum]
             -> [I.NDArrayHandle]
             -> [I.NDArrayHandle]
             -> [I.NDArrayHandle]
             -> [I.NDArrayHandle]
             -> [I.NDArrayHandle]
             -> IO ()

data ReqEnum = ReqNull
    | ReqWrite
    | ReqInplace
    | ReqAdd
    deriving (Bounded, Enum)

registerCustomOperator :: CustomOperationProp op => (Text, [(Text, Text)] -> IO op) -> IO ()
registerCustomOperator (op_type, op_ctor) = do
    ptr_creator <- I.mkCustomOpPropCreator creator
    I.mxCustomOpRegister op_type ptr_creator
  where
    creator name argc keys values ret = do
        let argc' = fromIntegral argc
        keys_ <- peekArray argc' keys   >>= mapM I.peekCStringT
        vals_ <- peekArray argc' values >>= mapM I.peekCStringT
        prop <- op_ctor $ zip keys_ vals_

        -- TODO: a safer way is use ForeignPtr,
        -- malloc the str as ForeignPtr, and save
        -- to the allocList. When the delete_entry is
        -- called, it take away all ForeignPtr, and
        -- effectively informs GC to revoke underlying memory.
        args <- mapM newCString (prop_list_arguments prop)
        ptr_args <- newArray $ args ++ [nullPtr]

        outs <- mapM newCString (prop_list_outputs prop)
        ptr_outs <- newArray $ outs ++ [nullPtr]

        auxs <- mapM newCString (prop_list_auxiliary_states prop)
        ptr_auxs <- newArray $ auxs ++ [nullPtr]

        let size = 10
        ptr_callbacks <- mallocArray size
        ptr_contexts <- newArray (replicate size nullPtr)

        allocList <- newMVar $ [castPtr ptr_args,
                                castPtr ptr_outs,
                                castPtr ptr_auxs,
                                castPtr ptr_callbacks,
                                castPtr ptr_contexts]
                               ++ map castPtr args
                               ++ map castPtr outs
                               ++ map castPtr auxs

        ptr_delete_entry                <- I.mkCustomOpDelFunc (delete_entry allocList)
        ptr_list_arguments_entry        <- I.mkCustomOpListFunc (list_entry ptr_args)
        ptr_list_outputs_entry          <- I.mkCustomOpListFunc (list_entry ptr_outs)
        ptr_list_auxiliary_states_entry <- I.mkCustomOpListFunc (list_entry ptr_auxs)
        ptr_infer_shape_entry           <- I.mkCustomOpInferShapeFunc (infer_shape_entry allocList prop)
        ptr_declare_backward_dependency_entry <- I.mkCustomOpBwdDepFunc (declare_backward_dependency_entry allocList prop)
        ptr_create_operator_entry       <- I.mkCustomOpCreateFunc (create_operator_entry prop)
        ptr_infer_type_entry            <- I.mkCustomOpInferTypeFunc (infer_type_entry prop)
        ptr_infer_storage_type_entry    <- I.mkCustomOpInferStorageTypeFunc (infer_storage_type_entry prop)
        ptr_infer_storage_type_backward_entry <- I.mkCustomOpBackwardInferStorageTypeFunc (infer_storage_type_backward_entry prop)

        pokeArray ptr_callbacks [
            castFunPtr ptr_delete_entry,
            castFunPtr ptr_list_arguments_entry,
            castFunPtr ptr_list_outputs_entry,
            castFunPtr ptr_list_auxiliary_states_entry,
            castFunPtr ptr_infer_shape_entry,
            castFunPtr ptr_declare_backward_dependency_entry,
            castFunPtr ptr_create_operator_entry,
            castFunPtr ptr_infer_type_entry,
            castFunPtr ptr_infer_storage_type_entry,
            castFunPtr ptr_infer_storage_type_backward_entry]

        poke (castPtr ret) (I.MXCallbackList size ptr_callbacks ptr_contexts)
        return 1

    delete_entry allocList _ = do
        list <- takeMVar allocList
        mapM_ free list
        return 1

    list_entry cstr out_cstr x = do
        poke out_cstr cstr
        return 1

    infer_shape_entry allocList prop num_tensor in_out_dim_tensor in_out_shapes_tensor _ = do
        let num_inp = length (prop_list_arguments prop)
            num_out = length (prop_list_outputs prop)
            num_aux = length (prop_list_auxiliary_states prop)
            num_tensor' = fromIntegral num_tensor

        assert (num_inp + num_out + num_aux == num_tensor') (return ())

        -- read dimensions of inputs
        inp_dim_0 <- peekArray num_inp in_out_dim_tensor
        -- read each shape of each input
        inp_shape_0 <- zipWithM (\i j -> do ptr <- peekElemOff in_out_shapes_tensor i
                                            arr <- peekArray j ptr
                                            case RNE.map fromIntegral <$> RNE.nonEmpty arr of
                                                Nothing  -> return SScalar
                                                Just shp -> return $ STensor shp)
                                [0..] (map fromIntegral inp_dim_0)

        let (inp_shape, out_shape, aux_shape) = prop_infer_shape prop inp_shape_0
            all_shape = inp_shape ++ out_shape ++ aux_shape
            all_shape_sizes = map shapeLength all_shape

        assert (length all_shape == num_tensor') (return ())

        pokeArray in_out_dim_tensor (map fromIntegral all_shape_sizes)
        -- no need to allocate new dimensions of inputs/outputs/auxiliaries
        -- only need to allocate new shapes for each
        let shape_vec = map fromIntegral $ concatMap shapeToList all_shape :: [CInt]
        ptr_0 <- newArray shape_vec
        modifyMVar_ allocList (return . (castPtr ptr_0 :))

        let size_UINT = sizeOf (undefined :: I.MX_UINT)
            -- all the first num_tensor' ptr refers to the shape.
            -- the last one is the end marker.
            offsets = take num_tensor' $ scanl (+) 0 $ map (size_UINT *) all_shape_sizes
            ptrs = map (plusPtr ptr_0) offsets
        pokeArray in_out_shapes_tensor ptrs
        return 1

    declare_backward_dependency_entry allocList prop grad_out data_in data_out out_num_dep out_deps _ = do
        let num_inp = length (prop_list_arguments prop)
            num_out = length (prop_list_outputs   prop)
        grad_out_inds <- peekArray num_out grad_out
        data_in_inds  <- peekArray num_inp data_in
        data_out_inds <- peekArray num_out data_out
        let deps = prop_declare_backward_dependency prop
                        (map fromIntegral grad_out_inds)
                        (map fromIntegral data_in_inds)
                        (map fromIntegral data_out_inds)
        poke out_num_dep (fromIntegral $ length deps)
        ptr_deps <- newArray (map fromIntegral deps)
        modifyMVar_ allocList (return . (castPtr ptr_deps :))

        poke out_deps ptr_deps
        return 1

    infer_type_entry prop num_tensor tensor_types _ = do
        let num_inp = length (prop_list_arguments prop)
            num_out = length (prop_list_outputs prop)
            num_aux = length (prop_list_auxiliary_states prop)
            num_tensor' = fromIntegral num_tensor
        assert (num_inp + num_out + num_aux == num_tensor') (return ())

        inp_types0 <- map toMXDType <$> peekArray num_inp tensor_types
        let (inp_types, out_types, aux_types) = prop_infer_type prop inp_types0
            all_types = map fromMXDType $ inp_types ++ out_types ++ aux_types
        pokeArray tensor_types all_types
        return 1

    infer_storage_type_entry prop num_tensor tensor_stypes _ = do
        let num_inp = length (prop_list_arguments prop)
            num_out = length (prop_list_outputs prop)
            num_aux = length (prop_list_auxiliary_states prop)
            num_tensor' = fromIntegral num_tensor
        assert (num_inp + num_out + num_aux == num_tensor') (return ())

        inp_stypes0 <- map toStorageType <$> peekArray num_inp tensor_stypes
        let (inp_stypes, out_stypes, aux_stypes) = prop_infer_storage_type prop inp_stypes0
            all_stypes = map fromStorageType $ inp_stypes ++ out_stypes ++ aux_stypes
        pokeArray tensor_stypes all_stypes
        return 1

    infer_storage_type_backward_entry prop num_tensor tensor_stypes tags _ = do
        let num_inp = length (prop_list_arguments prop)
            num_out = length (prop_list_outputs prop)
            num_aux = length (prop_list_auxiliary_states prop)
            num_tensor' = fromIntegral num_tensor
        tags' <- peekArray num_tensor' tags
        tensor_types' <- peekArray num_tensor' tensor_stypes
        let tensors = RNE.groupBy ((==) `on` fst) (zip tags' tensor_types')
            tensors_table = map (\t ->
                let tag = fst $ RNE.head t
                    sto = RNE.toList $ RNE.map (toStorageType . snd) t
                in (tag, sto)) tensors
            [ograd0, input0, output0, igrad0, aux0] = map (fromMaybe [] . flip lookup tensors_table) [3, 0, 1, 2, 4]
            (ograd, input, output, igrad, aux) = prop_infer_storage_type_backward prop ograd0 input0 output0 igrad0 aux0
            all_stypes = map fromStorageType $ ograd ++ input ++ output ++ igrad ++ aux

        assert (length tensors == 5) (return ())
        pokeArray tensor_stypes all_stypes
        return 1

    create_operator_entry prop ctx num_inputs shapes ndims dtypes ret _ = do
        let num_inputs_ = fromIntegral num_inputs
        ndims  <- map fromIntegral <$> peekArray num_inputs_ ndims
        dtypes <- map toMXDType    <$> peekArray num_inputs_ dtypes
        ptrs   <- peekArray num_inputs_ shapes
        shapes <- mapM ((map fromIntegral <$>) . uncurry peekArray) (zip ndims ptrs)
        op <- prop_create_operator prop shapes dtypes

        let size = 3
        ptr_callbacks <- mallocArray size
        ptr_contexts  <- newArray (replicate size nullPtr)
        allocList <- newMVar $ [castPtr ptr_callbacks, castPtr ptr_contexts]

        ptr_delete_entry        <- I.mkCustomFunctionDelFunc (delete_entry allocList)
        ptr_func_forward_entry  <- I.mkCustomFunctionBwdFunc (func_forward_entry  op)
        ptr_func_backward_entry <- I.mkCustomFunctionBwdFunc (func_backward_entry op)
        pokeArray ptr_callbacks [
            castFunPtr ptr_delete_entry,
            castFunPtr ptr_func_forward_entry,
            castFunPtr ptr_func_backward_entry]

        poke ret (I.MXCallbackList size ptr_callbacks ptr_contexts)
        return 1

    func_forward_entry op num_ndarray ndarrays tags reqs is_train _ = do
        let num_ndarray_ = fromIntegral num_ndarray
        ndarrays <- peekArray num_ndarray_ ndarrays >>= mapM I.newNDArrayHandle
        tags     <- peekArray num_ndarray_ tags

        let tensors = V.map reverse $ V.create $ do
                        vec <- VM.replicate 5 []
                        forM (zip tags ndarrays) $ \(ind, hdl) -> do
                            let ind_ = fromIntegral ind
                            lst <- VM.read vec ind_
                            VM.write vec ind_ (hdl : lst)
                        return vec

        let in_data  = tensors ! 0
            out_data = tensors ! 1
            aux      = tensors ! 4

        reqs <- map (toEnum . fromIntegral) <$> peekArray (length out_data) reqs :: IO [ReqEnum]
        forward op reqs in_data out_data aux (is_train == 1)
        return 1

    func_backward_entry op num_ndarray ndarrays tags reqs is_train _ = do
        let num_ndarray_ = fromIntegral num_ndarray
        ndarrays <- peekArray num_ndarray_ ndarrays >>= mapM I.newNDArrayHandle
        tags     <- peekArray num_ndarray_ tags

        let tensors = V.map reverse $ V.create $ do
                        vec <- VM.replicate 5 []
                        forM (zip tags ndarrays) $ \(ind, hdl) -> do
                            let ind_ = fromIntegral ind
                            lst <- VM.read vec ind_
                            VM.write vec ind_ (hdl : lst)
                        return vec

        let in_data  = tensors ! 0
            out_data = tensors ! 1
            in_grad  = tensors ! 2
            out_grad = tensors ! 3
            aux      = tensors ! 4

        reqs <- map (toEnum . fromIntegral) <$> peekArray (length out_data) reqs
        backward op reqs in_data out_data in_grad out_grad aux
        return 1

    (!) = V.unsafeIndex

