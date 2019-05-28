module MXNet.Base.Symbol where

import Foreign.Marshal.Array
import Foreign.Marshal.Alloc
import Foreign.Storable
import Foreign.Ptr
import Foreign.C.String
import Foreign.C.Types
import Control.Monad
import Control.Exception.Base
import Data.List
import Data.Function
import Data.Maybe

import qualified MXNet.Base.Raw as I

newtype Symbol a = Symbol { unSymbol :: I.SymbolHandle }

listArguments :: Symbol a -> IO [String]
listArguments (Symbol sym) = I.mxSymbolListArguments sym

listOutputs :: Symbol a -> IO [String]
listOutputs (Symbol sym) = I.mxSymbolListOutputs sym

listAuxiliaryStates :: Symbol a -> IO [String]
listAuxiliaryStates (Symbol sym) = I.mxSymbolListAuxiliaryStates sym

data CustomOperationException = Invalid

data StorageType = StorageTypeUndefined -- -1
                 | StorageTypeDefault   --  0
                 | StorageTypeRowSparse --  1
                 | StorageTypeCSR       --  2
  deriving (Eq, Bounded, Enum)

toStorageType :: Integral a => a -> StorageType
toStorageType = toEnum . (+1) . fromIntegral

fromStorageType :: Integral a => StorageType -> a
fromStorageType = fromIntegral . (subtract 1) . fromEnum

data MXDType = TyNone       -- -1
             | TyFloat32    --  0
             | TyFloat64    --  1 
             | TyFloat16    --  2
             | TyUInt8      --  3
             | TyInt32      --  4
             | TyInt8       --  5
             | TyInt64      --  6
  deriving (Eq, Bounded, Enum)

toMXDType :: Integral a => a -> MXDType
toMXDType = toEnum . (+1) . fromIntegral

fromMXDType :: Integral a => MXDType -> a
fromMXDType = fromIntegral . (subtract 1) . fromEnum

class CustomOperationProp prop where
    prop_list_arguments        :: prop -> [String]
    prop_list_outputs          :: prop -> [String]
    prop_list_auxiliary_states :: prop -> [String]
    prop_infer_shape :: prop -> [[Int]] -> ([[Int]], [[Int]], [[Int]])
    prop_declare_backward_dependency :: prop -> [Int] -> [Int] -> [Int] -> [Int]
    prop_infer_storage_type          :: prop -> [StorageType] -> ([StorageType], [StorageType], [StorageType])
    prop_infer_storage_type prop in_stype = 
        -- TODO: assert all in_stype are StorageTypeDefault
        let out_stype = replicate (length (prop_list_outputs prop)) StorageTypeDefault
            aux_stype = replicate (length (prop_list_auxiliary_states prop)) StorageTypeDefault
        in (in_stype, out_stype, aux_stype)
    prop_infer_storage_type_backward :: prop -> 
                                        [StorageType] -> 
                                        [StorageType] -> 
                                        [StorageType] -> 
                                        [StorageType] -> 
                                        [StorageType] ->
                                        ([StorageType], [StorageType], [StorageType], [StorageType], [StorageType])
    prop_infer_storage_type_backward prop ograd_stype in_stype out_stype igrad_stype aux_stype = 
        -- TODO: assert all ograd-stype are StorageTypeDefault
        -- TODO: assert all igrad-stype are StorageTypeDefault
        let ograd_stype' = replicate (length ograd_stype) StorageTypeDefault
            in_stype'    = replicate (length in_stype)    StorageTypeDefault
            out_stype'   = replicate (length out_stype)   StorageTypeDefault
            igrad_stype' = replicate (length igrad_stype) StorageTypeDefault
            aux_stype'   = replicate (length aux_stype)   StorageTypeDefault
        in (ograd_stype', in_stype', out_stype', igrad_stype', aux_stype')
    prop_infer_type :: prop -> [MXDType] -> ([MXDType],[MXDType], [MXDType])
    prop_infer_type prop in_type = 
        let t0 = head in_type
            out_type = replicate (length (prop_list_outputs prop)) t0
            aux_type = replicate (length (prop_list_auxiliary_states prop)) t0
        in (in_type, out_type, aux_type)

registerCustomOperator :: CustomOperationProp op => (String, [(String, String)] -> IO op) -> IO ()
registerCustomOperator (op_type, op_ctor) = do
    ptr_creator <- I.mkCustomOpPropCreator creator
    I.mxCustomOpRegister op_type ptr_creator
  where
    creator name argc keys values ret = do
        let argc' = fromIntegral argc
        keys_ <- peekArray argc' keys   >>= mapM peekCString
        vals_ <- peekArray argc' values >>= mapM peekCString
        prop <- op_ctor $ zip keys_ vals_

        -- TODO: free them
        ptr_args <- mapM newCString (prop_list_arguments prop) >>= newArray
        ptr_outs <- mapM newCString (prop_list_outputs prop) >>= newArray
        ptr_auxs <- mapM newCString (prop_list_auxiliary_states prop) >>= newArray
        let size = 10
        ptr_callbacks <- mallocArray size
        ptr_contexts <- newArray (replicate size nullPtr)

        ptr_delete_entry                <- I.mkCustomOpDelFunc (delete_entry ptr_callbacks ptr_contexts)
        ptr_list_arguments_entry        <- I.mkCustomOpListFunc (list_entry ptr_args)
        ptr_list_outputs_entry          <- I.mkCustomOpListFunc (list_entry ptr_outs)
        ptr_list_auxiliary_states_entry <- I.mkCustomOpListFunc (list_entry ptr_auxs)
        ptr_infer_shape_entry           <- I.mkCustomOpInferShapeFunc (infer_shape_entry prop)
        ptr_declare_backward_dependency_entry <- I.mkCustomOpBwdDepFunc (declare_backward_dependency_entry prop)
        ptr_create_operator_entry       <- I.mkCustomOpCreateFunc create_operator_entry
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
        return 0

    delete_entry p1 p2 _ = do
        free p1
        free p2
        return 0
    
    list_entry cstr out_cstr _ = poke out_cstr cstr >> return 0

    infer_shape_entry prop num_tensor in_out_dim_tensor in_out_shapes_tensor _ = do
        let num_inp = length (prop_list_arguments prop)
            num_out = length (prop_list_outputs prop)
            num_aux = length (prop_list_auxiliary_states prop)

        -- TODO: assert num_inp + num_out + num_aux == num_tensors

        -- read dimensions of inputs
        inp_dim_0 <- peekArray num_inp in_out_dim_tensor
        -- read each shape of each input
        inp_shape_0 <- zipWithM (\i j -> do ptr <- peekElemOff in_out_shapes_tensor i 
                                            arr <- peekArray j ptr 
                                            return $ map fromIntegral arr)
                                [0..] (map fromIntegral inp_dim_0)

        let (inp_shape, out_shape, aux_shape) = prop_infer_shape prop inp_shape_0
            all_shape = inp_shape ++ out_shape ++ aux_shape
            all_shape_sizes = map length all_shape

        -- TODO: assert length all_shape == num_tensors
        pokeArray in_out_dim_tensor (map fromIntegral all_shape_sizes)
        -- TODO: free ptr
        -- no need to allocate new dimensions of inputs/outputs/auxiliaries
        -- only need to allocate new shapes for each
        ptr_0 <- newArray (map fromIntegral (concat all_shape) :: [CInt])
        let offsets = scanl (+) 0 all_shape_sizes
            ptrs = map (plusPtr ptr_0) offsets
        pokeArray in_out_shapes_tensor ptrs
        return 0

    declare_backward_dependency_entry prop grad_out data_in data_out out_num_dep out_deps _ = do
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
        -- TODO: free ptr
        ptr_deps <- newArray (map fromIntegral deps)
        poke out_deps ptr_deps
        return 0

    infer_type_entry prop num_tensor tensor_types _ = do
        let num_inp = length (prop_list_arguments prop)
            num_out = length (prop_list_outputs prop)
            num_aux = length (prop_list_auxiliary_states prop)
            num_tensor' = fromIntegral num_tensor
        -- TODO: assert num_tensor == num_inp + num_out + num_aux
        inp_types0 <- map toMXDType <$> peekArray num_inp tensor_types
        let (inp_types, out_types, aux_types) = prop_infer_type prop inp_types0
            all_types = map fromMXDType $ inp_types ++ out_types ++ aux_types
        pokeArray tensor_types all_types
        return 0

    infer_storage_type_entry prop num_tensor tensor_stypes _ = do
        let num_inp = length (prop_list_arguments prop)
            num_out = length (prop_list_outputs prop)
            num_aux = length (prop_list_auxiliary_states prop)
            num_tensor' = fromIntegral num_tensor
        -- TODO: assert num_tensor == num_inp + num_out + num_aux
        inp_stypes0 <- map toStorageType <$> peekArray num_inp tensor_stypes
        let (inp_stypes, out_stypes, aux_stypes) = prop_infer_storage_type prop inp_stypes0
            all_stypes = map fromStorageType $ inp_stypes ++ out_stypes ++ aux_stypes
        pokeArray tensor_stypes all_stypes
        return 0 

    infer_storage_type_backward_entry prop num_tensor tensor_stypes tags _ = do
        let num_inp = length (prop_list_arguments prop)
            num_out = length (prop_list_outputs prop)
            num_aux = length (prop_list_auxiliary_states prop)
            num_tensor' = fromIntegral num_tensor
        tags' <- peekArray num_tensor' tags
        tensor_types' <- peekArray num_tensor' tensor_stypes
        let tensors = groupBy ((==) `on` fst) (zip tags' tensor_types')
            tensors_table = map (\t -> (fst (head t), map (toStorageType . snd) t)) tensors
        -- TODO: assert tensors of length 5
            [ograd0, input0, output0, igrad0, aux0] = map (fromJust . flip lookup tensors_table) [3, 0, 1, 2, 4] 
            (ograd, input, output, igrad, aux) = prop_infer_storage_type_backward prop ograd0 input0 output0 igrad0 aux0
            all_stypes = map fromStorageType $ ograd ++ input ++ output ++ igrad ++ aux
        pokeArray tensor_stypes all_stypes
        return 0

    create_operator_entry ctx num_inputs shapes ndims dtypes ret _ = do
        let size = 3
        ptr_callbacks <- mallocArray size
        ptr_contexts  <- newArray (replicate size nullPtr)

        ptr_delete_entry        <- I.mkCustomFunctionDelFunc (func_delete_entry ptr_callbacks)
        ptr_func_forward_entry  <- I.mkCustomFunctionBwdFunc func_forward_entry
        ptr_func_backward_entry <- I.mkCustomFunctionBwdFunc func_backward_entry
        pokeArray ptr_callbacks [
            castFunPtr ptr_delete_entry,
            castFunPtr ptr_func_forward_entry,
            castFunPtr ptr_func_backward_entry]

        poke ret (I.MXCallbackList size ptr_callbacks ptr_contexts)
        return 0

    func_forward_entry num_ndarray ndarraies tags reqs is_train _ = return 0

    func_backward_entry num_ndarray ndarraies tags reqs is_train _ = return 0

    func_delete_entry p1 _ = do
        free p1
        return 0
