{-# LANGUAGE CPP #-}
module MXNet.Base.Tensor.Functional where

import           GHC.Float                   (double2Float)
import           RIO

import qualified MXNet.Base.Operators.Tensor as S
import           MXNet.Base.Spec.HMap        (HMap (..), (.&))
import           MXNet.Base.Spec.Operator
import           MXNet.Base.Tensor.Class

#ifdef MXNET_VERSION

-----------------------------------------------------------------------------
-- For both Symbol and NDArray
--
-- Note: many functions in this module admit the numpy's shape semantic, i.e.
-- 1. () represents the shape of scalar tensors, and
-- 2. tuples with 0s, for example, (0,), (1, 0, 2), represent the shapes of
--  zero-size tensors
--
-- You should call `mxSetIsNumpyShape` to turn on the Numpy semantics before
-- calling the functions.
--
-----------------------------------------------------------------------------

pooling :: (PrimTensorOp t t, Fullfilled "_Pooling" t args)
        => ArgsHMap "_Pooling" t args -> TensorMonad t t
pooling = prim S._Pooling

activation :: (PrimTensorOp t t, Fullfilled "_Activation" t args)
           => ArgsHMap "_Activation" t args -> TensorMonad t t
activation = prim S._Activation

softmax :: (PrimTensorOp t t, Fullfilled "_softmax" t args)
        => ArgsHMap "_softmax" t args -> TensorMonad t t
softmax = prim S._softmax

softmaxoutput :: (PrimTensorOp t t, Fullfilled "_SoftmaxOutput" t args)
              => ArgsHMap "_SoftmaxOutput" t args -> TensorMonad t t
softmaxoutput = prim S._SoftmaxOutput

flatten t = prim S._Flatten (#data := t .& Nil)
dropout t p = prim S._Dropout (#data := t .& #p := p .& Nil)

fullLike :: (HasCallStack, PrimTensorOp t t) => Double -> t -> TensorMonad t t
fullLike v a = prim S.__npi_full_like (#a := a .& #fill_value := v .& Nil)

zerosLike, onesLike :: (HasCallStack, PrimTensorOp t t) => t -> TensorMonad t t
zerosLike = fullLike 0
onesLike  = fullLike 1

eye :: (HasCallStack, PrimTensorOp t t) => [Int] -> TensorMonad t t
eye shape = prim S.__npi_identity (#shape := shape .& Nil)

add_, sub_, mul_, div_, eq_, neq_, lt_, leq_, gt_, geq_, and_, or_, xor_ ::
    (HasCallStack, PrimTensorOp t t) => t -> t -> TensorMonad t t
add_ a b = prim S.__npi_add         (#lhs := a .& #rhs := b .& Nil)
sub_ a b = prim S.__npi_subtract    (#lhs := a .& #rhs := b .& Nil)
mul_ a b = prim S.__npi_multiply    (#lhs := a .& #rhs := b .& Nil)
div_ a b = prim S.__npi_true_divide (#lhs := a .& #rhs := b .& Nil)

eq_   a b = prim S.__npi_equal      (#lhs := a .& #rhs := b .& Nil)
neq_  a b = prim S.__npi_not_equal  (#lhs := a .& #rhs := b .& Nil)
lt_   a b = prim S.__npi_less       (#lhs := a .& #rhs := b .& Nil)
leq_  a b = prim S.__npi_less_equal (#lhs := a .& #rhs := b .& Nil)
gt_   a b = prim S.__npi_greater    (#lhs := a .& #rhs := b .& Nil)
geq_  a b = prim S.__npi_greater_equal (#lhs := a .& #rhs := b .& Nil)

and_  a b = prim S.__npi_bitwise_and (#lhs := a .& #rhs := b .& Nil)
or_   a b = prim S.__npi_bitwise_or  (#lhs := a .& #rhs := b .& Nil)
xor_  a b = prim S.__npi_bitwise_xor (#lhs := a .& #rhs := b .& Nil)

bitwise_not, logical_not, invert :: (HasCallStack, PrimTensorOp t t) => t -> TensorMonad t t
bitwise_not a = prim S.__npi_bitwise_not  (#x := a .& Nil)
invert        = bitwise_not
logical_not a = prim S.__npi_logical_not  (#x := a .& Nil)

#if MXNET_VERSION == 10600
_adaptDouble = double2Float
#elif MXNET_VERSION >= 10700
_adaptDouble = id
#endif

addScalar, subScalar, rsubScalar, mulScalar, divScalar, rdivScalar ::
    (HasCallStack, PrimTensorOp t t) => Double -> t -> TensorMonad t t
addScalar  b a = prim S.__npi_add_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
subScalar  b a = prim S.__npi_subtract_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
rsubScalar b a = prim S.__npi_rsubtract_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
mulScalar  b a = prim S.__npi_multiply_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
divScalar  b a = prim S.__npi_true_divide_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
rdivScalar b a = prim S.__npi_rtrue_divide_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)

eqScalar, neqScalar, ltScalar, leqScalar, gtScalar, geqScalar ::
    (HasCallStack, PrimTensorOp t t) => Double -> t -> TensorMonad t t
eqScalar  b a = prim S.__npi_equal_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
neqScalar b a = prim S.__npi_not_equal_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
ltScalar  b a = prim S.__npi_less_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
leqScalar b a = prim S.__npi_less_equal_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
gtScalar  b a = prim S.__npi_greater_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
geqScalar b a = prim S.__npi_greater_equal_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)


andScalar, orScalar, xorScalar ::
    (HasCallStack, PrimTensorOp t t) => Double -> t -> TensorMonad t t
andScalar b a = prim S.__npi_bitwise_and_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
orScalar  b a = prim S.__npi_bitwise_or_scalar  (#data := a .& #scalar := _adaptDouble b .& Nil)
xorScalar b a = prim S.__npi_bitwise_xor_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)

-- addBroadcast = add_
-- subBroadcast = sub_
-- mulBroadcast = mul_
-- divBroadcast = div_
--
-- eqBroadcast  = eq_
-- neqBroadcast = neq_
-- ltBroadcast  = lt_
-- leqBroadcast = leq_
-- gtBroadcast  = gt_
-- geqBroadcast = geq_

ceil_, floor_, sqrt_, log2_, square_ ::
    (HasCallStack, PrimTensorOp t t) => t -> TensorMonad t t
ceil_   a = prim S.__npi_ceil   (#x := a .& Nil)
floor_  a = prim S.__npi_floor  (#x := a .& Nil)
sqrt_   a = prim S.__npi_sqrt   (#x := a .& Nil)
log2_   a = prim S.__npi_log2   (#x := a .& Nil)
square_ a = prim S.__npi_square (#x := a .& Nil)

sum_, max_ :: (HasCallStack, PrimTensorOp t t) => t -> Maybe [Int] -> Bool -> TensorMonad t t
sum_ s axis keepdims = prim S.__np_sum (#a := s .& #axis:= axis .& #keepdims := keepdims .& Nil)
max_ s axis keepdims = prim S.__np_max (#a := s .& #axis:= axis .& #keepdims := keepdims .& Nil)

argmax, argmin :: (HasCallStack, PrimTensorOp t t) => t -> Maybe Int -> Bool -> TensorMonad t t
argmax a axis keepdims = prim S.__npi_argmax (#data := a .& #axis := axis .& #keepdims := keepdims .& Nil)
argmin a axis keepdims = prim S.__npi_argmin (#data := a .& #axis := axis .& #keepdims := keepdims .& Nil)

einsum :: (HasCallStack, PrimTensorOp t t) => Text -> [t] -> Bool -> TensorMonad t t
einsum spec ts opt = prim S.__npi_einsum (#data := ts
                                       .& #num_args := length(ts)
                                       .& #subscripts := spec
                                       .& #optimize := fromEnum opt .& Nil)

identity :: (HasCallStack, PrimTensorOp t t) => t -> TensorMonad t t
identity a = prim S.__np_copy (#a := a .& Nil)

stack :: (HasCallStack, PrimTensorOp t t) => Int -> [t] -> TensorMonad t t
stack axis ts = prim S._stack (#num_args := length ts .& #data := ts .& #axis := axis .& Nil)

reshape :: (HasCallStack, PrimTensorOp t t) => [Int] -> t -> TensorMonad t t
reshape shape a = prim S.__npx_reshape (#a := a .& #newshape := shape .& Nil)

concat_ :: (HasCallStack, PrimTensorOp t t) => Int -> [t] -> TensorMonad t t
concat_ a s = prim S.__npi_concatenate (#data := s .& #num_args := length s .& #dim := a .& Nil)

takeI :: (HasCallStack, PrimTensorOp t t)
      => t -> t -> TensorMonad t t
takeI i a = prim S._take (#a := a .& #indices := i .& Nil)

pickI :: (HasCallStack, PrimTensorOp t t)
      => t -> t -> TensorMonad t t
pickI i t = prim S._pick (#data := t .& #index := i .& Nil)

where_ :: (HasCallStack, PrimTensorOp t t)
       => t -> t -> t -> TensorMonad t t
where_ c a b = prim S.__npi_where (#condition := c .& #x := a .& #y := b .& Nil)

squeeze :: (HasCallStack, PrimTensorOp t t) => Maybe [Int] -> t -> TensorMonad t t
squeeze axis a = prim S.__np_squeeze (#a := a .& #axis := axis .& Nil)

expandDims :: (HasCallStack, PrimTensorOp t t) => Int -> t -> TensorMonad t t
expandDims axis a = prim S._expand_dims (#data := a .& #axis := axis .& Nil)

-- broadcastAxis axis size a = prim S._broadcast_axis (#data := a .& #axis := axis .& #size := size .& Nil)
-- broadcastLike lhs rhs = prim S._broadcast_like (#lhs := lhs .& #rhs := rhs .& Nil)

transpose :: (HasCallStack, PrimTensorOp t t) => t -> [Int] -> TensorMonad t t
transpose a axes = prim S.__np_transpose (#a := a .& #axes := axes .& Nil)

-- TODO: slice will always create a copy of the target sub-region. This can be
--  not ideal if the range is continuous. Use `mxNDArraySlice` instead.
slice :: (HasCallStack, PrimTensorOp t t)
      => t -> [Int] -> [Int] -> TensorMonad t t
slice a beg end = prim S._slice (#data := a .& #begin := beg .& #end := end .& Nil)

-- sliceAxis :: (HasCallStack, PrimTensorOp t t)
--           => t -> Int -> Int -> Maybe Int -> TensorMonad t t
-- sliceAxis a axis beg end = prim S._slice_axis (#data := a .& #axis := axis .& #begin := beg .& #end := end .& Nil)

splitBySections :: (HasCallStack, PrimTensorOp t t)
                => Int -> Int -> Bool -> t -> TensorMonad t [t]
splitBySections num_sections axis squeeze s =
    primMulti S.__split_v2 (#data := s
                         .& #axis := axis
                         .& #indices := []
                         .& #sections := num_sections
                         .& #squeeze_axis := squeeze .& Nil)


hsplitBySections :: (HasCallStack, PrimTensorOp t t) => Int -> t -> TensorMonad t [t]
hsplitBySections num_sections s =
    primMulti S.__npi_hsplit (#data := s
                         .& #axis := 1
                         .& #indices := []
                         .& #sections := num_sections
                         .& #squeeze_axis := False .& Nil)

vsplitBySections, dsplitBySections :: (HasCallStack, PrimTensorOp t t) => Int -> t -> TensorMonad t [t]
vsplitBySections num_sections s = splitBySections num_sections 0 False s
dsplitBySections num_sections s = splitBySections num_sections 2 False s


-- TODO constraint the `o` to conform to `dt`
cast :: PrimTensorOp t o
#if MXNET_VERSION == 10600
     => EnumType '["bool", "float16", "float32", "float64", "int32",
                   "int64", "int8", "uint8"]
#elif MXNET_VERSION == 10700
     => EnumType '["bfloat16", "bool", "float16", "float32", "float64",
                   "int32", "int64", "int8", "uint8"]
#endif
     -> t
     -> TensorMonad o o
cast dt t = prim S._Cast (#dtype := dt .& #data := t .& Nil)

----------------------------------------------------------------------------
data LossAgg = AggMean | AggSum

sigmoidBCE :: (HasCallStack, PrimTensorOp t t, Monad (TensorMonad t))
           => t -> t -> Maybe t -> LossAgg -> TensorMonad t t
sigmoidBCE pred label sample_weight agg = do
    -- pred: (B, N, C, ..)
    -- label: (B, N, C, ..)
    -- sample_weight: (B, N, C, ..)

    a <- prim S._relu (#data := pred .& Nil)
    b <- mul_ pred label
    c <- prim S._abs (#data := pred .& Nil) >>= rsubScalar 0
    c <- prim S._Activation (#data := c .& #act_type := #softrelu .& Nil)
    loss <- add_ c =<< sub_ a b
    loss <- case sample_weight of
              Just w  -> mul_ loss w
              Nothing -> return loss
    case agg of
      AggMean -> prim S._mean (#data := loss .& #axis := Just [0] .& #exclude := True .& Nil)
      AggSum  -> prim S._sum  (#data := loss .& #axis := Just [0] .& #exclude := True .& Nil)

softmaxCE :: (HasCallStack, PrimTensorOp t t, Monad (TensorMonad t))
          => Int -> t -> t -> Maybe t -> TensorMonad t t
softmaxCE axis pred label sample_weight = do
    pred <- prim S._log_softmax (#data := pred .& #axis := axis .& Nil)
    labl <- prim S._reshape_like (#lhs := label .& #rhs := pred .& Nil)
    loss <- mul_ pred labl
    loss <- sum_ loss (Just [axis]) True >>= rsubScalar 0
    loss <- case sample_weight of
              Just w  -> mul_ loss w
              Nothing -> return loss
    prim S._mean (#data := loss .& #axis := Just [0] .& #exclude := True .& Nil)

-----------------------------------------------------------------------------
-- For NDArray Only
-----------------------------------------------------------------------------
copy :: (HasCallStack, PrimTensorOp t t, TensorApply t ~ (Maybe [t] -> IO [t]))
     => t -> t -> IO t
copy src dst = do
    [ret] <- S.__copyto (#data := src .& Nil) (Just [dst])
    return ret

#endif
