module MXNet.Base.Tensor.Functional where

import           RIO

import qualified MXNet.Base.Operators.Tensor as S
import           MXNet.Base.Spec.HMap        (HMap (..), (.&))
import           MXNet.Base.Spec.Operator
import           MXNet.Base.Tensor.Class

-----------------------------------------------------------------------------
-- For both Symbol and NDArray
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

pick :: (PrimTensorOp t t, Fullfilled "_pick" t args)
     => ArgsHMap "_pick" t args -> TensorMonad t t
pick = prim S._pick

stack axis ts = prim S._stack (#num_args := length ts .& #data := ts .& #axis := axis .& Nil)
flatten t = prim S._Flatten (#data := t .& Nil)
identity s = prim S.__copy (#data := s .& Nil)
dropout t p = prim S._Dropout (#data := t .& #p := p .& Nil)
reshape shape a = prim S._Reshape (#data := a .& #shape := shape .& Nil)

add_, sub_, mul_, div_, eq_, neq_, lt_, leq_, gt_, geq_ ::
    PrimTensorOp t t => t -> t -> TensorMonad t t
add_ a b = prim S._elemwise_add (#lhs := a .& #rhs := b .& Nil)
sub_ a b = prim S._elemwise_sub (#lhs := a .& #rhs := b .& Nil)
mul_ a b = prim S._elemwise_mul (#lhs := a .& #rhs := b .& Nil)
div_ a b = prim S._elemwise_div (#lhs := a .& #rhs := b .& Nil)

eq_   a b = prim S.__equal (#lhs := a .& #rhs := b .& Nil)
neq_  a b = prim S.__not_equal (#lhs := a .& #rhs := b .& Nil)
lt_   a b = prim S.__lesser (#lhs := a .& #rhs := b .& Nil)
leq_  a b = prim S.__lesser_equal (#lhs := a .& #rhs := b .& Nil)
gt_   a b = prim S.__greater (#lhs := a .& #rhs := b .& Nil)
geq_  a b = prim S.__greater_equal (#lhs := a .& #rhs := b .& Nil)

and_  a b = prim S.__logical_and (#lhs := a .& #rhs := b .& Nil)
or_   a b = prim S.__logical_or  (#lhs := a .& #rhs := b .& Nil)
xor_  a b = prim S.__logical_xor (#lhs := a .& #rhs := b .& Nil)
not_  a   = prim S._logical_not  (#data := a .& Nil)

addScalar  b a = prim S.__plus_scalar (#data := a .& #scalar := b .& Nil)
subScalar  b a = prim S.__minus_scalar (#data := a .& #scalar := b .& Nil)
rsubScalar b a = prim S.__rminus_scalar (#data := a .& #scalar := b .& Nil)
mulScalar  b a = prim S.__mul_scalar (#data := a .& #scalar := b .& Nil)
divScalar  b a = prim S.__div_scalar (#data := a .& #scalar := b .& Nil)
rdivScalar b a = prim S.__rdiv_scalar (#data := a .& #scalar := b .& Nil)

eqScalar  b a = prim S.__equal_scalar (#data := a .& #scalar := b .& Nil)
neqScalar b a = prim S.__not_equal_scalar (#data := a .& #scalar := b .& Nil)
ltScalar  b a = prim S.__lesser_scalar (#data := a .& #scalar := b .& Nil)
leqScalar b a = prim S.__lesser_equal_scalar (#data := a .& #scalar := b .& Nil)
gtScalar  b a = prim S.__greater_scalar (#data := a .& #scalar := b .& Nil)
geqScalar b a = prim S.__greater_equal_scalar (#data := a .& #scalar := b .& Nil)

andScalar b a = prim S.__logical_and_scalar (#data := a .& #scalar := b .& Nil)
orScalar  b a = prim S.__logical_or_scalar  (#data := a .& #scalar := b .& Nil)
xorScalar b a = prim S.__logical_xor_scalar (#data := a .& #scalar := b .& Nil)

addBroadcast a b = prim S._broadcast_add (#lhs := a .& #rhs := b .& Nil)
subBroadcast a b = prim S._broadcast_sub (#lhs := a .& #rhs := b .& Nil)
mulBroadcast a b = prim S._broadcast_mul (#lhs := a .& #rhs := b .& Nil)
divBroadcast a b = prim S._broadcast_div (#lhs := a .& #rhs := b .& Nil)

eqBroadcast  a b = prim S._broadcast_equal (#lhs := a .& #rhs := b .& Nil)
neqBroadcast a b = prim S._broadcast_not_equal (#lhs := a .& #rhs := b .& Nil)
ltBroadcast  a b = prim S._broadcast_lesser (#lhs := a .& #rhs := b .& Nil)
leqBroadcast a b = prim S._broadcast_lesser_equal (#lhs := a .& #rhs := b .& Nil)
gtBroadcast  a b = prim S._broadcast_greater (#lhs := a .& #rhs := b .& Nil)
geqBroadcast a b = prim S._broadcast_greater_equal (#lhs := a .& #rhs := b .& Nil)

ceil_   a = prim S._ceil   (#data := a .& Nil)
floor_  a = prim S._floor  (#data := a .& Nil)
sqrt_   a = prim S._sqrt   (#data := a .& Nil)
log2_   a = prim S._log2   (#data := a .& Nil)
square_ a = prim S._square (#data := a .& Nil)

concat_ :: PrimTensorOp t t => Int -> [t] -> TensorMonad t t
concat_ d s = prim S._Concat (#data := s .& #num_args := length s .& #dim := d .& Nil)

takeI :: (HasCallStack, PrimTensorOp t t)
      => t -> t -> TensorMonad t t
takeI i a = prim S._take (#a := a .& #indices := i .& Nil)

pickI :: (HasCallStack, PrimTensorOp t t)
      => t -> t -> TensorMonad t t
pickI i t = prim S._pick (#data := t .& #index := i .& Nil)

where_ c a b = prim S._where (#condition := c .& #x := a .& #y := b .& Nil)

zerosLike a = prim S._zeros_like (#data := a .& Nil)
onesLike  a = prim S._ones_like  (#data := a .& Nil)

squeeze axis a = prim S._squeeze (#data := a .& #axis := axis .& Nil)
expandDims axis a = prim S._expand_dims (#data := a .& #axis := axis .& Nil)

broadcastAxis axis size a = prim S._broadcast_axis (#data := a .& #axis := axis .& #size := size .& Nil)
broadcastLike lhs rhs = prim S._broadcast_like (#lhs := lhs .& #rhs := rhs .& Nil)

sum_ s axis keepdims = prim S._sum (#data := s .& #axis:= axis .& #keepdims := keepdims .& Nil)

max_ s axis keepdims = prim S._max (#data := s .& #axis:= axis .& #keepdims := keepdims .& Nil)

transpose a axes = prim S._transpose (#data := a .& #axes := axes .& Nil)

argmax a axis keepdims = prim S._argmax (#data := a .& #axis := axis .& #keepdims := keepdims .& Nil)

sliceAxis a axis beg end = prim S._slice_axis (#data := a .& #axis := axis .& #begin := beg .& #end := end .& Nil)

splitBySections num_sections axis squeeze s =
    primMulti S.__split_v2 (#data := s
                         .& #axis := axis
                         .& #indices := []
                         .& #sections := num_sections
                         .& #squeeze_axis := squeeze .& Nil)

-- TODO constraint the `o` to conform to `dt`
cast :: PrimTensorOp t o
     => EnumType '["bool", "float16", "float32", "float64", "int32", "int64", "int8", "uint8"]
     -> t
     -> TensorMonad o o
cast dt t = prim S._Cast (#dtype := dt .& #data := t .& Nil)

----------------------------------------------------------------------------
data LossAgg = AggMean | AggSum

sigmoidBCE :: (PrimTensorOp t t, Monad (TensorMonad t))
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
              Just w  -> mulBroadcast loss w
              Nothing -> return loss
    case agg of
      AggMean -> prim S._mean (#data := loss .& #axis := Just [0] .& #exclude := True .& Nil)
      AggSum  -> prim S._sum  (#data := loss .& #axis := Just [0] .& #exclude := True .& Nil)

softmaxCE :: (PrimTensorOp t t, Monad (TensorMonad t)) => Int -> t -> t -> Maybe t -> TensorMonad t t
softmaxCE axis pred label sample_weight = do
    pred <- prim S._log_softmax (#data := pred .& #axis := axis .& Nil)
    labl <- prim S._reshape_like (#lhs := label .& #rhs := pred .& Nil)
    loss <- mul_ pred labl
    loss <- sum_ loss (Just [axis]) True >>= rsubScalar 0
    loss <- case sample_weight of
              Just w  -> mulBroadcast loss w
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

