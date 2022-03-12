{-# LANGUAGE CPP              #-}
{-# LANGUAGE TypeApplications #-}
module MXNet.Base.Tensor.Functional where

import           Data.Constraint
import           Data.Typeable               (eqT, (:~:) (..))
import           GHC.Float                   (double2Float)
import           GHC.TypeLits                (KnownSymbol)
import           RIO
import           Type.Set                    (Insert)

import           MXNet.Base.NDArray          (NDArray)
import qualified MXNet.Base.Operators.Tensor as S
import           MXNet.Base.Spec.HMap        (HMap (..), (.&))
import           MXNet.Base.Spec.Operator
import           MXNet.Base.Tensor.Class
import           MXNet.Base.Types

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

pooling :: (PrimTensorOp t, DType u, Fullfilled "_Pooling" '(t, u) args)
        => ArgsHMap "_Pooling" '(t, u) args -> TensorMonad t (t u)
pooling = prim S._Pooling

_common_pool :: (PrimTensorOp t, DType u)
             => EnumType '["avg", "lp", "max", "sum"]
             -> t u -> [Int] -> Maybe [Int] -> Maybe [Int] -> TensorMonad t (t u)
_common_pool pool_type t kernel_size stride padding =
    let stride'  = fromMaybe kernel_size stride
        padding' = fromMaybe (replicate (length kernel_size) 0) padding
     in pooling (#data := t .& #kernel := kernel_size .& #stride := stride' .& #pad := padding' .& #pool_type := pool_type .& Nil)

maxPool :: (PrimTensorOp t, DType u)
        => t u -> [Int] -> Maybe [Int] -> Maybe [Int] -> TensorMonad t (t u)
maxPool = _common_pool #max


avgPool :: (PrimTensorOp t, DType u)
        => t u -> [Int] -> Maybe [Int] -> Maybe [Int] -> TensorMonad t (t u)
avgPool = _common_pool #avg

activation :: (PrimTensorOp t, DType u, Fullfilled "_Activation" '(t, u) args)
           => ArgsHMap "_Activation" '(t, u) args -> TensorMonad t (t u)
activation = prim S._Activation

relu a = prim S.__npx_relu (#data := a .& Nil)

sigmoid a = prim S.__npx_sigmoid (#data := a .& Nil)

softrelu a = activation (#data := a .& #act_type := #softrelu .& Nil)

softsign a = activation (#data := a .& #act_type := #softsign .& Nil)

tanh :: (PrimTensorOp t, DType u) => t u -> TensorMonad t (t u)
tanh a = prim S.__npi_tanh (#x := a .& Nil)

gelu a = prim S._LeakyReLU (#data := a .& #act_type := #gelu .& Nil)

selu a = prim S._LeakyReLU (#data := a .& #act_type := #selu .& Nil)

elu alpha a = prim S._LeakyReLU (#data := a .& #slope := alpha .& #act_type := #elu .& Nil)

leaky negative_slope a = prim S._LeakyReLU (#data := a .& #act_type := #leaky .& #slope := negative_slope .& Nil)

rrelu lower_bound upper_bound a = prim S._LeakyReLU (#data := a .& #act_type := #rrelu .& #lower_bound := lower_bound .& #upper_bound := upper_bound .& Nil)

softmax :: (PrimTensorOp t, DType u, Fullfilled "_softmax" '(t, u) args)
        => ArgsHMap "_softmax" '(t, u) args -> TensorMonad t (t u)
softmax = prim S._softmax

softmaxoutput :: (PrimTensorOp t, DType u, Fullfilled "_SoftmaxOutput" '(t, u) args)
              => ArgsHMap "_SoftmaxOutput" '(t, u) args -> TensorMonad t (t u)
softmaxoutput = prim S._SoftmaxOutput

flatten t = prim S._Flatten (#data := t .& Nil)
dropout t p = prim S._Dropout (#data := t .& #p := p .& Nil)

-- | create a new tensor like the provided one
-- The dtype will be the same as the input.
fullLike :: (HasCallStack, PrimTensorOp t, DType u) => Double -> t u -> TensorMonad t (t u)
fullLike v a = prim S.__npi_full_like (#a := a .& #fill_value := v .& Nil)

zerosLike, onesLike :: (HasCallStack, PrimTensorOp t, DType u) => t u -> TensorMonad t (t u)
zerosLike = fullLike 0
onesLike  = fullLike 1

ones, zeros :: (HasCallStack, PrimTensorOp t, DType u, KnownSymbol dty, InEnum dty AllDTypes, DTypeName u ~ dty)
            => Proxy dty -> [Int] -> TensorMonad t (t u)
zeros dty shp = prim S.__npi_zeros (#shape := shp .& #dtype := EnumType dty .& Nil)
ones  dty shp = prim S.__npi_ones  (#shape := shp .& #dtype := EnumType dty .& Nil)

onesF, zerosF :: forall t u dty . (HasCallStack, PrimTensorOp t, FloatDType u, KnownSymbol dty, DTypeName u ~ dty)
            => Proxy dty -> [Int] -> TensorMonad t (t u)
onesF  = case enumWeaken @FloatDTypes @AllDTypes @dty of
           Sub Dict -> ones
zerosF = case enumWeaken @FloatDTypes @AllDTypes @dty of
           Sub Dict -> zeros

eye :: (HasCallStack, PrimTensorOp t, DType u, KnownSymbol dty, InEnum dty AllDTypes, DTypeName u ~ dty)
    => Proxy dty -> [Int] -> TensorMonad t (t u)
eye dtype shape = prim S.__npi_identity (#shape := shape .& #dtype := EnumType dtype .& Nil)

eyeF :: forall t u dty . (HasCallStack, PrimTensorOp t, FloatDType u, KnownSymbol dty, DTypeName u ~ dty)
    => Proxy dty -> [Int] -> TensorMonad t (t u)
eyeF = case enumWeaken @FloatDTypes @AllDTypes @dty of
         Sub Dict -> eye

arange :: (HasCallStack, PrimTensorOp t, DType u, KnownSymbol dty, InEnum dty NumericDTypes, DTypeName u ~ dty)
       => Proxy dty -> Double -> Maybe Double -> Maybe Double -> TensorMonad t (t u)
arange dtype start stop step =
    let args = #start := start .& #stop := stop .& #dtype := EnumType dtype .& Nil
     in case step of
          Nothing -> prim S.__npi_arange args
          Just st -> prim S.__npi_arange (#step := st .& args)

arangeF :: forall t u dty . (HasCallStack, PrimTensorOp t, FloatDType u, KnownSymbol dty, DTypeName u ~ dty)
        => Proxy dty -> Double -> Maybe Double -> Maybe Double -> TensorMonad t (t u)
arangeF = case enumWeaken @FloatDTypes @NumericDTypes @dty of
            Sub Dict -> arange

addNoBroadcast, subNoBroadcast, mulNoBroadcast, divNoBroadcast ::
    (HasCallStack, PrimTensorOp t, DType u) => t u -> t u -> TensorMonad t (t u)
addNoBroadcast a b = prim S._elemwise_add (#lhs := a .& #rhs := b .& Nil)
subNoBroadcast a b = prim S._elemwise_sub (#lhs := a .& #rhs := b .& Nil)
mulNoBroadcast a b = prim S._elemwise_mul (#lhs := a .& #rhs := b .& Nil)
divNoBroadcast a b = prim S._elemwise_div (#lhs := a .& #rhs := b .& Nil)

add_, sub_, mul_, div_, and_, or_, xor_ ::
    (HasCallStack, PrimTensorOp t, DType u) => t u -> t u -> TensorMonad t (t u)
add_ a b = prim S.__npi_add         (#lhs := a .& #rhs := b .& Nil)
sub_ a b = prim S.__npi_subtract    (#lhs := a .& #rhs := b .& Nil)
mul_ a b = prim S.__npi_multiply    (#lhs := a .& #rhs := b .& Nil)
div_ a b = prim S.__npi_true_divide (#lhs := a .& #rhs := b .& Nil)
and_  a b = prim S.__npi_bitwise_and (#lhs := a .& #rhs := b .& Nil)
or_   a b = prim S.__npi_bitwise_or  (#lhs := a .& #rhs := b .& Nil)
xor_  a b = prim S.__npi_bitwise_xor (#lhs := a .& #rhs := b .& Nil)

eq_, neq_, lt_, leq_, gt_, geq_ ::
    (HasCallStack, PrimTensorOp t, DType u)
    => t u -> t u -> TensorMonad t (t Bool)
eq_   a b = prim S.__npi_equal      (#lhs := a .& #rhs := b .& Nil)
neq_  a b = prim S.__npi_not_equal  (#lhs := a .& #rhs := b .& Nil)
lt_   a b = prim S.__npi_less       (#lhs := a .& #rhs := b .& Nil)
leq_  a b = prim S.__npi_less_equal (#lhs := a .& #rhs := b .& Nil)
gt_   a b = prim S.__npi_greater    (#lhs := a .& #rhs := b .& Nil)
geq_  a b = prim S.__npi_greater_equal (#lhs := a .& #rhs := b .& Nil)

bitwise_not, logical_not, invert ::
    (HasCallStack, PrimTensorOp t, DType u) => t u -> TensorMonad t (t u)
bitwise_not a = prim S.__npi_bitwise_not  (#x := a .& Nil)
invert        = bitwise_not
logical_not a = prim S.__npi_logical_not  (#x := a .& Nil)

#if MXNET_VERSION == 10600
_adaptDouble = double2Float
#elif MXNET_VERSION >= 10700
_adaptDouble = id
#endif

addScalar, subScalar, rsubScalar, mulScalar, divScalar, rdivScalar ::
    (HasCallStack, PrimTensorOp t, DType u) => Double -> t u -> TensorMonad t (t u)
addScalar  b a = prim S.__npi_add_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
subScalar  b a = prim S.__npi_subtract_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
rsubScalar b a = prim S.__npi_rsubtract_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
mulScalar  b a = prim S.__npi_multiply_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
divScalar  b a = prim S.__npi_true_divide_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
rdivScalar b a = prim S.__npi_rtrue_divide_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)

eqScalar, neqScalar, ltScalar, leqScalar, gtScalar, geqScalar ::
    (HasCallStack, PrimTensorOp t, DType u)
    => Double -> t u -> TensorMonad t (t Bool)
eqScalar  b a = prim S.__npi_equal_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
neqScalar b a = prim S.__npi_not_equal_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
ltScalar  b a = prim S.__npi_less_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
leqScalar b a = prim S.__npi_less_equal_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
gtScalar  b a = prim S.__npi_greater_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
geqScalar b a = prim S.__npi_greater_equal_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)


andScalar, orScalar, xorScalar ::
    (HasCallStack, PrimTensorOp t, DType u) => Double -> t u -> TensorMonad t (t u)
andScalar b a = prim S.__npi_bitwise_and_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)
orScalar  b a = prim S.__npi_bitwise_or_scalar  (#data := a .& #scalar := _adaptDouble b .& Nil)
xorScalar b a = prim S.__npi_bitwise_xor_scalar (#data := a .& #scalar := _adaptDouble b .& Nil)

ceil_, floor_, sqrt_, log2_, square_ ::
    (HasCallStack, PrimTensorOp t, DType u) => t u -> TensorMonad t (t u)
ceil_   a = prim S.__npi_ceil   (#x := a .& Nil)
floor_  a = prim S.__npi_floor  (#x := a .& Nil)
sqrt_   a = prim S.__npi_sqrt   (#x := a .& Nil)
log2_   a = prim S.__npi_log2   (#x := a .& Nil)
square_ a = prim S.__npi_square (#x := a .& Nil)

mean, sum_, max_ :: (HasCallStack, PrimTensorOp t, DType u)
                 => t u -> Maybe [Int] -> Bool -> TensorMonad t (t u)
sum_ s axis keepdims = prim S.__np_sum (#a := s .& #axis:= axis .& #keepdims := keepdims .& Nil)
max_ s axis keepdims = prim S.__np_max (#a := s .& #axis:= axis .& #keepdims := keepdims .& Nil)
mean a axis keepdims = prim S.__npi_mean (#a := a .& #axis := axis .& #keepdims := keepdims .& Nil)

argmax, argmin :: (HasCallStack, PrimTensorOp t, DType u)
               => t u -> Maybe Int -> Bool -> TensorMonad t (t Int64)
argmax a axis keepdims = prim S.__npi_argmax (#data := a .& #axis := axis .& #keepdims := keepdims .& Nil)
argmin a axis keepdims = prim S.__npi_argmin (#data := a .& #axis := axis .& #keepdims := keepdims .& Nil)

einsum :: (HasCallStack, PrimTensorOp t, DType u)
       => Text -> [t u] -> Bool -> TensorMonad t (t u)
einsum spec ts opt = prim S.__npi_einsum (#data := ts
                                       .& #num_args := length(ts)
                                       .& #subscripts := spec
                                       .& #optimize := fromEnum opt .& Nil)

abs_, relu, sigmoid :: (HasCallStack, PrimTensorOp t, DType u) => t u -> TensorMonad t (t u)
abs_ a = prim S.__npi_absolute (#x := a .& Nil)

logSoftmax :: (HasCallStack, PrimTensorOp t, DType u) => t u -> Int -> Maybe Double -> TensorMonad t (t u)
logSoftmax a axis temp = prim S._log_softmax (#data := a .& #axis := axis .& #temperature := temp .& Nil)

identity :: (HasCallStack, PrimTensorOp t, DType u) => t u -> TensorMonad t (t u)
identity a = prim S.__np_copy (#a := a .& Nil)

stack :: (HasCallStack, PrimTensorOp t, DType u) => Int -> [t u] -> TensorMonad t (t u)
stack axis ts = prim S._stack (#num_args := length ts .& #data := ts .& #axis := axis .& Nil)

reshape :: (HasCallStack, PrimTensorOp t, DType u) => [Int] -> t u -> TensorMonad t (t u)
reshape shape a = prim S.__npx_reshape (#a := a .& #newshape := shape .& Nil)

-- reshapeLegacy has a few magic numbers:
-- *  0 copy this dimension from the input to the output shape.
-- * -1 infers the dimension of the output shape by using the remainder of the input dimensions keeping the size of the new array same as that of the input array. At most one dimension of shape can be -1.
-- * -2 copy all/remainder of the input dimensions to the output shape.
-- * -3 use the product of two consecutive dimensions of the input shape as the output dimension.
-- * -4 split one dimension of the input into two dimensions passed subsequent to -4 in shape (can contain -1).
reshapeLegacy :: (HasCallStack, PrimTensorOp t, DType u) => [Int] -> t u -> TensorMonad t (t u)
reshapeLegacy shape a = prim S._Reshape (#data := a .& #shape := shape .& Nil)

reshapeLike :: (HasCallStack, PrimTensorOp t, DType u)
            => t u -> Maybe Int -> Maybe Int
            -> t u -> Maybe Int -> Maybe Int
            -> TensorMonad t (t u)
reshapeLike lhs lhs_beg lhs_end rhs rhs_beg rhs_end =
    prim S._reshape_like (#lhs := lhs .& #lhs_begin := lhs_beg .& #lhs_end := lhs_end
                       .& #rhs := rhs .& #rhs_begin := rhs_beg .& #rhs_end := rhs_end .& Nil)

concat_ :: (HasCallStack, PrimTensorOp t, DType u) => Int -> [t u] -> TensorMonad t (t u)
concat_ a s = prim S.__npi_concatenate (#data := s .& #num_args := length s .& #axis := a .& Nil)

takeI :: (HasCallStack, PrimTensorOp t, DType u)
      => t u -> t u -> TensorMonad t (t u)
takeI i a = prim S._take (#a := a .& #indices := i .& Nil)

pick :: (HasCallStack, PrimTensorOp t, DType u)
      => Maybe Int -> t u -> t u -> TensorMonad t (t u)
pick a i t = prim S._pick (#data := t .& #index := i .& #axis := a .& Nil)

gather :: (HasCallStack, PrimTensorOp t, DType u)
       => t u -> t u -> TensorMonad t (t u)
gather a i = prim S._gather_nd (#data := a .& #indices := i .& Nil)

where_ :: (HasCallStack, PrimTensorOp t, DType u)
       => t Bool -> t u -> t u -> TensorMonad t (t u)
where_ c a b = prim S.__npi_where (#condition := c .& #x := a .& #y := b .& Nil)

squeeze :: (HasCallStack, PrimTensorOp t, DType u) => Maybe [Int] -> t u -> TensorMonad t (t u)
squeeze axis a = prim S.__np_squeeze (#a := a .& #axis := axis .& Nil)

expandDims :: (HasCallStack, PrimTensorOp t, DType u) => Int -> t u -> TensorMonad t (t u)
expandDims axis a = prim S._expand_dims (#data := a .& #axis := axis .& Nil)

broadcastAxis axis size a = prim S._broadcast_axis (#data := a .& #axis := axis .& #size := size .& Nil)
broadcastLike lhs rhs = prim S._broadcast_like (#lhs := lhs .& #rhs := rhs .& Nil)
broadcastLikeAxis :: (HasCallStack, PrimTensorOp t, DType u)
                  => (t u, [Int]) -> (t u, [Int]) -> TensorMonad t (t u)
broadcastLikeAxis (lhs, la) (rhs, ra) =
    prim S._broadcast_like
        (#lhs := lhs .& #rhs := rhs .& #lhs_axes := Just la .& #rhs_axes := Just ra .& Nil)

transpose :: (HasCallStack, PrimTensorOp t, DType u) => t u -> [Int] -> TensorMonad t (t u)
transpose a axes = prim S.__np_transpose (#a := a .& #axes := axes .& Nil)

scatter :: forall t u. (HasCallStack, PrimTensorOp t, DType u, Typeable u, InEnum (DTypeName u) AllDTypes)
        => t Int64 -> t u -> [Int] -> TensorMonad t (t u)
scatter indices values shape = do
    case eqT @Int64 @u of
      Just Refl ->
          prim S._scatter_nd (#shape := shape .& #data := values .& #indices := indices .& Nil)
      Nothing -> do
          indices <- cast (Proxy :: Proxy (DTypeName u)) indices
          prim S._scatter_nd (#shape := shape .& #data := values .& #indices := indices .& Nil)

-- TODO: slice will always create a copy of the target sub-region. This can be
--  not ideal if the range is continuous. Use `mxNDArraySlice` instead.
slice :: (HasCallStack, PrimTensorOp t, DType u)
      => t u -> [Int] -> [Int] -> TensorMonad t (t u)
slice a beg end = prim S._slice (#data := a .& #begin := beg .& #end := end .& Nil)

sliceAxis :: (HasCallStack, PrimTensorOp t, DType u)
          => t u -> Int -> Int -> Maybe Int -> TensorMonad t (t u)
sliceAxis a axis beg end = prim S._slice_axis (#data := a .& #axis := axis .& #begin := beg .& #end := end .& Nil)

splitBySections :: (HasCallStack, PrimTensorOp t, DType u)
                => Int -> Int -> Bool -> t u -> TensorMonad t [t u]
splitBySections num_sections axis squeeze s =
    primMulti S.__split_v2 (#data := s
                         .& #axis := axis
                         .& #indices := []
                         .& #sections := num_sections
                         .& #squeeze_axis := squeeze .& Nil)


hsplitBySections :: (HasCallStack, PrimTensorOp t, DType u)
                 => Int -> t u -> TensorMonad t [t u]
hsplitBySections num_sections s =
    primMulti S.__npi_hsplit (#data := s
                         .& #axis := 1
                         .& #indices := []
                         .& #sections := num_sections
                         .& #squeeze_axis := False .& Nil)

vsplitBySections, dsplitBySections :: (HasCallStack, PrimTensorOp t, DType u)
                                   => Int -> t u -> TensorMonad t [t u]
vsplitBySections num_sections s = splitBySections num_sections 0 False s
dsplitBySections num_sections s = splitBySections num_sections 2 False s

repeat_ :: (HasCallStack, PrimTensorOp t, DType u) => Int -> Maybe Int -> t u -> TensorMonad t (t u)
repeat_ n a t = prim S._repeat (#data := t .& #repeats := n .& #axis := a .& Nil)

cast :: (HasCallStack, PrimTensorOp t, DType u, DType v, KnownSymbol dty, InEnum dty AllDTypes, DTypeName v ~ dty)
     => Proxy dty
     -> t u
     -> TensorMonad t (t v)
cast dty t = prim S._Cast (#dtype := EnumType dty .& #data := t .& Nil)

-- | restricted form of `cast`. When only having the knowledge that the `dty` belongs
--   to `FloatDTypes`, `cast` is not useable because of the constraint
--   `InEnum dty AllDTypes`. In this case, `castToFloat` helps.
castToFloat :: forall dty t u v . (HasCallStack, PrimTensorOp t, DType u, DType v,
                                   KnownSymbol dty, InEnum dty FloatDTypes, DTypeName v ~ dty)
            => t u -> TensorMonad t (t v)
castToFloat t = case enumWeaken @FloatDTypes @AllDTypes @dty of
                  Sub Dict -> cast (Proxy :: Proxy dty) t

-- | resetricted form of `cast`. Similarly to `castToFloat`, it is applicable when
--   only having the knowledge that the `dty` belongs to `NumericDTypes`.
castToNum :: forall dty t u v . (HasCallStack, PrimTensorOp t, DType u, DType v,
                                 KnownSymbol dty, InEnum dty NumericDTypes, DTypeName v ~ dty)
          => t u -> TensorMonad t (t v)
castToNum t = case enumWeaken @NumericDTypes @AllDTypes @dty of
                  Sub Dict -> cast (Proxy :: Proxy dty) t

----------------------------------------------------------------------------
data LossAgg = AggMean | AggSum | NoAgg

sigmoidBCE :: (HasCallStack, PrimTensorOp t, DType u)
           => t u -> t u -> Maybe (t u) -> LossAgg -> TensorMonad t (t u)
sigmoidBCE pred label sample_weight agg = do
    -- pred: (B, N, C, ..)
    -- label: (B, N, C, ..)
    -- sample_weight: (B, N, C, ..)

    -- Note that mulNoBroadcast ensures the agreements of shapes

    a <- relu pred
    b <- mulNoBroadcast pred label
    c <- abs_ pred >>= rsubScalar 0
    c <- prim S._Activation (#data := c .& #act_type := #softrelu .& Nil)
    loss <- add_ c =<< sub_ a b
    loss <- case sample_weight of
              Just w  -> mulNoBroadcast loss w
              Nothing -> return loss
    case agg of
      AggMean -> mean loss Nothing False
      AggSum  -> sum_ loss Nothing False
      NoAgg   -> return loss

softmaxCE :: (HasCallStack, PrimTensorOp t, DType u)
          => Int -> t u -> t u -> Maybe (t u) -> TensorMonad t (t u)
softmaxCE axis pred label sample_weight = do
    pred <- logSoftmax pred axis Nothing
    loss <- mulNoBroadcast pred label
    loss <- sum_ loss (Just [axis]) True >>= rsubScalar 0
    loss <- case sample_weight of
              Just w  -> mulNoBroadcast loss w
              Nothing -> return loss
    mean loss Nothing False

-----------------------------------------------------------------------------
-- For NDArray Only
-----------------------------------------------------------------------------
copy :: (HasCallStack, DType u) => NDArray u -> NDArray u -> IO (NDArray u)
copy src dst = do
    [ret] <- S.__copyto (#data := src .& Nil) (Just [dst])
    return ret

#endif
