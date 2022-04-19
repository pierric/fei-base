{-# LANGUAGE CPP              #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -fplugin=Data.Record.Anon.Plugin #-}
module MXNet.Base.Tensor.Functional where

import           Data.Constraint
import           Data.Record.Anon.Simple     (Record)
import qualified Data.Record.Anon.Simple     as Anon
import           Data.Typeable               (eqT, (:~:) (..))
import           GHC.Float                   (double2Float)
import           GHC.TypeLits                (KnownSymbol)
import           RIO

import           MXNet.Base.Core.Enum
import           MXNet.Base.Core.Spec
import           MXNet.Base.NDArray          (NDArray)
import qualified MXNet.Base.Operators.Tensor as S
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

pooling :: (PrimTensorOp t, DType u, FieldsAcc (S.ParameterList_Pooling t u) r)
        => Record r -> TensorMonad t (t u)
pooling = prim S._Pooling

_common_pool :: (PrimTensorOp t, DType u)
             => EnumType '["avg", "lp", "max", "sum"]
             -> Bool
             -> t u -> [Int] -> Maybe [Int] -> Maybe [Int] -> TensorMonad t (t u)
_common_pool pool_type global_pool t kernel_size stride padding =
    let stride'  = fromMaybe kernel_size stride
        padding' = fromMaybe (replicate (length kernel_size) 0) padding
     in pooling ANON{_data = t, kernel = Just kernel_size, stride = stride, pad = padding, pool_type = Just pool_type, global_pool = Just global_pool}

maxPool :: (PrimTensorOp t, DType u)
        => t u -> [Int] -> Maybe [Int] -> Maybe [Int] -> TensorMonad t (t u)
maxPool = _common_pool #max False

avgPool :: (PrimTensorOp t, DType u)
        => t u -> [Int] -> Maybe [Int] -> Maybe [Int] -> TensorMonad t (t u)
avgPool = _common_pool #avg False

globalMaxPool :: (PrimTensorOp t, DType u)
              => t u -> [Int] -> Maybe [Int] -> Maybe [Int] -> TensorMonad t (t u)
globalMaxPool = _common_pool #max True

globalAvgPool :: (PrimTensorOp t, DType u)
              => t u -> [Int] -> Maybe [Int] -> Maybe [Int] -> TensorMonad t (t u)
globalAvgPool = _common_pool #avg True

activation :: (PrimTensorOp t, DType u, FieldsAcc (S.ParameterList_Activation t u) r)
           => Record r -> TensorMonad t (t u)
activation = prim S._Activation

relu, sigmoid, softrelu, softsign, gelu, selu :: (HasCallStack, PrimTensorOp t, DType u) => t u -> TensorMonad t (t u)
relu a = prim S.__npx_relu ANON{_data = a}
sigmoid a = prim S.__npx_sigmoid ANON{_data = a}
softrelu a = activation ANON{_data = a, act_type = #softrelu}
softsign a = activation ANON{_data = a, act_type = #softsign}

tanh :: (PrimTensorOp t, DType u) => t u -> TensorMonad t (t u)
tanh a = prim S.__npi_tanh ANON{x = Just a}

gelu a = prim S._LeakyReLU ANON{_data = a, act_type = Just $ #gelu}
selu a = prim S._LeakyReLU ANON{_data = a, act_type = Just $ #selu}

elu :: (HasCallStack, PrimTensorOp t, DType u) => Float -> t u -> TensorMonad t (t u)
elu alpha a = prim S._LeakyReLU ANON{_data = a, slope = Just alpha, act_type = Just #elu}

leaky :: (HasCallStack, PrimTensorOp t, DType u) => Float -> t u -> TensorMonad t (t u)
leaky negative_slope a = prim S._LeakyReLU ANON{_data = a, act_type = Just #leaky, slope = Just negative_slope}

rrelu :: (HasCallStack, PrimTensorOp t, DType u) => Float -> Float -> t u -> TensorMonad t (t u)
rrelu lower_bound upper_bound a =
    prim S._LeakyReLU ANON{_data = a, act_type = Just #rrelu, lower_bound = Just lower_bound, upper_bound = Just upper_bound}

softmax :: (PrimTensorOp t, DType u, FieldsAcc (S.ParameterList_softmax t u) r)
        => Record r -> TensorMonad t (t u)
softmax = prim S._softmax

softmaxoutput :: (PrimTensorOp t, DType u, FieldsAcc (S.ParameterList_SoftmaxOutput t u) r)
              => Record r -> TensorMonad t (t u)
softmaxoutput = prim S._SoftmaxOutput

flatten t = prim S._Flatten ANON{_data = t}
dropout t p = prim S._Dropout ANON{_data = t, p = p}

-- | create a new tensor like the provided one
-- The dtype will be the same as the input.
fullLike :: (HasCallStack, PrimTensorOp t, DType u) => Double -> t u -> TensorMonad t (t u)
fullLike v a = prim S.__npi_full_like ANON{a = Just a, fill_value = v}

zerosLike, onesLike :: (HasCallStack, PrimTensorOp t, DType u) => t u -> TensorMonad t (t u)
zerosLike = fullLike 0
onesLike  = fullLike 1

ones, zeros :: (HasCallStack, PrimTensorOp t, DType u, KnownSymbol dty, InEnum dty AllDTypes, DTypeName u ~ dty)
            => Proxy dty -> [Int] -> TensorMonad t (t u)
zeros dty shp = prim S.__npi_zeros ANON{shape = Just shp, dtype = Just $ EnumType dty}
ones  dty shp = prim S.__npi_ones  ANON{shape = Just shp, dtype = Just $ EnumType dty}

onesF, zerosF :: forall t u dty . (HasCallStack, PrimTensorOp t, FloatDType u, KnownSymbol dty, DTypeName u ~ dty)
            => Proxy dty -> [Int] -> TensorMonad t (t u)
onesF  = case enumWeaken @FloatDTypes @AllDTypes @dty of
           Sub Dict -> ones
zerosF = case enumWeaken @FloatDTypes @AllDTypes @dty of
           Sub Dict -> zeros

eye :: (HasCallStack, PrimTensorOp t, DType u, KnownSymbol dty, InEnum dty AllDTypes, DTypeName u ~ dty)
    => Proxy dty -> [Int] -> TensorMonad t (t u)
eye dtype shape = prim S.__npi_identity ANON{shape = Just shape, dtype = Just $ EnumType dtype}

eyeF :: forall t u dty . (HasCallStack, PrimTensorOp t, FloatDType u, KnownSymbol dty, DTypeName u ~ dty)
    => Proxy dty -> [Int] -> TensorMonad t (t u)
eyeF = case enumWeaken @FloatDTypes @AllDTypes @dty of
         Sub Dict -> eye

arange :: (HasCallStack, PrimTensorOp t, DType u, KnownSymbol dty, InEnum dty NumericDTypes, DTypeName u ~ dty)
       => Proxy dty -> Double -> Maybe Double -> Maybe Double -> TensorMonad t (t u)
arange dtype start stop step =
    let args = ANON{start = start, stop = Just stop, dtype = Just $ EnumType dtype}
     in case step of
          Nothing -> prim S.__npi_arange args
          Just st -> prim S.__npi_arange (Anon.insert #step (Just st) args)

arangeF :: forall t u dty . (HasCallStack, PrimTensorOp t, FloatDType u, KnownSymbol dty, DTypeName u ~ dty)
        => Proxy dty -> Double -> Maybe Double -> Maybe Double -> TensorMonad t (t u)
arangeF = case enumWeaken @FloatDTypes @NumericDTypes @dty of
            Sub Dict -> arange

addNoBroadcast, subNoBroadcast, mulNoBroadcast, divNoBroadcast ::
    (HasCallStack, PrimTensorOp t, DType u) => t u -> t u -> TensorMonad t (t u)
addNoBroadcast a b = prim S._elemwise_add ANON{lhs = Just a, rhs = Just b}
subNoBroadcast a b = prim S._elemwise_sub ANON{lhs = Just a, rhs = Just b}
mulNoBroadcast a b = prim S._elemwise_mul ANON{lhs = Just a, rhs = Just b}
divNoBroadcast a b = prim S._elemwise_div ANON{lhs = Just a, rhs = Just b}

add_, sub_, mul_, div_, and_, or_, xor_ ::
    (HasCallStack, PrimTensorOp t, DType u) => t u -> t u -> TensorMonad t (t u)
add_ a b = prim S.__npi_add          ANON{lhs = Just a, rhs = Just b}
sub_ a b = prim S.__npi_subtract     ANON{lhs = Just a, rhs = Just b}
mul_ a b = prim S.__npi_multiply     ANON{lhs = Just a, rhs = Just b}
div_ a b = prim S.__npi_true_divide  ANON{lhs = Just a, rhs = Just b}
and_  a b = prim S.__npi_bitwise_and ANON{lhs = Just a, rhs = Just b}
or_   a b = prim S.__npi_bitwise_or  ANON{lhs = Just a, rhs = Just b}
xor_  a b = prim S.__npi_bitwise_xor ANON{lhs = Just a, rhs = Just b}

eq_, neq_, lt_, leq_, gt_, geq_ ::
    (HasCallStack, PrimTensorOp t, DType u)
    => t u -> t u -> TensorMonad t (t Bool)
eq_   a b = prim S.__npi_equal         ANON{lhs = Just a, rhs = Just b}
neq_  a b = prim S.__npi_not_equal     ANON{lhs = Just a, rhs = Just b}
lt_   a b = prim S.__npi_less          ANON{lhs = Just a, rhs = Just b}
leq_  a b = prim S.__npi_less_equal    ANON{lhs = Just a, rhs = Just b}
gt_   a b = prim S.__npi_greater       ANON{lhs = Just a, rhs = Just b}
geq_  a b = prim S.__npi_greater_equal ANON{lhs = Just a, rhs = Just b}

bitwise_not, logical_not, invert ::
    (HasCallStack, PrimTensorOp t, DType u) => t u -> TensorMonad t (t u)
bitwise_not a = prim S.__npi_bitwise_not  ANON{x = Just a}
invert        = bitwise_not
logical_not a = prim S.__npi_logical_not  ANON{x = Just a}

#if MXNET_VERSION == 10600
_adaptDouble = double2Float
#elif MXNET_VERSION >= 10700
_adaptDouble = id
#endif

addScalar, subScalar, rsubScalar, mulScalar, divScalar, rdivScalar ::
    (HasCallStack, PrimTensorOp t, DType u) => Double -> t u -> TensorMonad t (t u)
addScalar  b a = prim S.__npi_add_scalar       ANON{_data = a, scalar = Just $ _adaptDouble b}
subScalar  b a = prim S.__npi_subtract_scalar  ANON{_data = a, scalar = Just $ _adaptDouble b}
rsubScalar b a = prim S.__npi_rsubtract_scalar ANON{_data = a, scalar = Just $ _adaptDouble b}
mulScalar  b a = prim S.__npi_multiply_scalar  ANON{_data = a, scalar = Just $ _adaptDouble b}
divScalar  b a = prim S.__npi_true_divide_scalar  ANON{_data = a, scalar = Just $ _adaptDouble b}
rdivScalar b a = prim S.__npi_rtrue_divide_scalar ANON{_data = a, scalar = Just $ _adaptDouble b}

eqScalar, neqScalar, ltScalar, leqScalar, gtScalar, geqScalar ::
    (HasCallStack, PrimTensorOp t, DType u)
    => Double -> t u -> TensorMonad t (t Bool)
eqScalar  b a = prim S.__npi_equal_scalar      ANON{_data = a, scalar = Just $ _adaptDouble b}
neqScalar b a = prim S.__npi_not_equal_scalar  ANON{_data = a, scalar = Just $ _adaptDouble b}
ltScalar  b a = prim S.__npi_less_scalar       ANON{_data = a, scalar = Just $ _adaptDouble b}
leqScalar b a = prim S.__npi_less_equal_scalar ANON{_data = a, scalar = Just $ _adaptDouble b}
gtScalar  b a = prim S.__npi_greater_scalar    ANON{_data = a, scalar = Just $ _adaptDouble b}
geqScalar b a = prim S.__npi_greater_equal_scalar ANON{_data = a, scalar = Just $ _adaptDouble b}


andScalar, orScalar, xorScalar ::
    (HasCallStack, PrimTensorOp t, DType u) => Double -> t u -> TensorMonad t (t u)
andScalar b a = prim S.__npi_bitwise_and_scalar ANON{_data = a, scalar = Just $ _adaptDouble b}
orScalar  b a = prim S.__npi_bitwise_or_scalar  ANON{_data = a, scalar = Just $ _adaptDouble b}
xorScalar b a = prim S.__npi_bitwise_xor_scalar ANON{_data = a, scalar = Just $ _adaptDouble b}

ceil_, floor_, sqrt_, log2_, square_ ::
    (HasCallStack, PrimTensorOp t, DType u) => t u -> TensorMonad t (t u)
ceil_   a = prim S.__npi_ceil   ANON{x = Just a}
floor_  a = prim S.__npi_floor  ANON{x = Just a}
sqrt_   a = prim S.__npi_sqrt   ANON{x = Just a}
log2_   a = prim S.__npi_log2   ANON{x = Just a}
square_ a = prim S.__npi_square ANON{x = Just a}

mean, sum_, max_ :: (HasCallStack, PrimTensorOp t, DType u)
                 => t u -> Maybe [Int] -> Bool -> TensorMonad t (t u)
sum_ s axis keepdims = prim S.__np_sum   ANON{a = Just s, axis = Just axis, keepdims = Just keepdims}
max_ s axis keepdims = prim S.__np_max   ANON{a = Just s, axis = Just axis, keepdims = Just keepdims}
mean a axis keepdims = prim S.__npi_mean ANON{a = Just a, axis = Just axis, keepdims = Just keepdims}

argmax, argmin :: (HasCallStack, PrimTensorOp t, DType u)
               => t u -> Maybe Int -> Bool -> TensorMonad t (t Int64)
argmax a axis keepdims = prim S.__npi_argmax ANON{_data = a, axis = Just axis, keepdims = Just keepdims}
argmin a axis keepdims = prim S.__npi_argmin ANON{_data = a, axis = Just axis, keepdims = Just keepdims}

einsum :: (HasCallStack, PrimTensorOp t, DType u)
       => Text -> [t u] -> Bool -> TensorMonad t (t u)
einsum spec ts opt = prim S.__npi_einsum ANON{_data = ts, num_args = length(ts), subscripts = Just spec, optimize = Just $ fromEnum opt}

abs_ :: (HasCallStack, PrimTensorOp t, DType u) => t u -> TensorMonad t (t u)
abs_ a = prim S.__npi_absolute ANON{x = Just a}

logSoftmax :: (HasCallStack, PrimTensorOp t, DType u) => t u -> Int -> Maybe Double -> TensorMonad t (t u)
logSoftmax a axis temp = prim S._log_softmax ANON{_data = a, axis = Just axis, temperature = Just temp}

identity :: (HasCallStack, PrimTensorOp t, DType u) => t u -> TensorMonad t (t u)
identity a = prim S.__np_copy ANON{a = Just a}

stack :: (HasCallStack, PrimTensorOp t, DType u) => Int -> [t u] -> TensorMonad t (t u)
stack axis ts = prim S._stack ANON{num_args = length ts,  _data = ts, axis = Just axis}

reshape :: (HasCallStack, PrimTensorOp t, DType u) => [Int] -> t u -> TensorMonad t (t u)
reshape shape a = prim S.__npx_reshape ANON{a = Just a, newshape = shape}

-- reshapeLegacy has a few magic numbers:
-- *  0 copy this dimension from the input to the output shape.
-- * -1 infers the dimension of the output shape by using the remainder of the input dimensions keeping the size of the new array same as that of the input array. At most one dimension of shape can be -1.
-- * -2 copy all/remainder of the input dimensions to the output shape.
-- * -3 use the product of two consecutive dimensions of the input shape as the output dimension.
-- * -4 split one dimension of the input into two dimensions passed subsequent to -4 in shape (can contain -1).
reshapeLegacy :: (HasCallStack, PrimTensorOp t, DType u) => [Int] -> t u -> TensorMonad t (t u)
reshapeLegacy shape a = prim S._Reshape ANON{_data = a, shape = Just shape}

reshapeLike :: (HasCallStack, PrimTensorOp t, DType u)
            => t u -> Maybe Int -> Maybe Int
            -> t u -> Maybe Int -> Maybe Int
            -> TensorMonad t (t u)
reshapeLike lhs lhs_beg lhs_end rhs rhs_beg rhs_end =
    prim S._reshape_like ANON{lhs = Just lhs, lhs_begin = Just lhs_beg, lhs_end = Just lhs_end
                             ,rhs = Just rhs, rhs_begin = Just rhs_beg, rhs_end = Just rhs_end}

concat_ :: (HasCallStack, PrimTensorOp t, DType u) => Int -> [t u] -> TensorMonad t (t u)
concat_ a s = prim S.__npi_concatenate ANON{_data = s, num_args = length s, axis = Just a}

takeI :: (HasCallStack, PrimTensorOp t, DType u)
      => t u -> t u -> TensorMonad t (t u)
takeI i a = prim S._take ANON{a = Just a, indices = Just i}

pick :: (HasCallStack, PrimTensorOp t, DType u)
      => Maybe Int -> t u -> t u -> TensorMonad t (t u)
pick a i t = prim S._pick ANON{_data = t, index = Just i, axis = Just a}

gather :: (HasCallStack, PrimTensorOp t, DType u)
       => t u -> t u -> TensorMonad t (t u)
gather a i = prim S._gather_nd ANON{_data = a, indices = Just i}

where_ :: (HasCallStack, PrimTensorOp t, DType u)
       => t Bool -> t u -> t u -> TensorMonad t (t u)
where_ c a b = prim S.__npi_where ANON{condition = Just c, x = Just a, y = Just b}

squeeze :: (HasCallStack, PrimTensorOp t, DType u) => Maybe [Int] -> t u -> TensorMonad t (t u)
squeeze axis a = prim S.__np_squeeze ANON{a = Just a, axis = Just axis}

expandDims :: (HasCallStack, PrimTensorOp t, DType u) => Int -> t u -> TensorMonad t (t u)
expandDims axis a = prim S._expand_dims ANON{_data = a, axis = axis}

broadcastAxis :: (HasCallStack, PrimTensorOp t, DType u) => [Int] -> [Int] -> t u -> TensorMonad t (t u)
broadcastAxis axis size a = prim S._broadcast_axis ANON{_data = a, axis = Just axis, size = Just size}
broadcastLike :: (HasCallStack, PrimTensorOp t, DType u) => t u -> t u -> TensorMonad t (t u)
broadcastLike lhs rhs = prim S._broadcast_like ANON{lhs = Just lhs, rhs = Just rhs}
broadcastLikeAxis :: (HasCallStack, PrimTensorOp t, DType u)
                  => (t u, [Int]) -> (t u, [Int]) -> TensorMonad t (t u)
broadcastLikeAxis (lhs, la) (rhs, ra) =
    prim S._broadcast_like
        ANON{lhs = Just lhs, rhs = Just rhs, lhs_axes = Just (Just la), rhs_axes = Just (Just ra)}

transpose :: (HasCallStack, PrimTensorOp t, DType u) => t u -> [Int] -> TensorMonad t (t u)
transpose a axes = prim S.__np_transpose ANON{a = Just a, axes = Just axes}

scatter :: forall t u. (HasCallStack, PrimTensorOp t, DType u, Typeable u, InEnum (DTypeName u) AllDTypes)
        => t Int64 -> t u -> [Int] -> TensorMonad t (t u)
scatter indices values shape = do
    case eqT @Int64 @u of
      Just Refl ->
          prim S._scatter_nd ANON{shape = shape, _data = values, indices = Just indices}
      Nothing -> do
          indices <- cast (Proxy :: Proxy (DTypeName u)) indices
          prim S._scatter_nd ANON{shape = shape, _data = values, indices = Just indices}

-- TODO: slice will always create a copy of the target sub-region. This can be
--  not ideal if the range is continuous. Use `mxNDArraySlice` instead.
slice :: (HasCallStack, PrimTensorOp t, DType u)
      => t u -> [Int] -> [Int] -> TensorMonad t (t u)
slice a beg end = prim S._slice ANON{_data = a, begin = beg, end = end}

sliceAxis :: (HasCallStack, PrimTensorOp t, DType u)
          => t u -> Int -> Int -> Maybe Int -> TensorMonad t (t u)
sliceAxis a axis beg end = prim S._slice_axis ANON{_data = a, axis = axis, begin = beg, end = end}

splitBySections :: (HasCallStack, PrimTensorOp t, DType u)
                => Int -> Int -> Bool -> t u -> TensorMonad t [t u]
splitBySections num_sections axis squeeze s =
    primMulti S.__split_v2 ANON{_data = s, axis = Just axis, indices = [], sections = Just num_sections, squeeze_axis = Just squeeze}


hsplitBySections :: (HasCallStack, PrimTensorOp t, DType u)
                 => Int -> t u -> TensorMonad t [t u]
hsplitBySections num_sections s =
    primMulti S.__npi_hsplit ANON{_data = s, axis = Just 1, indices = [], sections = Just num_sections, squeeze_axis = Just False}

vsplitBySections, dsplitBySections :: (HasCallStack, PrimTensorOp t, DType u)
                                   => Int -> t u -> TensorMonad t [t u]
vsplitBySections num_sections s = splitBySections num_sections 0 False s
dsplitBySections num_sections s = splitBySections num_sections 2 False s

repeat_ :: (HasCallStack, PrimTensorOp t, DType u) => Int -> Maybe Int -> t u -> TensorMonad t (t u)
repeat_ n a t = prim S._repeat ANON{_data = t, repeats = n, axis = Just a}

cast :: (HasCallStack, PrimTensorOp t, DType u, DType v, KnownSymbol dty, InEnum dty AllDTypes, DTypeName v ~ dty)
     => Proxy dty
     -> t u
     -> TensorMonad t (t v)
cast dty t = prim S._Cast ANON{dtype = EnumType dty, _data = t}

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
    c <- prim S._Activation ANON{_data = c, act_type = #softrelu}
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
    [ret] <- S.__copyto ANON{_data = src} (Just [dst])
    return ret

#endif
