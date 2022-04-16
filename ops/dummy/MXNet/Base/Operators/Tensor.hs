module MXNet.Base.Operators.Tensor where

import           RIO

import           MXNet.Base.Core.Spec
import           MXNet.Base.Tensor.Class
import           MXNet.Base.Types        (DType)

type ParameterList_copyto t u = '[ '("data", AttrOpt (t u))]

__copyto :: forall t u . (Tensor t, DType u, HasCallStack)
         => ParamListFull (ParameterList_copyto t u) -> TensorApply (t u)
__copyto = error "tensors operators not generated."
