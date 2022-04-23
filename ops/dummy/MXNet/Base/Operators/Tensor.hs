module MXNet.Base.Operators.Tensor where

import           Data.Record.Anon.Simple (Record)
import           RIO

import           MXNet.Base.Core.Spec
import           MXNet.Base.Tensor.Class
import           MXNet.Base.Types        (DType)

type ParameterList__copyto t u = '[ '("_data", AttrReq (t u))]

__copyto :: forall t u . (Tensor t, DType u, HasCallStack)
         => Record (FieldsFull (ParameterList__copyto t u)) -> TensorApply (t u)
__copyto = error "tensors operators not generated."
