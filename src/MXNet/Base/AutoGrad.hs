module MXNet.Base.AutoGrad where

import           RIO

import           MXNet.Base.NDArray
import qualified MXNet.Base.Raw               as I
import           MXNet.Base.Tensor.Functional (zerosLike)
import           MXNet.Base.Types             (DType, ReqType)

attachGradient :: DType a => NDArray a -> ReqType -> IO ()
attachGradient arr req_type = do
    grad <- zerosLike arr
    void $ I.mxAutogradMarkVariables [(unNDArray arr, fromEnum req_type, unNDArray grad)]

backward :: DType a => [NDArray a] -> [NDArray a] -> Bool -> Bool -> Bool -> IO ()
backward heads variables retrain_graph create_graph is_train =
    void $ I.mxAutogradBackwardEx
        (Left $ map unNDArray heads)
        (map unNDArray variables)
        retrain_graph create_graph is_train False

recording :: Bool -> IO a -> IO a
recording isRecording io = bracket (I.mxAutogradSetIsRecording flag) restore (const io)
    where
    flag = if isRecording then 1 else 0
    restore prev = when (prev /= flag) (void $ I.mxAutogradSetIsRecording prev)

training :: Bool -> IO a -> IO a
training isTraining io = bracket (I.mxAutogradSetIsTraining flag) restore (const io)
    where
    flag = if isTraining then 1 else 0
    restore prev = when (prev /= flag) (void $ I.mxAutogradSetIsTraining prev)
