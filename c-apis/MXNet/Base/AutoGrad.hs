module MXNet.Base.AutoGrad where

import           Control.Monad.Trans.Resource
import           RIO

import           MXNet.Base.NDArray
import qualified MXNet.Base.Raw               as I

backward :: [NDArray a] -> [NDArray a] -> Bool -> Bool -> Bool -> IO ()
backward heads variables retrain_graph create_graph is_train =
    void $ I.mxAutogradBackwardEx
        (Left $ map unNDArray heads)
        (map unNDArray variables)
        retrain_graph create_graph is_train False

recording isRecording io = runResourceT $ do
    allocate (I.mxAutogradSetIsRecording isRecording)
             (\prev -> when
                (prev /= isRecording)
                (void $ I.mxAutogradSetIsRecording prev))

training isTraining io = runResourceT $ do
    allocate (I.mxAutogradSetIsTraining isTraining)
             (\prev -> when
                (prev /= isTraining)
                (void $ I.mxAutogradSetIsTraining prev))
