{-# LANGUAGE TemplateHaskell #-}
module MXNet.NN.Types where

import           Control.Lens (makeLenses)
import           RIO

data Statistics = Statistics
    { _stat_num_upd :: !Int
    , _stat_last_lr :: !Float
    }

makeLenses ''Statistics
