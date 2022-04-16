{-# LANGUAGE DataKinds #-}
module MXNet.Base.Profiler (
    module MXNet.Base.Raw.Profiler,
    setConfig,
    withProfiler,
    StatsFormat(..),
    StatsOrder(..),
    StatsSortBy(..),
    stats,
) where

import           RIO

import           MXNet.Base.Core.Spec
import           MXNet.Base.Raw.Common   (flagSafeToFree)
import           MXNet.Base.Raw.Profiler


type ParamListProfilerConfig = [
  '("filename", AttrReq String),
  '("gpu_memory_profile_filename_prefix", AttrOpt String),
  '("profile_all", AttrReq Bool),
  '("profile_symbolic", AttrOpt Bool),
  '("profile_imperative", AttrOpt Bool),
  '("profile_memory", AttrOpt Bool),
  '("profile_api", AttrOpt Bool),
  '("continuous_dump", AttrOpt Bool),
  '("dump_period", AttrOpt Float),
  '("aggregate_stats", AttrOpt Bool)
  ]

setConfig :: (HasCallStack, Dump (FieldsFull ParamListProfilerConfig))
          => ParamListFull ParamListProfilerConfig -> IO ()
setConfig args = do
    let kwargs = dump args
    mxSetProfilerConfig kwargs

withProfiler :: IO a -> IO a
withProfiler = bracket_ (withMVar flagSafeToFree $ \_ -> mxSetProfilerState 1) (mxSetProfilerState 0)

beginProfiler :: IO ()
beginProfiler = mxSetProfilerState 1

endProfiler :: IO ()
endProfiler = mxSetProfilerState 0

data StatsFormat = Table | Json
  deriving Enum
data StatsSortBy = Total | Average | Min | Max | Count
  deriving Enum
data StatsOrder = Ascending | Descending
  deriving Enum

stats :: HasCallStack => Bool -> StatsFormat -> StatsSortBy -> StatsOrder -> IO Text
stats reset format sort_by order =
    mxAggregateProfileStatsPrintEx
        (fromEnum reset)
        (fromEnum format)
        (fromEnum sort_by)
        (fromEnum order)

