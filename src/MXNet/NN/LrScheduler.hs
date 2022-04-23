{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -fplugin=Data.Record.Anon.Plugin#-}
module MXNet.NN.LrScheduler where

import           Data.Record.Anon.Simple (Record)
import qualified Data.Record.Anon.Simple as Anon
import           RIO                     hiding (Const)

import           MXNet.Base.Core.Spec

class Show sch => LrScheduler sch where
    baseLR :: sch -> Float
    getLR  :: sch -> Int -> Float

instance LrScheduler Float where
    baseLR = id
    getLR = const

newtype Const = Const Float
    deriving Show
instance LrScheduler Const where
    baseLR (Const lr) = lr
    getLR  (Const lr) = const lr

lrOfConst :: Float -> Const
lrOfConst = Const

data FactorScheduler = Factor Float Float Int Float
    deriving Show
instance LrScheduler FactorScheduler where
    baseLR (Factor base _ _ _) = base
    getLR  (Factor base factor step stop) nup =
        let lr = base * factor ^ (nup `div` step)
        in if lr < stop then stop else lr

type ParameterListLrOfFactor =
    '[ '("factor", 'AttrReq Float), '("step", 'AttrReq Int),
       '("base", 'AttrOpt Float), '("stop", 'AttrOpt Float)]

lrOfFactor :: FieldsAcc ParameterListLrOfFactor rec
           => Record rec -> FactorScheduler
lrOfFactor args = Factor base factor step stop
  where
    fullargs = paramListWithDefault (Proxy @ParameterListLrOfFactor) args
    factor = Anon.get #factor fullargs
    step   = Anon.get #step   fullargs
    base   = fromMaybe 0.01 (Anon.get #base fullargs)
    stop   = fromMaybe 1e-8 (Anon.get #stop fullargs)

data MultifactorScheduler = Multifactor Float Float [Int]
    deriving Show
instance LrScheduler MultifactorScheduler where
    baseLR (Multifactor base _ _) = base
    getLR  (Multifactor base factor steps) nup = base * factor ^ (index nup steps)
      where
        index a bs = go a bs (0 :: Int)
        go _ [] n     = n
        go a (b:bs) n = if b > a then n else go a bs (n+1)

type ParameterListLrOfMultifactor =
    '[ '("factor", 'AttrReq Float), '("steps", 'AttrReq [Int]), '("base", 'AttrOpt Float)]

lrOfMultifactor :: FieldsAcc ParameterListLrOfMultifactor r
                => Record r -> MultifactorScheduler
lrOfMultifactor args = Multifactor base factor steps
  where
    fullargs = paramListWithDefault (Proxy @ParameterListLrOfMultifactor) args
    factor = Anon.get #factor fullargs
    steps  = Anon.get #steps  fullargs
    base = fromMaybe 0.01 (Anon.get #base fullargs)

data PolyScheduler = Poly Float Float Int
    deriving Show
instance LrScheduler PolyScheduler where
    baseLR (Poly base _ _) = base
    getLR  (Poly base power maxnup) nup =
        if nup < maxnup
          then base * (1 - fromIntegral nup / fromIntegral maxnup) ** power
          else 0

type ParameterListLrOfPoly =
    '[ '("maxnup", 'AttrReq Int), '("power", 'AttrOpt Float), '("base", 'AttrOpt Float)]

lrOfPoly :: FieldsAcc ParameterListLrOfPoly r
           => Record r -> PolyScheduler
lrOfPoly args = Poly base power maxnup
  where
    fullargs = paramListWithDefault (Proxy @ParameterListLrOfPoly) args
    maxnup = Anon.get #maxnup fullargs
    base   = fromMaybe 0.01 (Anon.get #base  fullargs)
    power  = fromMaybe 2    (Anon.get #power fullargs)

data WarmupScheduler a = WarmupScheduler Int a
    deriving Show
instance LrScheduler a => LrScheduler (WarmupScheduler a) where
    baseLR (WarmupScheduler _ sch) = baseLR sch
    getLR  (WarmupScheduler warmup_steps sch) nup =
        let base = baseLR sch
         in if nup >= warmup_steps
            then getLR sch (nup - warmup_steps)
            else base / fromIntegral warmup_steps * fromIntegral nup

