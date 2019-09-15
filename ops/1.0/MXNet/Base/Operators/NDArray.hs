module MXNet.Base.Operators.NDArray where
import MXNet.Base.Raw
import MXNet.Base.Spec.Operator
import MXNet.Base.Spec.HMap
import Data.Maybe (catMaybes, fromMaybe)

type instance ParameterList "broadcast_power(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

broadcast_power ::
                forall args . Fullfilled "broadcast_power(ndarray)" args =>
                  ArgsHMap "broadcast_power(ndarray)" args -> IO [NDArrayHandle]
broadcast_power args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_power"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

broadcast_power_upd ::
                    forall args . Fullfilled "broadcast_power(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "broadcast_power(ndarray)" args -> IO ()
broadcast_power_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_power"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_broadcast_power(ndarray)" =
     '[]

_backward_broadcast_power ::
                          forall args .
                            Fullfilled "_backward_broadcast_power(ndarray)" args =>
                            ArgsHMap "_backward_broadcast_power(ndarray)" args ->
                              IO [NDArrayHandle]
_backward_broadcast_power args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_power"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_broadcast_power_upd ::
                              forall args .
                                Fullfilled "_backward_broadcast_power(ndarray)" args =>
                                [NDArrayHandle] ->
                                  ArgsHMap "_backward_broadcast_power(ndarray)" args -> IO ()
_backward_broadcast_power_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_power"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "broadcast_maximum(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

broadcast_maximum ::
                  forall args . Fullfilled "broadcast_maximum(ndarray)" args =>
                    ArgsHMap "broadcast_maximum(ndarray)" args -> IO [NDArrayHandle]
broadcast_maximum args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_maximum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

broadcast_maximum_upd ::
                      forall args . Fullfilled "broadcast_maximum(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "broadcast_maximum(ndarray)" args -> IO ()
broadcast_maximum_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_maximum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_broadcast_maximum(ndarray)"
     = '[]

_backward_broadcast_maximum ::
                            forall args .
                              Fullfilled "_backward_broadcast_maximum(ndarray)" args =>
                              ArgsHMap "_backward_broadcast_maximum(ndarray)" args ->
                                IO [NDArrayHandle]
_backward_broadcast_maximum args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_maximum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_broadcast_maximum_upd ::
                                forall args .
                                  Fullfilled "_backward_broadcast_maximum(ndarray)" args =>
                                  [NDArrayHandle] ->
                                    ArgsHMap "_backward_broadcast_maximum(ndarray)" args -> IO ()
_backward_broadcast_maximum_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_maximum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "broadcast_minimum(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

broadcast_minimum ::
                  forall args . Fullfilled "broadcast_minimum(ndarray)" args =>
                    ArgsHMap "broadcast_minimum(ndarray)" args -> IO [NDArrayHandle]
broadcast_minimum args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_minimum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

broadcast_minimum_upd ::
                      forall args . Fullfilled "broadcast_minimum(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "broadcast_minimum(ndarray)" args -> IO ()
broadcast_minimum_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_minimum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_broadcast_minimum(ndarray)"
     = '[]

_backward_broadcast_minimum ::
                            forall args .
                              Fullfilled "_backward_broadcast_minimum(ndarray)" args =>
                              ArgsHMap "_backward_broadcast_minimum(ndarray)" args ->
                                IO [NDArrayHandle]
_backward_broadcast_minimum args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_minimum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_broadcast_minimum_upd ::
                                forall args .
                                  Fullfilled "_backward_broadcast_minimum(ndarray)" args =>
                                  [NDArrayHandle] ->
                                    ArgsHMap "_backward_broadcast_minimum(ndarray)" args -> IO ()
_backward_broadcast_minimum_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_minimum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "broadcast_hypot(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

broadcast_hypot ::
                forall args . Fullfilled "broadcast_hypot(ndarray)" args =>
                  ArgsHMap "broadcast_hypot(ndarray)" args -> IO [NDArrayHandle]
broadcast_hypot args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_hypot"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

broadcast_hypot_upd ::
                    forall args . Fullfilled "broadcast_hypot(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "broadcast_hypot(ndarray)" args -> IO ()
broadcast_hypot_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_hypot"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_broadcast_hypot(ndarray)" =
     '[]

_backward_broadcast_hypot ::
                          forall args .
                            Fullfilled "_backward_broadcast_hypot(ndarray)" args =>
                            ArgsHMap "_backward_broadcast_hypot(ndarray)" args ->
                              IO [NDArrayHandle]
_backward_broadcast_hypot args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_hypot"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_broadcast_hypot_upd ::
                              forall args .
                                Fullfilled "_backward_broadcast_hypot(ndarray)" args =>
                                [NDArrayHandle] ->
                                  ArgsHMap "_backward_broadcast_hypot(ndarray)" args -> IO ()
_backward_broadcast_hypot_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_hypot"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_equal(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_equal ::
       forall args . Fullfilled "_equal(ndarray)" args =>
         ArgsHMap "_equal(ndarray)" args -> IO [NDArrayHandle]
_equal args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_equal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_equal_upd ::
           forall args . Fullfilled "_equal(ndarray)" args =>
             [NDArrayHandle] -> ArgsHMap "_equal(ndarray)" args -> IO ()
_equal_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_equal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_not_equal(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_not_equal ::
           forall args . Fullfilled "_not_equal(ndarray)" args =>
             ArgsHMap "_not_equal(ndarray)" args -> IO [NDArrayHandle]
_not_equal args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_not_equal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_not_equal_upd ::
               forall args . Fullfilled "_not_equal(ndarray)" args =>
                 [NDArrayHandle] -> ArgsHMap "_not_equal(ndarray)" args -> IO ()
_not_equal_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_not_equal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_greater(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_greater ::
         forall args . Fullfilled "_greater(ndarray)" args =>
           ArgsHMap "_greater(ndarray)" args -> IO [NDArrayHandle]
_greater args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_greater"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_greater_upd ::
             forall args . Fullfilled "_greater(ndarray)" args =>
               [NDArrayHandle] -> ArgsHMap "_greater(ndarray)" args -> IO ()
_greater_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_greater"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_greater_equal(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_greater_equal ::
               forall args . Fullfilled "_greater_equal(ndarray)" args =>
                 ArgsHMap "_greater_equal(ndarray)" args -> IO [NDArrayHandle]
_greater_equal args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_greater_equal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_greater_equal_upd ::
                   forall args . Fullfilled "_greater_equal(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_greater_equal(ndarray)" args -> IO ()
_greater_equal_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_greater_equal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_lesser(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_lesser ::
        forall args . Fullfilled "_lesser(ndarray)" args =>
          ArgsHMap "_lesser(ndarray)" args -> IO [NDArrayHandle]
_lesser args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_lesser"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_lesser_upd ::
            forall args . Fullfilled "_lesser(ndarray)" args =>
              [NDArrayHandle] -> ArgsHMap "_lesser(ndarray)" args -> IO ()
_lesser_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_lesser"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_lesser_equal(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_lesser_equal ::
              forall args . Fullfilled "_lesser_equal(ndarray)" args =>
                ArgsHMap "_lesser_equal(ndarray)" args -> IO [NDArrayHandle]
_lesser_equal args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_lesser_equal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_lesser_equal_upd ::
                  forall args . Fullfilled "_lesser_equal(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_lesser_equal(ndarray)" args -> IO ()
_lesser_equal_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_lesser_equal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_power(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_power ::
       forall args . Fullfilled "_power(ndarray)" args =>
         ArgsHMap "_power(ndarray)" args -> IO [NDArrayHandle]
_power args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_power"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_power_upd ::
           forall args . Fullfilled "_power(ndarray)" args =>
             [NDArrayHandle] -> ArgsHMap "_power(ndarray)" args -> IO ()
_power_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_power"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_power(ndarray)" = '[]

_backward_power ::
                forall args . Fullfilled "_backward_power(ndarray)" args =>
                  ArgsHMap "_backward_power(ndarray)" args -> IO [NDArrayHandle]
_backward_power args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_power"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_power_upd ::
                    forall args . Fullfilled "_backward_power(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_backward_power(ndarray)" args -> IO ()
_backward_power_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_power"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_maximum(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_maximum ::
         forall args . Fullfilled "_maximum(ndarray)" args =>
           ArgsHMap "_maximum(ndarray)" args -> IO [NDArrayHandle]
_maximum args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_maximum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_maximum_upd ::
             forall args . Fullfilled "_maximum(ndarray)" args =>
               [NDArrayHandle] -> ArgsHMap "_maximum(ndarray)" args -> IO ()
_maximum_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_maximum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_maximum(ndarray)" = '[]

_backward_maximum ::
                  forall args . Fullfilled "_backward_maximum(ndarray)" args =>
                    ArgsHMap "_backward_maximum(ndarray)" args -> IO [NDArrayHandle]
_backward_maximum args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_maximum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_maximum_upd ::
                      forall args . Fullfilled "_backward_maximum(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "_backward_maximum(ndarray)" args -> IO ()
_backward_maximum_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_maximum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_minimum(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_minimum ::
         forall args . Fullfilled "_minimum(ndarray)" args =>
           ArgsHMap "_minimum(ndarray)" args -> IO [NDArrayHandle]
_minimum args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_minimum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_minimum_upd ::
             forall args . Fullfilled "_minimum(ndarray)" args =>
               [NDArrayHandle] -> ArgsHMap "_minimum(ndarray)" args -> IO ()
_minimum_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_minimum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_minimum(ndarray)" = '[]

_backward_minimum ::
                  forall args . Fullfilled "_backward_minimum(ndarray)" args =>
                    ArgsHMap "_backward_minimum(ndarray)" args -> IO [NDArrayHandle]
_backward_minimum args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_minimum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_minimum_upd ::
                      forall args . Fullfilled "_backward_minimum(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "_backward_minimum(ndarray)" args -> IO ()
_backward_minimum_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_minimum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_hypot(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_hypot ::
       forall args . Fullfilled "_hypot(ndarray)" args =>
         ArgsHMap "_hypot(ndarray)" args -> IO [NDArrayHandle]
_hypot args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_hypot"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_hypot_upd ::
           forall args . Fullfilled "_hypot(ndarray)" args =>
             [NDArrayHandle] -> ArgsHMap "_hypot(ndarray)" args -> IO ()
_hypot_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_hypot"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_hypot(ndarray)" = '[]

_backward_hypot ::
                forall args . Fullfilled "_backward_hypot(ndarray)" args =>
                  ArgsHMap "_backward_hypot(ndarray)" args -> IO [NDArrayHandle]
_backward_hypot args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_hypot"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_hypot_upd ::
                    forall args . Fullfilled "_backward_hypot(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_backward_hypot(ndarray)" args -> IO ()
_backward_hypot_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_hypot"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_square_sum(ndarray)" =
     '[ '("axis", AttrOpt [Int]), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt NDArrayHandle)]

_square_sum ::
            forall args . Fullfilled "_square_sum(ndarray)" args =>
              ArgsHMap "_square_sum(ndarray)" args -> IO [NDArrayHandle]
_square_sum args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int]),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_square_sum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_square_sum_upd ::
                forall args . Fullfilled "_square_sum(ndarray)" args =>
                  [NDArrayHandle] -> ArgsHMap "_square_sum(ndarray)" args -> IO ()
_square_sum_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int]),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_square_sum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_square_sum(ndarray)" = '[]

_backward_square_sum ::
                     forall args . Fullfilled "_backward_square_sum(ndarray)" args =>
                       ArgsHMap "_backward_square_sum(ndarray)" args -> IO [NDArrayHandle]
_backward_square_sum args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_square_sum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_square_sum_upd ::
                         forall args . Fullfilled "_backward_square_sum(ndarray)" args =>
                           [NDArrayHandle] ->
                             ArgsHMap "_backward_square_sum(ndarray)" args -> IO ()
_backward_square_sum_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_square_sum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "add_n(ndarray)" =
     '[ '("args", AttrOpt [NDArrayHandle])]

add_n ::
      forall args . Fullfilled "add_n(ndarray)" args =>
        ArgsHMap "add_n(ndarray)" args -> IO [NDArrayHandle]
add_n args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
        array = fromMaybe [] (args !? #args :: Maybe [NDArrayHandle])
      in
      do op <- nnGetOpHandle "add_n"
         let scalarArgs'
               = if hasKey args #num_args then scalarArgs else
                   (,) "num_args" (showValue (length array)) : scalarArgs
         listndarr <- mxImperativeInvoke (fromOpHandle op) array scalarArgs'
                        Nothing
         return listndarr

add_n_upd ::
          forall args . Fullfilled "add_n(ndarray)" args =>
            [NDArrayHandle] -> ArgsHMap "add_n(ndarray)" args -> IO ()
add_n_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
        array = fromMaybe [] (args !? #args :: Maybe [NDArrayHandle])
      in
      do op <- nnGetOpHandle "add_n"
         let scalarArgs'
               = if hasKey args #num_args then scalarArgs else
                   (,) "num_args" (showValue (length array)) : scalarArgs
         listndarr <- mxImperativeInvoke (fromOpHandle op) array scalarArgs'
                        (Just outputs)
         return ()

type instance ParameterList "_zeros(ndarray)" =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt
            (EnumType '["float16", "float32", "float64", "int32", "uint8"]))]

_zeros ::
       forall args . Fullfilled "_zeros(ndarray)" args =>
         ArgsHMap "_zeros(ndarray)" args -> IO [NDArrayHandle]
_zeros args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType '["float16", "float32", "float64", "int32", "uint8"]))]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_zeros"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_zeros_upd ::
           forall args . Fullfilled "_zeros(ndarray)" args =>
             [NDArrayHandle] -> ArgsHMap "_zeros(ndarray)" args -> IO ()
_zeros_upd outputs args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType '["float16", "float32", "float64", "int32", "uint8"]))]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_zeros"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_ones(ndarray)" =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt
            (EnumType '["float16", "float32", "float64", "int32", "uint8"]))]

_ones ::
      forall args . Fullfilled "_ones(ndarray)" args =>
        ArgsHMap "_ones(ndarray)" args -> IO [NDArrayHandle]
_ones args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType '["float16", "float32", "float64", "int32", "uint8"]))]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_ones"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_ones_upd ::
          forall args . Fullfilled "_ones(ndarray)" args =>
            [NDArrayHandle] -> ArgsHMap "_ones(ndarray)" args -> IO ()
_ones_upd outputs args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType '["float16", "float32", "float64", "int32", "uint8"]))]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_ones"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_full(ndarray)" =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt
            (EnumType '["float16", "float32", "float64", "int32", "uint8"])),
        '("value", AttrReq Double)]

_full ::
      forall args . Fullfilled "_full(ndarray)" args =>
        ArgsHMap "_full(ndarray)" args -> IO [NDArrayHandle]
_full args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType '["float16", "float32", "float64", "int32", "uint8"])),
               ("value",) . showValue <$> (args !? #value :: Maybe Double)]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_full"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_full_upd ::
          forall args . Fullfilled "_full(ndarray)" args =>
            [NDArrayHandle] -> ArgsHMap "_full(ndarray)" args -> IO ()
_full_upd outputs args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType '["float16", "float32", "float64", "int32", "uint8"])),
               ("value",) . showValue <$> (args !? #value :: Maybe Double)]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_full"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_arange(ndarray)" =
     '[ '("start", AttrReq Double), '("stop", AttrOpt (Maybe Double)),
        '("step", AttrOpt Double), '("repeat", AttrOpt Int),
        '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "uint8"]))]

_arange ::
        forall args . Fullfilled "_arange(ndarray)" args =>
          ArgsHMap "_arange(ndarray)" args -> IO [NDArrayHandle]
_arange args
  = let scalarArgs
          = catMaybes
              [("start",) . showValue <$> (args !? #start :: Maybe Double),
               ("stop",) . showValue <$> (args !? #stop :: Maybe (Maybe Double)),
               ("step",) . showValue <$> (args !? #step :: Maybe Double),
               ("repeat",) . showValue <$> (args !? #repeat :: Maybe Int),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["float16", "float32", "float64", "int32", "int64", "uint8"]))]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_arange"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_arange_upd ::
            forall args . Fullfilled "_arange(ndarray)" args =>
              [NDArrayHandle] -> ArgsHMap "_arange(ndarray)" args -> IO ()
_arange_upd outputs args
  = let scalarArgs
          = catMaybes
              [("start",) . showValue <$> (args !? #start :: Maybe Double),
               ("stop",) . showValue <$> (args !? #stop :: Maybe (Maybe Double)),
               ("step",) . showValue <$> (args !? #step :: Maybe Double),
               ("repeat",) . showValue <$> (args !? #repeat :: Maybe Int),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["float16", "float32", "float64", "int32", "int64", "uint8"]))]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_arange"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "zeros_like(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

zeros_like ::
           forall args . Fullfilled "zeros_like(ndarray)" args =>
             ArgsHMap "zeros_like(ndarray)" args -> IO [NDArrayHandle]
zeros_like args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "zeros_like"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

zeros_like_upd ::
               forall args . Fullfilled "zeros_like(ndarray)" args =>
                 [NDArrayHandle] -> ArgsHMap "zeros_like(ndarray)" args -> IO ()
zeros_like_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "zeros_like"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "ones_like(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

ones_like ::
          forall args . Fullfilled "ones_like(ndarray)" args =>
            ArgsHMap "ones_like(ndarray)" args -> IO [NDArrayHandle]
ones_like args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "ones_like"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

ones_like_upd ::
              forall args . Fullfilled "ones_like(ndarray)" args =>
                [NDArrayHandle] -> ArgsHMap "ones_like(ndarray)" args -> IO ()
ones_like_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "ones_like"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "broadcast_add(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

broadcast_add ::
              forall args . Fullfilled "broadcast_add(ndarray)" args =>
                ArgsHMap "broadcast_add(ndarray)" args -> IO [NDArrayHandle]
broadcast_add args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_add"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

broadcast_add_upd ::
                  forall args . Fullfilled "broadcast_add(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "broadcast_add(ndarray)" args -> IO ()
broadcast_add_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_add"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_broadcast_add(ndarray)" =
     '[]

_backward_broadcast_add ::
                        forall args . Fullfilled "_backward_broadcast_add(ndarray)" args =>
                          ArgsHMap "_backward_broadcast_add(ndarray)" args ->
                            IO [NDArrayHandle]
_backward_broadcast_add args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_add"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_broadcast_add_upd ::
                            forall args . Fullfilled "_backward_broadcast_add(ndarray)" args =>
                              [NDArrayHandle] ->
                                ArgsHMap "_backward_broadcast_add(ndarray)" args -> IO ()
_backward_broadcast_add_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_add"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "broadcast_sub(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

broadcast_sub ::
              forall args . Fullfilled "broadcast_sub(ndarray)" args =>
                ArgsHMap "broadcast_sub(ndarray)" args -> IO [NDArrayHandle]
broadcast_sub args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_sub"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

broadcast_sub_upd ::
                  forall args . Fullfilled "broadcast_sub(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "broadcast_sub(ndarray)" args -> IO ()
broadcast_sub_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_sub"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_broadcast_sub(ndarray)" =
     '[]

_backward_broadcast_sub ::
                        forall args . Fullfilled "_backward_broadcast_sub(ndarray)" args =>
                          ArgsHMap "_backward_broadcast_sub(ndarray)" args ->
                            IO [NDArrayHandle]
_backward_broadcast_sub args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_sub"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_broadcast_sub_upd ::
                            forall args . Fullfilled "_backward_broadcast_sub(ndarray)" args =>
                              [NDArrayHandle] ->
                                ArgsHMap "_backward_broadcast_sub(ndarray)" args -> IO ()
_backward_broadcast_sub_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_sub"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "broadcast_mul(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

broadcast_mul ::
              forall args . Fullfilled "broadcast_mul(ndarray)" args =>
                ArgsHMap "broadcast_mul(ndarray)" args -> IO [NDArrayHandle]
broadcast_mul args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_mul"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

broadcast_mul_upd ::
                  forall args . Fullfilled "broadcast_mul(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "broadcast_mul(ndarray)" args -> IO ()
broadcast_mul_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_mul"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_broadcast_mul(ndarray)" =
     '[]

_backward_broadcast_mul ::
                        forall args . Fullfilled "_backward_broadcast_mul(ndarray)" args =>
                          ArgsHMap "_backward_broadcast_mul(ndarray)" args ->
                            IO [NDArrayHandle]
_backward_broadcast_mul args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_mul"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_broadcast_mul_upd ::
                            forall args . Fullfilled "_backward_broadcast_mul(ndarray)" args =>
                              [NDArrayHandle] ->
                                ArgsHMap "_backward_broadcast_mul(ndarray)" args -> IO ()
_backward_broadcast_mul_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_mul"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "broadcast_div(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

broadcast_div ::
              forall args . Fullfilled "broadcast_div(ndarray)" args =>
                ArgsHMap "broadcast_div(ndarray)" args -> IO [NDArrayHandle]
broadcast_div args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_div"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

broadcast_div_upd ::
                  forall args . Fullfilled "broadcast_div(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "broadcast_div(ndarray)" args -> IO ()
broadcast_div_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_div"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_broadcast_div(ndarray)" =
     '[]

_backward_broadcast_div ::
                        forall args . Fullfilled "_backward_broadcast_div(ndarray)" args =>
                          ArgsHMap "_backward_broadcast_div(ndarray)" args ->
                            IO [NDArrayHandle]
_backward_broadcast_div args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_div"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_broadcast_div_upd ::
                            forall args . Fullfilled "_backward_broadcast_div(ndarray)" args =>
                              [NDArrayHandle] ->
                                ArgsHMap "_backward_broadcast_div(ndarray)" args -> IO ()
_backward_broadcast_div_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_div"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "broadcast_mod(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

broadcast_mod ::
              forall args . Fullfilled "broadcast_mod(ndarray)" args =>
                ArgsHMap "broadcast_mod(ndarray)" args -> IO [NDArrayHandle]
broadcast_mod args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_mod"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

broadcast_mod_upd ::
                  forall args . Fullfilled "broadcast_mod(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "broadcast_mod(ndarray)" args -> IO ()
broadcast_mod_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_mod"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_broadcast_mod(ndarray)" =
     '[]

_backward_broadcast_mod ::
                        forall args . Fullfilled "_backward_broadcast_mod(ndarray)" args =>
                          ArgsHMap "_backward_broadcast_mod(ndarray)" args ->
                            IO [NDArrayHandle]
_backward_broadcast_mod args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_mod"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_broadcast_mod_upd ::
                            forall args . Fullfilled "_backward_broadcast_mod(ndarray)" args =>
                              [NDArrayHandle] ->
                                ArgsHMap "_backward_broadcast_mod(ndarray)" args -> IO ()
_backward_broadcast_mod_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_mod"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_linalg_gemm(ndarray)" =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("alpha", AttrOpt Double), '("beta", AttrOpt Double),
        '("_A", AttrOpt NDArrayHandle), '("_B", AttrOpt NDArrayHandle),
        '("_C", AttrOpt NDArrayHandle)]

_linalg_gemm ::
             forall args . Fullfilled "_linalg_gemm(ndarray)" args =>
               ArgsHMap "_linalg_gemm(ndarray)" args -> IO [NDArrayHandle]
_linalg_gemm args
  = let scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$>
                 (args !? #transpose_a :: Maybe Bool),
               ("transpose_b",) . showValue <$>
                 (args !? #transpose_b :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double),
               ("beta",) . showValue <$> (args !? #beta :: Maybe Double)]
        tensorArgs
          = catMaybes
              [("_A",) <$> (args !? #_A :: Maybe NDArrayHandle),
               ("_B",) <$> (args !? #_B :: Maybe NDArrayHandle),
               ("_C",) <$> (args !? #_C :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_gemm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_linalg_gemm_upd ::
                 forall args . Fullfilled "_linalg_gemm(ndarray)" args =>
                   [NDArrayHandle] -> ArgsHMap "_linalg_gemm(ndarray)" args -> IO ()
_linalg_gemm_upd outputs args
  = let scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$>
                 (args !? #transpose_a :: Maybe Bool),
               ("transpose_b",) . showValue <$>
                 (args !? #transpose_b :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double),
               ("beta",) . showValue <$> (args !? #beta :: Maybe Double)]
        tensorArgs
          = catMaybes
              [("_A",) <$> (args !? #_A :: Maybe NDArrayHandle),
               ("_B",) <$> (args !? #_B :: Maybe NDArrayHandle),
               ("_C",) <$> (args !? #_C :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_gemm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_linalg_gemm(ndarray)" = '[]

_backward_linalg_gemm ::
                      forall args . Fullfilled "_backward_linalg_gemm(ndarray)" args =>
                        ArgsHMap "_backward_linalg_gemm(ndarray)" args ->
                          IO [NDArrayHandle]
_backward_linalg_gemm args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_gemm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_linalg_gemm_upd ::
                          forall args . Fullfilled "_backward_linalg_gemm(ndarray)" args =>
                            [NDArrayHandle] ->
                              ArgsHMap "_backward_linalg_gemm(ndarray)" args -> IO ()
_backward_linalg_gemm_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_gemm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_linalg_gemm2(ndarray)" =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("alpha", AttrOpt Double), '("_A", AttrOpt NDArrayHandle),
        '("_B", AttrOpt NDArrayHandle)]

_linalg_gemm2 ::
              forall args . Fullfilled "_linalg_gemm2(ndarray)" args =>
                ArgsHMap "_linalg_gemm2(ndarray)" args -> IO [NDArrayHandle]
_linalg_gemm2 args
  = let scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$>
                 (args !? #transpose_a :: Maybe Bool),
               ("transpose_b",) . showValue <$>
                 (args !? #transpose_b :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double)]
        tensorArgs
          = catMaybes
              [("_A",) <$> (args !? #_A :: Maybe NDArrayHandle),
               ("_B",) <$> (args !? #_B :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_gemm2"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_linalg_gemm2_upd ::
                  forall args . Fullfilled "_linalg_gemm2(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_linalg_gemm2(ndarray)" args -> IO ()
_linalg_gemm2_upd outputs args
  = let scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$>
                 (args !? #transpose_a :: Maybe Bool),
               ("transpose_b",) . showValue <$>
                 (args !? #transpose_b :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double)]
        tensorArgs
          = catMaybes
              [("_A",) <$> (args !? #_A :: Maybe NDArrayHandle),
               ("_B",) <$> (args !? #_B :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_gemm2"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_linalg_gemm2(ndarray)" = '[]

_backward_linalg_gemm2 ::
                       forall args . Fullfilled "_backward_linalg_gemm2(ndarray)" args =>
                         ArgsHMap "_backward_linalg_gemm2(ndarray)" args ->
                           IO [NDArrayHandle]
_backward_linalg_gemm2 args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_gemm2"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_linalg_gemm2_upd ::
                           forall args . Fullfilled "_backward_linalg_gemm2(ndarray)" args =>
                             [NDArrayHandle] ->
                               ArgsHMap "_backward_linalg_gemm2(ndarray)" args -> IO ()
_backward_linalg_gemm2_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_gemm2"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_linalg_potrf(ndarray)" =
     '[ '("_A", AttrOpt NDArrayHandle)]

_linalg_potrf ::
              forall args . Fullfilled "_linalg_potrf(ndarray)" args =>
                ArgsHMap "_linalg_potrf(ndarray)" args -> IO [NDArrayHandle]
_linalg_potrf args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("_A",) <$> (args !? #_A :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_potrf"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_linalg_potrf_upd ::
                  forall args . Fullfilled "_linalg_potrf(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_linalg_potrf(ndarray)" args -> IO ()
_linalg_potrf_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("_A",) <$> (args !? #_A :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_potrf"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_linalg_potrf(ndarray)" = '[]

_backward_linalg_potrf ::
                       forall args . Fullfilled "_backward_linalg_potrf(ndarray)" args =>
                         ArgsHMap "_backward_linalg_potrf(ndarray)" args ->
                           IO [NDArrayHandle]
_backward_linalg_potrf args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_potrf"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_linalg_potrf_upd ::
                           forall args . Fullfilled "_backward_linalg_potrf(ndarray)" args =>
                             [NDArrayHandle] ->
                               ArgsHMap "_backward_linalg_potrf(ndarray)" args -> IO ()
_backward_linalg_potrf_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_potrf"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_linalg_potri(ndarray)" =
     '[ '("_A", AttrOpt NDArrayHandle)]

_linalg_potri ::
              forall args . Fullfilled "_linalg_potri(ndarray)" args =>
                ArgsHMap "_linalg_potri(ndarray)" args -> IO [NDArrayHandle]
_linalg_potri args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("_A",) <$> (args !? #_A :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_potri"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_linalg_potri_upd ::
                  forall args . Fullfilled "_linalg_potri(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_linalg_potri(ndarray)" args -> IO ()
_linalg_potri_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("_A",) <$> (args !? #_A :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_potri"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_linalg_potri(ndarray)" = '[]

_backward_linalg_potri ::
                       forall args . Fullfilled "_backward_linalg_potri(ndarray)" args =>
                         ArgsHMap "_backward_linalg_potri(ndarray)" args ->
                           IO [NDArrayHandle]
_backward_linalg_potri args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_potri"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_linalg_potri_upd ::
                           forall args . Fullfilled "_backward_linalg_potri(ndarray)" args =>
                             [NDArrayHandle] ->
                               ArgsHMap "_backward_linalg_potri(ndarray)" args -> IO ()
_backward_linalg_potri_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_potri"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_linalg_trmm(ndarray)" =
     '[ '("transpose", AttrOpt Bool), '("rightside", AttrOpt Bool),
        '("alpha", AttrOpt Double), '("_A", AttrOpt NDArrayHandle),
        '("_B", AttrOpt NDArrayHandle)]

_linalg_trmm ::
             forall args . Fullfilled "_linalg_trmm(ndarray)" args =>
               ArgsHMap "_linalg_trmm(ndarray)" args -> IO [NDArrayHandle]
_linalg_trmm args
  = let scalarArgs
          = catMaybes
              [("transpose",) . showValue <$> (args !? #transpose :: Maybe Bool),
               ("rightside",) . showValue <$> (args !? #rightside :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double)]
        tensorArgs
          = catMaybes
              [("_A",) <$> (args !? #_A :: Maybe NDArrayHandle),
               ("_B",) <$> (args !? #_B :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_trmm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_linalg_trmm_upd ::
                 forall args . Fullfilled "_linalg_trmm(ndarray)" args =>
                   [NDArrayHandle] -> ArgsHMap "_linalg_trmm(ndarray)" args -> IO ()
_linalg_trmm_upd outputs args
  = let scalarArgs
          = catMaybes
              [("transpose",) . showValue <$> (args !? #transpose :: Maybe Bool),
               ("rightside",) . showValue <$> (args !? #rightside :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double)]
        tensorArgs
          = catMaybes
              [("_A",) <$> (args !? #_A :: Maybe NDArrayHandle),
               ("_B",) <$> (args !? #_B :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_trmm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_linalg_trmm(ndarray)" = '[]

_backward_linalg_trmm ::
                      forall args . Fullfilled "_backward_linalg_trmm(ndarray)" args =>
                        ArgsHMap "_backward_linalg_trmm(ndarray)" args ->
                          IO [NDArrayHandle]
_backward_linalg_trmm args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_trmm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_linalg_trmm_upd ::
                          forall args . Fullfilled "_backward_linalg_trmm(ndarray)" args =>
                            [NDArrayHandle] ->
                              ArgsHMap "_backward_linalg_trmm(ndarray)" args -> IO ()
_backward_linalg_trmm_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_trmm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_linalg_trsm(ndarray)" =
     '[ '("transpose", AttrOpt Bool), '("rightside", AttrOpt Bool),
        '("alpha", AttrOpt Double), '("_A", AttrOpt NDArrayHandle),
        '("_B", AttrOpt NDArrayHandle)]

_linalg_trsm ::
             forall args . Fullfilled "_linalg_trsm(ndarray)" args =>
               ArgsHMap "_linalg_trsm(ndarray)" args -> IO [NDArrayHandle]
_linalg_trsm args
  = let scalarArgs
          = catMaybes
              [("transpose",) . showValue <$> (args !? #transpose :: Maybe Bool),
               ("rightside",) . showValue <$> (args !? #rightside :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double)]
        tensorArgs
          = catMaybes
              [("_A",) <$> (args !? #_A :: Maybe NDArrayHandle),
               ("_B",) <$> (args !? #_B :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_trsm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_linalg_trsm_upd ::
                 forall args . Fullfilled "_linalg_trsm(ndarray)" args =>
                   [NDArrayHandle] -> ArgsHMap "_linalg_trsm(ndarray)" args -> IO ()
_linalg_trsm_upd outputs args
  = let scalarArgs
          = catMaybes
              [("transpose",) . showValue <$> (args !? #transpose :: Maybe Bool),
               ("rightside",) . showValue <$> (args !? #rightside :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double)]
        tensorArgs
          = catMaybes
              [("_A",) <$> (args !? #_A :: Maybe NDArrayHandle),
               ("_B",) <$> (args !? #_B :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_trsm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_linalg_trsm(ndarray)" = '[]

_backward_linalg_trsm ::
                      forall args . Fullfilled "_backward_linalg_trsm(ndarray)" args =>
                        ArgsHMap "_backward_linalg_trsm(ndarray)" args ->
                          IO [NDArrayHandle]
_backward_linalg_trsm args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_trsm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_linalg_trsm_upd ::
                          forall args . Fullfilled "_backward_linalg_trsm(ndarray)" args =>
                            [NDArrayHandle] ->
                              ArgsHMap "_backward_linalg_trsm(ndarray)" args -> IO ()
_backward_linalg_trsm_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_trsm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_linalg_sumlogdiag(ndarray)" =
     '[ '("_A", AttrOpt NDArrayHandle)]

_linalg_sumlogdiag ::
                   forall args . Fullfilled "_linalg_sumlogdiag(ndarray)" args =>
                     ArgsHMap "_linalg_sumlogdiag(ndarray)" args -> IO [NDArrayHandle]
_linalg_sumlogdiag args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("_A",) <$> (args !? #_A :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_sumlogdiag"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_linalg_sumlogdiag_upd ::
                       forall args . Fullfilled "_linalg_sumlogdiag(ndarray)" args =>
                         [NDArrayHandle] ->
                           ArgsHMap "_linalg_sumlogdiag(ndarray)" args -> IO ()
_linalg_sumlogdiag_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("_A",) <$> (args !? #_A :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_sumlogdiag"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_linalg_sumlogdiag(ndarray)"
     = '[]

_backward_linalg_sumlogdiag ::
                            forall args .
                              Fullfilled "_backward_linalg_sumlogdiag(ndarray)" args =>
                              ArgsHMap "_backward_linalg_sumlogdiag(ndarray)" args ->
                                IO [NDArrayHandle]
_backward_linalg_sumlogdiag args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_sumlogdiag"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_linalg_sumlogdiag_upd ::
                                forall args .
                                  Fullfilled "_backward_linalg_sumlogdiag(ndarray)" args =>
                                  [NDArrayHandle] ->
                                    ArgsHMap "_backward_linalg_sumlogdiag(ndarray)" args -> IO ()
_backward_linalg_sumlogdiag_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_sumlogdiag"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_linalg_syrk(ndarray)" =
     '[ '("transpose", AttrOpt Bool), '("alpha", AttrOpt Double),
        '("_A", AttrOpt NDArrayHandle)]

_linalg_syrk ::
             forall args . Fullfilled "_linalg_syrk(ndarray)" args =>
               ArgsHMap "_linalg_syrk(ndarray)" args -> IO [NDArrayHandle]
_linalg_syrk args
  = let scalarArgs
          = catMaybes
              [("transpose",) . showValue <$> (args !? #transpose :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double)]
        tensorArgs
          = catMaybes [("_A",) <$> (args !? #_A :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_syrk"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_linalg_syrk_upd ::
                 forall args . Fullfilled "_linalg_syrk(ndarray)" args =>
                   [NDArrayHandle] -> ArgsHMap "_linalg_syrk(ndarray)" args -> IO ()
_linalg_syrk_upd outputs args
  = let scalarArgs
          = catMaybes
              [("transpose",) . showValue <$> (args !? #transpose :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double)]
        tensorArgs
          = catMaybes [("_A",) <$> (args !? #_A :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_syrk"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_linalg_syrk(ndarray)" = '[]

_backward_linalg_syrk ::
                      forall args . Fullfilled "_backward_linalg_syrk(ndarray)" args =>
                        ArgsHMap "_backward_linalg_syrk(ndarray)" args ->
                          IO [NDArrayHandle]
_backward_linalg_syrk args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_syrk"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_linalg_syrk_upd ::
                          forall args . Fullfilled "_backward_linalg_syrk(ndarray)" args =>
                            [NDArrayHandle] ->
                              ArgsHMap "_backward_linalg_syrk(ndarray)" args -> IO ()
_backward_linalg_syrk_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_syrk"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_linalg_gelqf(ndarray)" =
     '[ '("_A", AttrOpt NDArrayHandle)]

_linalg_gelqf ::
              forall args . Fullfilled "_linalg_gelqf(ndarray)" args =>
                ArgsHMap "_linalg_gelqf(ndarray)" args -> IO [NDArrayHandle]
_linalg_gelqf args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("_A",) <$> (args !? #_A :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_gelqf"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_linalg_gelqf_upd ::
                  forall args . Fullfilled "_linalg_gelqf(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_linalg_gelqf(ndarray)" args -> IO ()
_linalg_gelqf_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("_A",) <$> (args !? #_A :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_gelqf"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_linalg_gelqf(ndarray)" = '[]

_backward_linalg_gelqf ::
                       forall args . Fullfilled "_backward_linalg_gelqf(ndarray)" args =>
                         ArgsHMap "_backward_linalg_gelqf(ndarray)" args ->
                           IO [NDArrayHandle]
_backward_linalg_gelqf args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_gelqf"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_linalg_gelqf_upd ::
                           forall args . Fullfilled "_backward_linalg_gelqf(ndarray)" args =>
                             [NDArrayHandle] ->
                               ArgsHMap "_backward_linalg_gelqf(ndarray)" args -> IO ()
_backward_linalg_gelqf_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_gelqf"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_linalg_syevd(ndarray)" =
     '[ '("_A", AttrOpt NDArrayHandle)]

_linalg_syevd ::
              forall args . Fullfilled "_linalg_syevd(ndarray)" args =>
                ArgsHMap "_linalg_syevd(ndarray)" args -> IO [NDArrayHandle]
_linalg_syevd args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("_A",) <$> (args !? #_A :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_syevd"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_linalg_syevd_upd ::
                  forall args . Fullfilled "_linalg_syevd(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_linalg_syevd(ndarray)" args -> IO ()
_linalg_syevd_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("_A",) <$> (args !? #_A :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_syevd"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_linalg_syevd(ndarray)" = '[]

_backward_linalg_syevd ::
                       forall args . Fullfilled "_backward_linalg_syevd(ndarray)" args =>
                         ArgsHMap "_backward_linalg_syevd(ndarray)" args ->
                           IO [NDArrayHandle]
_backward_linalg_syevd args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_syevd"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_linalg_syevd_upd ::
                           forall args . Fullfilled "_backward_linalg_syevd(ndarray)" args =>
                             [NDArrayHandle] ->
                               ArgsHMap "_backward_linalg_syevd(ndarray)" args -> IO ()
_backward_linalg_syevd_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_syevd"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "cast_storage(ndarray)" =
     '[ '("stype",
          AttrReq (EnumType '["csr", "default", "row_sparse"])),
        '("data", AttrOpt NDArrayHandle)]

cast_storage ::
             forall args . Fullfilled "cast_storage(ndarray)" args =>
               ArgsHMap "cast_storage(ndarray)" args -> IO [NDArrayHandle]
cast_storage args
  = let scalarArgs
          = catMaybes
              [("stype",) . showValue <$>
                 (args !? #stype ::
                    Maybe (EnumType '["csr", "default", "row_sparse"]))]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "cast_storage"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

cast_storage_upd ::
                 forall args . Fullfilled "cast_storage(ndarray)" args =>
                   [NDArrayHandle] -> ArgsHMap "cast_storage(ndarray)" args -> IO ()
cast_storage_upd outputs args
  = let scalarArgs
          = catMaybes
              [("stype",) . showValue <$>
                 (args !? #stype ::
                    Maybe (EnumType '["csr", "default", "row_sparse"]))]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "cast_storage"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_equal_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_equal_scalar ::
              forall args . Fullfilled "_equal_scalar(ndarray)" args =>
                ArgsHMap "_equal_scalar(ndarray)" args -> IO [NDArrayHandle]
_equal_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_equal_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_equal_scalar_upd ::
                  forall args . Fullfilled "_equal_scalar(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_equal_scalar(ndarray)" args -> IO ()
_equal_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_equal_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_not_equal_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_not_equal_scalar ::
                  forall args . Fullfilled "_not_equal_scalar(ndarray)" args =>
                    ArgsHMap "_not_equal_scalar(ndarray)" args -> IO [NDArrayHandle]
_not_equal_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_not_equal_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_not_equal_scalar_upd ::
                      forall args . Fullfilled "_not_equal_scalar(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "_not_equal_scalar(ndarray)" args -> IO ()
_not_equal_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_not_equal_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_greater_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_greater_scalar ::
                forall args . Fullfilled "_greater_scalar(ndarray)" args =>
                  ArgsHMap "_greater_scalar(ndarray)" args -> IO [NDArrayHandle]
_greater_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_greater_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_greater_scalar_upd ::
                    forall args . Fullfilled "_greater_scalar(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_greater_scalar(ndarray)" args -> IO ()
_greater_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_greater_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_greater_equal_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_greater_equal_scalar ::
                      forall args . Fullfilled "_greater_equal_scalar(ndarray)" args =>
                        ArgsHMap "_greater_equal_scalar(ndarray)" args ->
                          IO [NDArrayHandle]
_greater_equal_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_greater_equal_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_greater_equal_scalar_upd ::
                          forall args . Fullfilled "_greater_equal_scalar(ndarray)" args =>
                            [NDArrayHandle] ->
                              ArgsHMap "_greater_equal_scalar(ndarray)" args -> IO ()
_greater_equal_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_greater_equal_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_lesser_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_lesser_scalar ::
               forall args . Fullfilled "_lesser_scalar(ndarray)" args =>
                 ArgsHMap "_lesser_scalar(ndarray)" args -> IO [NDArrayHandle]
_lesser_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_lesser_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_lesser_scalar_upd ::
                   forall args . Fullfilled "_lesser_scalar(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_lesser_scalar(ndarray)" args -> IO ()
_lesser_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_lesser_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_lesser_equal_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_lesser_equal_scalar ::
                     forall args . Fullfilled "_lesser_equal_scalar(ndarray)" args =>
                       ArgsHMap "_lesser_equal_scalar(ndarray)" args -> IO [NDArrayHandle]
_lesser_equal_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_lesser_equal_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_lesser_equal_scalar_upd ::
                         forall args . Fullfilled "_lesser_equal_scalar(ndarray)" args =>
                           [NDArrayHandle] ->
                             ArgsHMap "_lesser_equal_scalar(ndarray)" args -> IO ()
_lesser_equal_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_lesser_equal_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "relu(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

relu ::
     forall args . Fullfilled "relu(ndarray)" args =>
       ArgsHMap "relu(ndarray)" args -> IO [NDArrayHandle]
relu args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "relu"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

relu_upd ::
         forall args . Fullfilled "relu(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "relu(ndarray)" args -> IO ()
relu_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "relu"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_relu(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_relu ::
               forall args . Fullfilled "_backward_relu(ndarray)" args =>
                 ArgsHMap "_backward_relu(ndarray)" args -> IO [NDArrayHandle]
_backward_relu args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_relu"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_relu_upd ::
                   forall args . Fullfilled "_backward_relu(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_backward_relu(ndarray)" args -> IO ()
_backward_relu_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_relu"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "sigmoid(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

sigmoid ::
        forall args . Fullfilled "sigmoid(ndarray)" args =>
          ArgsHMap "sigmoid(ndarray)" args -> IO [NDArrayHandle]
sigmoid args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sigmoid"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

sigmoid_upd ::
            forall args . Fullfilled "sigmoid(ndarray)" args =>
              [NDArrayHandle] -> ArgsHMap "sigmoid(ndarray)" args -> IO ()
sigmoid_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sigmoid"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_sigmoid(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_sigmoid ::
                  forall args . Fullfilled "_backward_sigmoid(ndarray)" args =>
                    ArgsHMap "_backward_sigmoid(ndarray)" args -> IO [NDArrayHandle]
_backward_sigmoid args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sigmoid"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_sigmoid_upd ::
                      forall args . Fullfilled "_backward_sigmoid(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "_backward_sigmoid(ndarray)" args -> IO ()
_backward_sigmoid_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sigmoid"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_copy(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

_copy ::
      forall args . Fullfilled "_copy(ndarray)" args =>
        ArgsHMap "_copy(ndarray)" args -> IO [NDArrayHandle]
_copy args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_copy"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_copy_upd ::
          forall args . Fullfilled "_copy(ndarray)" args =>
            [NDArrayHandle] -> ArgsHMap "_copy(ndarray)" args -> IO ()
_copy_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_copy"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_copy(ndarray)" = '[]

_backward_copy ::
               forall args . Fullfilled "_backward_copy(ndarray)" args =>
                 ArgsHMap "_backward_copy(ndarray)" args -> IO [NDArrayHandle]
_backward_copy args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_copy"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_copy_upd ::
                   forall args . Fullfilled "_backward_copy(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_backward_copy(ndarray)" args -> IO ()
_backward_copy_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_copy"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_BlockGrad(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

_BlockGrad ::
           forall args . Fullfilled "_BlockGrad(ndarray)" args =>
             ArgsHMap "_BlockGrad(ndarray)" args -> IO [NDArrayHandle]
_BlockGrad args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "BlockGrad"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_BlockGrad_upd ::
               forall args . Fullfilled "_BlockGrad(ndarray)" args =>
                 [NDArrayHandle] -> ArgsHMap "_BlockGrad(ndarray)" args -> IO ()
_BlockGrad_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "BlockGrad"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "make_loss(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

make_loss ::
          forall args . Fullfilled "make_loss(ndarray)" args =>
            ArgsHMap "make_loss(ndarray)" args -> IO [NDArrayHandle]
make_loss args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "make_loss"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

make_loss_upd ::
              forall args . Fullfilled "make_loss(ndarray)" args =>
                [NDArrayHandle] -> ArgsHMap "make_loss(ndarray)" args -> IO ()
make_loss_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "make_loss"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_identity_with_attr_like_rhs(ndarray)"
     =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_identity_with_attr_like_rhs ::
                             forall args .
                               Fullfilled "_identity_with_attr_like_rhs(ndarray)" args =>
                               ArgsHMap "_identity_with_attr_like_rhs(ndarray)" args ->
                                 IO [NDArrayHandle]
_identity_with_attr_like_rhs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_identity_with_attr_like_rhs"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_identity_with_attr_like_rhs_upd ::
                                 forall args .
                                   Fullfilled "_identity_with_attr_like_rhs(ndarray)" args =>
                                   [NDArrayHandle] ->
                                     ArgsHMap "_identity_with_attr_like_rhs(ndarray)" args -> IO ()
_identity_with_attr_like_rhs_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_identity_with_attr_like_rhs"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "reshape_like(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

reshape_like ::
             forall args . Fullfilled "reshape_like(ndarray)" args =>
               ArgsHMap "reshape_like(ndarray)" args -> IO [NDArrayHandle]
reshape_like args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "reshape_like"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

reshape_like_upd ::
                 forall args . Fullfilled "reshape_like(ndarray)" args =>
                   [NDArrayHandle] -> ArgsHMap "reshape_like(ndarray)" args -> IO ()
reshape_like_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "reshape_like"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_Cast(ndarray)" =
     '[ '("dtype",
          AttrReq
            (EnumType '["float16", "float32", "float64", "int32", "uint8"])),
        '("data", AttrOpt NDArrayHandle)]

_Cast ::
      forall args . Fullfilled "_Cast(ndarray)" args =>
        ArgsHMap "_Cast(ndarray)" args -> IO [NDArrayHandle]
_Cast args
  = let scalarArgs
          = catMaybes
              [("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType '["float16", "float32", "float64", "int32", "uint8"]))]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Cast"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_Cast_upd ::
          forall args . Fullfilled "_Cast(ndarray)" args =>
            [NDArrayHandle] -> ArgsHMap "_Cast(ndarray)" args -> IO ()
_Cast_upd outputs args
  = let scalarArgs
          = catMaybes
              [("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType '["float16", "float32", "float64", "int32", "uint8"]))]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Cast"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_cast(ndarray)" = '[]

_backward_cast ::
               forall args . Fullfilled "_backward_cast(ndarray)" args =>
                 ArgsHMap "_backward_cast(ndarray)" args -> IO [NDArrayHandle]
_backward_cast args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_cast"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_cast_upd ::
                   forall args . Fullfilled "_backward_cast(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_backward_cast(ndarray)" args -> IO ()
_backward_cast_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_cast"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "negative(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

negative ::
         forall args . Fullfilled "negative(ndarray)" args =>
           ArgsHMap "negative(ndarray)" args -> IO [NDArrayHandle]
negative args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "negative"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

negative_upd ::
             forall args . Fullfilled "negative(ndarray)" args =>
               [NDArrayHandle] -> ArgsHMap "negative(ndarray)" args -> IO ()
negative_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "negative"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "reciprocal(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

reciprocal ::
           forall args . Fullfilled "reciprocal(ndarray)" args =>
             ArgsHMap "reciprocal(ndarray)" args -> IO [NDArrayHandle]
reciprocal args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "reciprocal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

reciprocal_upd ::
               forall args . Fullfilled "reciprocal(ndarray)" args =>
                 [NDArrayHandle] -> ArgsHMap "reciprocal(ndarray)" args -> IO ()
reciprocal_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "reciprocal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_reciprocal(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_reciprocal ::
                     forall args . Fullfilled "_backward_reciprocal(ndarray)" args =>
                       ArgsHMap "_backward_reciprocal(ndarray)" args -> IO [NDArrayHandle]
_backward_reciprocal args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_reciprocal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_reciprocal_upd ::
                         forall args . Fullfilled "_backward_reciprocal(ndarray)" args =>
                           [NDArrayHandle] ->
                             ArgsHMap "_backward_reciprocal(ndarray)" args -> IO ()
_backward_reciprocal_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_reciprocal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "abs(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

abs ::
    forall args . Fullfilled "abs(ndarray)" args =>
      ArgsHMap "abs(ndarray)" args -> IO [NDArrayHandle]
abs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "abs"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

abs_upd ::
        forall args . Fullfilled "abs(ndarray)" args =>
          [NDArrayHandle] -> ArgsHMap "abs(ndarray)" args -> IO ()
abs_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "abs"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_abs(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_abs ::
              forall args . Fullfilled "_backward_abs(ndarray)" args =>
                ArgsHMap "_backward_abs(ndarray)" args -> IO [NDArrayHandle]
_backward_abs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_abs"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_abs_upd ::
                  forall args . Fullfilled "_backward_abs(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_backward_abs(ndarray)" args -> IO ()
_backward_abs_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_abs"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "sign(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

sign ::
     forall args . Fullfilled "sign(ndarray)" args =>
       ArgsHMap "sign(ndarray)" args -> IO [NDArrayHandle]
sign args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sign"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

sign_upd ::
         forall args . Fullfilled "sign(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "sign(ndarray)" args -> IO ()
sign_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sign"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_sign(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_sign ::
               forall args . Fullfilled "_backward_sign(ndarray)" args =>
                 ArgsHMap "_backward_sign(ndarray)" args -> IO [NDArrayHandle]
_backward_sign args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sign"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_sign_upd ::
                   forall args . Fullfilled "_backward_sign(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_backward_sign(ndarray)" args -> IO ()
_backward_sign_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sign"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "round(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

round ::
      forall args . Fullfilled "round(ndarray)" args =>
        ArgsHMap "round(ndarray)" args -> IO [NDArrayHandle]
round args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "round"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

round_upd ::
          forall args . Fullfilled "round(ndarray)" args =>
            [NDArrayHandle] -> ArgsHMap "round(ndarray)" args -> IO ()
round_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "round"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "rint(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

rint ::
     forall args . Fullfilled "rint(ndarray)" args =>
       ArgsHMap "rint(ndarray)" args -> IO [NDArrayHandle]
rint args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "rint"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

rint_upd ::
         forall args . Fullfilled "rint(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "rint(ndarray)" args -> IO ()
rint_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "rint"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "ceil(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

ceil ::
     forall args . Fullfilled "ceil(ndarray)" args =>
       ArgsHMap "ceil(ndarray)" args -> IO [NDArrayHandle]
ceil args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "ceil"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

ceil_upd ::
         forall args . Fullfilled "ceil(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "ceil(ndarray)" args -> IO ()
ceil_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "ceil"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "floor(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

floor ::
      forall args . Fullfilled "floor(ndarray)" args =>
        ArgsHMap "floor(ndarray)" args -> IO [NDArrayHandle]
floor args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "floor"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

floor_upd ::
          forall args . Fullfilled "floor(ndarray)" args =>
            [NDArrayHandle] -> ArgsHMap "floor(ndarray)" args -> IO ()
floor_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "floor"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "trunc(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

trunc ::
      forall args . Fullfilled "trunc(ndarray)" args =>
        ArgsHMap "trunc(ndarray)" args -> IO [NDArrayHandle]
trunc args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "trunc"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

trunc_upd ::
          forall args . Fullfilled "trunc(ndarray)" args =>
            [NDArrayHandle] -> ArgsHMap "trunc(ndarray)" args -> IO ()
trunc_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "trunc"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "fix(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

fix ::
    forall args . Fullfilled "fix(ndarray)" args =>
      ArgsHMap "fix(ndarray)" args -> IO [NDArrayHandle]
fix args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "fix"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

fix_upd ::
        forall args . Fullfilled "fix(ndarray)" args =>
          [NDArrayHandle] -> ArgsHMap "fix(ndarray)" args -> IO ()
fix_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "fix"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "square(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

square ::
       forall args . Fullfilled "square(ndarray)" args =>
         ArgsHMap "square(ndarray)" args -> IO [NDArrayHandle]
square args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "square"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

square_upd ::
           forall args . Fullfilled "square(ndarray)" args =>
             [NDArrayHandle] -> ArgsHMap "square(ndarray)" args -> IO ()
square_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "square"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_square(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_square ::
                 forall args . Fullfilled "_backward_square(ndarray)" args =>
                   ArgsHMap "_backward_square(ndarray)" args -> IO [NDArrayHandle]
_backward_square args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_square"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_square_upd ::
                     forall args . Fullfilled "_backward_square(ndarray)" args =>
                       [NDArrayHandle] ->
                         ArgsHMap "_backward_square(ndarray)" args -> IO ()
_backward_square_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_square"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "sqrt(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

sqrt ::
     forall args . Fullfilled "sqrt(ndarray)" args =>
       ArgsHMap "sqrt(ndarray)" args -> IO [NDArrayHandle]
sqrt args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sqrt"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

sqrt_upd ::
         forall args . Fullfilled "sqrt(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "sqrt(ndarray)" args -> IO ()
sqrt_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sqrt"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_sqrt(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_sqrt ::
               forall args . Fullfilled "_backward_sqrt(ndarray)" args =>
                 ArgsHMap "_backward_sqrt(ndarray)" args -> IO [NDArrayHandle]
_backward_sqrt args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sqrt"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_sqrt_upd ::
                   forall args . Fullfilled "_backward_sqrt(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_backward_sqrt(ndarray)" args -> IO ()
_backward_sqrt_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sqrt"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "rsqrt(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

rsqrt ::
      forall args . Fullfilled "rsqrt(ndarray)" args =>
        ArgsHMap "rsqrt(ndarray)" args -> IO [NDArrayHandle]
rsqrt args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "rsqrt"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

rsqrt_upd ::
          forall args . Fullfilled "rsqrt(ndarray)" args =>
            [NDArrayHandle] -> ArgsHMap "rsqrt(ndarray)" args -> IO ()
rsqrt_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "rsqrt"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_rsqrt(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_rsqrt ::
                forall args . Fullfilled "_backward_rsqrt(ndarray)" args =>
                  ArgsHMap "_backward_rsqrt(ndarray)" args -> IO [NDArrayHandle]
_backward_rsqrt args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_rsqrt"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_rsqrt_upd ::
                    forall args . Fullfilled "_backward_rsqrt(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_backward_rsqrt(ndarray)" args -> IO ()
_backward_rsqrt_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_rsqrt"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "cbrt(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

cbrt ::
     forall args . Fullfilled "cbrt(ndarray)" args =>
       ArgsHMap "cbrt(ndarray)" args -> IO [NDArrayHandle]
cbrt args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "cbrt"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

cbrt_upd ::
         forall args . Fullfilled "cbrt(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "cbrt(ndarray)" args -> IO ()
cbrt_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "cbrt"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_cbrt(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_cbrt ::
               forall args . Fullfilled "_backward_cbrt(ndarray)" args =>
                 ArgsHMap "_backward_cbrt(ndarray)" args -> IO [NDArrayHandle]
_backward_cbrt args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_cbrt"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_cbrt_upd ::
                   forall args . Fullfilled "_backward_cbrt(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_backward_cbrt(ndarray)" args -> IO ()
_backward_cbrt_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_cbrt"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "rcbrt(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

rcbrt ::
      forall args . Fullfilled "rcbrt(ndarray)" args =>
        ArgsHMap "rcbrt(ndarray)" args -> IO [NDArrayHandle]
rcbrt args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "rcbrt"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

rcbrt_upd ::
          forall args . Fullfilled "rcbrt(ndarray)" args =>
            [NDArrayHandle] -> ArgsHMap "rcbrt(ndarray)" args -> IO ()
rcbrt_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "rcbrt"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_rcbrt(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_rcbrt ::
                forall args . Fullfilled "_backward_rcbrt(ndarray)" args =>
                  ArgsHMap "_backward_rcbrt(ndarray)" args -> IO [NDArrayHandle]
_backward_rcbrt args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_rcbrt"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_rcbrt_upd ::
                    forall args . Fullfilled "_backward_rcbrt(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_backward_rcbrt(ndarray)" args -> IO ()
_backward_rcbrt_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_rcbrt"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "exp(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

exp ::
    forall args . Fullfilled "exp(ndarray)" args =>
      ArgsHMap "exp(ndarray)" args -> IO [NDArrayHandle]
exp args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "exp"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

exp_upd ::
        forall args . Fullfilled "exp(ndarray)" args =>
          [NDArrayHandle] -> ArgsHMap "exp(ndarray)" args -> IO ()
exp_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "exp"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "log(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

log ::
    forall args . Fullfilled "log(ndarray)" args =>
      ArgsHMap "log(ndarray)" args -> IO [NDArrayHandle]
log args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "log"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

log_upd ::
        forall args . Fullfilled "log(ndarray)" args =>
          [NDArrayHandle] -> ArgsHMap "log(ndarray)" args -> IO ()
log_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "log"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "log10(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

log10 ::
      forall args . Fullfilled "log10(ndarray)" args =>
        ArgsHMap "log10(ndarray)" args -> IO [NDArrayHandle]
log10 args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "log10"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

log10_upd ::
          forall args . Fullfilled "log10(ndarray)" args =>
            [NDArrayHandle] -> ArgsHMap "log10(ndarray)" args -> IO ()
log10_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "log10"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "log2(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

log2 ::
     forall args . Fullfilled "log2(ndarray)" args =>
       ArgsHMap "log2(ndarray)" args -> IO [NDArrayHandle]
log2 args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "log2"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

log2_upd ::
         forall args . Fullfilled "log2(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "log2(ndarray)" args -> IO ()
log2_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "log2"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_log(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_log ::
              forall args . Fullfilled "_backward_log(ndarray)" args =>
                ArgsHMap "_backward_log(ndarray)" args -> IO [NDArrayHandle]
_backward_log args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_log"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_log_upd ::
                  forall args . Fullfilled "_backward_log(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_backward_log(ndarray)" args -> IO ()
_backward_log_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_log"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_log10(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_log10 ::
                forall args . Fullfilled "_backward_log10(ndarray)" args =>
                  ArgsHMap "_backward_log10(ndarray)" args -> IO [NDArrayHandle]
_backward_log10 args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_log10"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_log10_upd ::
                    forall args . Fullfilled "_backward_log10(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_backward_log10(ndarray)" args -> IO ()
_backward_log10_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_log10"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_log2(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_log2 ::
               forall args . Fullfilled "_backward_log2(ndarray)" args =>
                 ArgsHMap "_backward_log2(ndarray)" args -> IO [NDArrayHandle]
_backward_log2 args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_log2"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_log2_upd ::
                   forall args . Fullfilled "_backward_log2(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_backward_log2(ndarray)" args -> IO ()
_backward_log2_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_log2"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "log1p(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

log1p ::
      forall args . Fullfilled "log1p(ndarray)" args =>
        ArgsHMap "log1p(ndarray)" args -> IO [NDArrayHandle]
log1p args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "log1p"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

log1p_upd ::
          forall args . Fullfilled "log1p(ndarray)" args =>
            [NDArrayHandle] -> ArgsHMap "log1p(ndarray)" args -> IO ()
log1p_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "log1p"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_log1p(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_log1p ::
                forall args . Fullfilled "_backward_log1p(ndarray)" args =>
                  ArgsHMap "_backward_log1p(ndarray)" args -> IO [NDArrayHandle]
_backward_log1p args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_log1p"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_log1p_upd ::
                    forall args . Fullfilled "_backward_log1p(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_backward_log1p(ndarray)" args -> IO ()
_backward_log1p_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_log1p"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "expm1(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

expm1 ::
      forall args . Fullfilled "expm1(ndarray)" args =>
        ArgsHMap "expm1(ndarray)" args -> IO [NDArrayHandle]
expm1 args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "expm1"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

expm1_upd ::
          forall args . Fullfilled "expm1(ndarray)" args =>
            [NDArrayHandle] -> ArgsHMap "expm1(ndarray)" args -> IO ()
expm1_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "expm1"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_expm1(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_expm1 ::
                forall args . Fullfilled "_backward_expm1(ndarray)" args =>
                  ArgsHMap "_backward_expm1(ndarray)" args -> IO [NDArrayHandle]
_backward_expm1 args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_expm1"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_expm1_upd ::
                    forall args . Fullfilled "_backward_expm1(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_backward_expm1(ndarray)" args -> IO ()
_backward_expm1_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_expm1"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "gamma(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

gamma ::
      forall args . Fullfilled "gamma(ndarray)" args =>
        ArgsHMap "gamma(ndarray)" args -> IO [NDArrayHandle]
gamma args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "gamma"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

gamma_upd ::
          forall args . Fullfilled "gamma(ndarray)" args =>
            [NDArrayHandle] -> ArgsHMap "gamma(ndarray)" args -> IO ()
gamma_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "gamma"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_gamma(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_gamma ::
                forall args . Fullfilled "_backward_gamma(ndarray)" args =>
                  ArgsHMap "_backward_gamma(ndarray)" args -> IO [NDArrayHandle]
_backward_gamma args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_gamma"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_gamma_upd ::
                    forall args . Fullfilled "_backward_gamma(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_backward_gamma(ndarray)" args -> IO ()
_backward_gamma_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_gamma"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "gammaln(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

gammaln ::
        forall args . Fullfilled "gammaln(ndarray)" args =>
          ArgsHMap "gammaln(ndarray)" args -> IO [NDArrayHandle]
gammaln args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "gammaln"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

gammaln_upd ::
            forall args . Fullfilled "gammaln(ndarray)" args =>
              [NDArrayHandle] -> ArgsHMap "gammaln(ndarray)" args -> IO ()
gammaln_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "gammaln"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_gammaln(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_gammaln ::
                  forall args . Fullfilled "_backward_gammaln(ndarray)" args =>
                    ArgsHMap "_backward_gammaln(ndarray)" args -> IO [NDArrayHandle]
_backward_gammaln args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_gammaln"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_gammaln_upd ::
                      forall args . Fullfilled "_backward_gammaln(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "_backward_gammaln(ndarray)" args -> IO ()
_backward_gammaln_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_gammaln"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "sum(ndarray)" =
     '[ '("axis", AttrOpt [Int]), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt NDArrayHandle)]

sum ::
    forall args . Fullfilled "sum(ndarray)" args =>
      ArgsHMap "sum(ndarray)" args -> IO [NDArrayHandle]
sum args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int]),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

sum_upd ::
        forall args . Fullfilled "sum(ndarray)" args =>
          [NDArrayHandle] -> ArgsHMap "sum(ndarray)" args -> IO ()
sum_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int]),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_sum(ndarray)" = '[]

_backward_sum ::
              forall args . Fullfilled "_backward_sum(ndarray)" args =>
                ArgsHMap "_backward_sum(ndarray)" args -> IO [NDArrayHandle]
_backward_sum args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_sum_upd ::
                  forall args . Fullfilled "_backward_sum(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_backward_sum(ndarray)" args -> IO ()
_backward_sum_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "mean(ndarray)" =
     '[ '("axis", AttrOpt [Int]), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt NDArrayHandle)]

mean ::
     forall args . Fullfilled "mean(ndarray)" args =>
       ArgsHMap "mean(ndarray)" args -> IO [NDArrayHandle]
mean args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int]),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "mean"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

mean_upd ::
         forall args . Fullfilled "mean(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "mean(ndarray)" args -> IO ()
mean_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int]),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "mean"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_mean(ndarray)" = '[]

_backward_mean ::
               forall args . Fullfilled "_backward_mean(ndarray)" args =>
                 ArgsHMap "_backward_mean(ndarray)" args -> IO [NDArrayHandle]
_backward_mean args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_mean"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_mean_upd ::
                   forall args . Fullfilled "_backward_mean(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_backward_mean(ndarray)" args -> IO ()
_backward_mean_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_mean"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "prod(ndarray)" =
     '[ '("axis", AttrOpt [Int]), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt NDArrayHandle)]

prod ::
     forall args . Fullfilled "prod(ndarray)" args =>
       ArgsHMap "prod(ndarray)" args -> IO [NDArrayHandle]
prod args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int]),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "prod"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

prod_upd ::
         forall args . Fullfilled "prod(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "prod(ndarray)" args -> IO ()
prod_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int]),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "prod"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_prod(ndarray)" = '[]

_backward_prod ::
               forall args . Fullfilled "_backward_prod(ndarray)" args =>
                 ArgsHMap "_backward_prod(ndarray)" args -> IO [NDArrayHandle]
_backward_prod args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_prod"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_prod_upd ::
                   forall args . Fullfilled "_backward_prod(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_backward_prod(ndarray)" args -> IO ()
_backward_prod_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_prod"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "nansum(ndarray)" =
     '[ '("axis", AttrOpt [Int]), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt NDArrayHandle)]

nansum ::
       forall args . Fullfilled "nansum(ndarray)" args =>
         ArgsHMap "nansum(ndarray)" args -> IO [NDArrayHandle]
nansum args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int]),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "nansum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

nansum_upd ::
           forall args . Fullfilled "nansum(ndarray)" args =>
             [NDArrayHandle] -> ArgsHMap "nansum(ndarray)" args -> IO ()
nansum_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int]),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "nansum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_nansum(ndarray)" = '[]

_backward_nansum ::
                 forall args . Fullfilled "_backward_nansum(ndarray)" args =>
                   ArgsHMap "_backward_nansum(ndarray)" args -> IO [NDArrayHandle]
_backward_nansum args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_nansum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_nansum_upd ::
                     forall args . Fullfilled "_backward_nansum(ndarray)" args =>
                       [NDArrayHandle] ->
                         ArgsHMap "_backward_nansum(ndarray)" args -> IO ()
_backward_nansum_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_nansum"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "nanprod(ndarray)" =
     '[ '("axis", AttrOpt [Int]), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt NDArrayHandle)]

nanprod ::
        forall args . Fullfilled "nanprod(ndarray)" args =>
          ArgsHMap "nanprod(ndarray)" args -> IO [NDArrayHandle]
nanprod args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int]),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "nanprod"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

nanprod_upd ::
            forall args . Fullfilled "nanprod(ndarray)" args =>
              [NDArrayHandle] -> ArgsHMap "nanprod(ndarray)" args -> IO ()
nanprod_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int]),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "nanprod"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_nanprod(ndarray)" = '[]

_backward_nanprod ::
                  forall args . Fullfilled "_backward_nanprod(ndarray)" args =>
                    ArgsHMap "_backward_nanprod(ndarray)" args -> IO [NDArrayHandle]
_backward_nanprod args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_nanprod"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_nanprod_upd ::
                      forall args . Fullfilled "_backward_nanprod(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "_backward_nanprod(ndarray)" args -> IO ()
_backward_nanprod_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_nanprod"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "max(ndarray)" =
     '[ '("axis", AttrOpt [Int]), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt NDArrayHandle)]

max ::
    forall args . Fullfilled "max(ndarray)" args =>
      ArgsHMap "max(ndarray)" args -> IO [NDArrayHandle]
max args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int]),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "max"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

max_upd ::
        forall args . Fullfilled "max(ndarray)" args =>
          [NDArrayHandle] -> ArgsHMap "max(ndarray)" args -> IO ()
max_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int]),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "max"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_max(ndarray)" = '[]

_backward_max ::
              forall args . Fullfilled "_backward_max(ndarray)" args =>
                ArgsHMap "_backward_max(ndarray)" args -> IO [NDArrayHandle]
_backward_max args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_max"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_max_upd ::
                  forall args . Fullfilled "_backward_max(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_backward_max(ndarray)" args -> IO ()
_backward_max_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_max"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "min(ndarray)" =
     '[ '("axis", AttrOpt [Int]), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt NDArrayHandle)]

min ::
    forall args . Fullfilled "min(ndarray)" args =>
      ArgsHMap "min(ndarray)" args -> IO [NDArrayHandle]
min args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int]),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "min"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

min_upd ::
        forall args . Fullfilled "min(ndarray)" args =>
          [NDArrayHandle] -> ArgsHMap "min(ndarray)" args -> IO ()
min_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int]),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "min"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_min(ndarray)" = '[]

_backward_min ::
              forall args . Fullfilled "_backward_min(ndarray)" args =>
                ArgsHMap "_backward_min(ndarray)" args -> IO [NDArrayHandle]
_backward_min args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_min"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_min_upd ::
                  forall args . Fullfilled "_backward_min(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_backward_min(ndarray)" args -> IO ()
_backward_min_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_min"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "broadcast_axis(ndarray)" =
     '[ '("axis", AttrOpt [Int]), '("size", AttrOpt [Int]),
        '("data", AttrOpt NDArrayHandle)]

broadcast_axis ::
               forall args . Fullfilled "broadcast_axis(ndarray)" args =>
                 ArgsHMap "broadcast_axis(ndarray)" args -> IO [NDArrayHandle]
broadcast_axis args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int]),
               ("size",) . showValue <$> (args !? #size :: Maybe [Int])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_axis"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

broadcast_axis_upd ::
                   forall args . Fullfilled "broadcast_axis(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "broadcast_axis(ndarray)" args -> IO ()
broadcast_axis_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int]),
               ("size",) . showValue <$> (args !? #size :: Maybe [Int])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_axis"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "broadcast_to(ndarray)" =
     '[ '("shape", AttrOpt [Int]), '("data", AttrOpt NDArrayHandle)]

broadcast_to ::
             forall args . Fullfilled "broadcast_to(ndarray)" args =>
               ArgsHMap "broadcast_to(ndarray)" args -> IO [NDArrayHandle]
broadcast_to args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_to"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

broadcast_to_upd ::
                 forall args . Fullfilled "broadcast_to(ndarray)" args =>
                   [NDArrayHandle] -> ArgsHMap "broadcast_to(ndarray)" args -> IO ()
broadcast_to_upd outputs args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_to"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_broadcast_backward(ndarray)" = '[]

_broadcast_backward ::
                    forall args . Fullfilled "_broadcast_backward(ndarray)" args =>
                      ArgsHMap "_broadcast_backward(ndarray)" args -> IO [NDArrayHandle]
_broadcast_backward args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_broadcast_backward"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_broadcast_backward_upd ::
                        forall args . Fullfilled "_broadcast_backward(ndarray)" args =>
                          [NDArrayHandle] ->
                            ArgsHMap "_broadcast_backward(ndarray)" args -> IO ()
_broadcast_backward_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_broadcast_backward"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "norm(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

norm ::
     forall args . Fullfilled "norm(ndarray)" args =>
       ArgsHMap "norm(ndarray)" args -> IO [NDArrayHandle]
norm args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "norm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

norm_upd ::
         forall args . Fullfilled "norm(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "norm(ndarray)" args -> IO ()
norm_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "norm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "elemwise_add(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

elemwise_add ::
             forall args . Fullfilled "elemwise_add(ndarray)" args =>
               ArgsHMap "elemwise_add(ndarray)" args -> IO [NDArrayHandle]
elemwise_add args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "elemwise_add"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

elemwise_add_upd ::
                 forall args . Fullfilled "elemwise_add(ndarray)" args =>
                   [NDArrayHandle] -> ArgsHMap "elemwise_add(ndarray)" args -> IO ()
elemwise_add_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "elemwise_add"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_grad_add(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_grad_add ::
          forall args . Fullfilled "_grad_add(ndarray)" args =>
            ArgsHMap "_grad_add(ndarray)" args -> IO [NDArrayHandle]
_grad_add args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_grad_add"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_grad_add_upd ::
              forall args . Fullfilled "_grad_add(ndarray)" args =>
                [NDArrayHandle] -> ArgsHMap "_grad_add(ndarray)" args -> IO ()
_grad_add_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_grad_add"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_add(ndarray)" = '[]

_backward_add ::
              forall args . Fullfilled "_backward_add(ndarray)" args =>
                ArgsHMap "_backward_add(ndarray)" args -> IO [NDArrayHandle]
_backward_add args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_add"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_add_upd ::
                  forall args . Fullfilled "_backward_add(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_backward_add(ndarray)" args -> IO ()
_backward_add_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_add"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "elemwise_sub(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

elemwise_sub ::
             forall args . Fullfilled "elemwise_sub(ndarray)" args =>
               ArgsHMap "elemwise_sub(ndarray)" args -> IO [NDArrayHandle]
elemwise_sub args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "elemwise_sub"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

elemwise_sub_upd ::
                 forall args . Fullfilled "elemwise_sub(ndarray)" args =>
                   [NDArrayHandle] -> ArgsHMap "elemwise_sub(ndarray)" args -> IO ()
elemwise_sub_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "elemwise_sub"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_sub(ndarray)" = '[]

_backward_sub ::
              forall args . Fullfilled "_backward_sub(ndarray)" args =>
                ArgsHMap "_backward_sub(ndarray)" args -> IO [NDArrayHandle]
_backward_sub args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sub"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_sub_upd ::
                  forall args . Fullfilled "_backward_sub(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_backward_sub(ndarray)" args -> IO ()
_backward_sub_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sub"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "elemwise_mul(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

elemwise_mul ::
             forall args . Fullfilled "elemwise_mul(ndarray)" args =>
               ArgsHMap "elemwise_mul(ndarray)" args -> IO [NDArrayHandle]
elemwise_mul args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "elemwise_mul"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

elemwise_mul_upd ::
                 forall args . Fullfilled "elemwise_mul(ndarray)" args =>
                   [NDArrayHandle] -> ArgsHMap "elemwise_mul(ndarray)" args -> IO ()
elemwise_mul_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "elemwise_mul"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_mul(ndarray)" = '[]

_backward_mul ::
              forall args . Fullfilled "_backward_mul(ndarray)" args =>
                ArgsHMap "_backward_mul(ndarray)" args -> IO [NDArrayHandle]
_backward_mul args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_mul"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_mul_upd ::
                  forall args . Fullfilled "_backward_mul(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_backward_mul(ndarray)" args -> IO ()
_backward_mul_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_mul"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "elemwise_div(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

elemwise_div ::
             forall args . Fullfilled "elemwise_div(ndarray)" args =>
               ArgsHMap "elemwise_div(ndarray)" args -> IO [NDArrayHandle]
elemwise_div args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "elemwise_div"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

elemwise_div_upd ::
                 forall args . Fullfilled "elemwise_div(ndarray)" args =>
                   [NDArrayHandle] -> ArgsHMap "elemwise_div(ndarray)" args -> IO ()
elemwise_div_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "elemwise_div"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_div(ndarray)" = '[]

_backward_div ::
              forall args . Fullfilled "_backward_div(ndarray)" args =>
                ArgsHMap "_backward_div(ndarray)" args -> IO [NDArrayHandle]
_backward_div args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_div"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_div_upd ::
                  forall args . Fullfilled "_backward_div(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_backward_div(ndarray)" args -> IO ()
_backward_div_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_div"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_mod(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_mod ::
     forall args . Fullfilled "_mod(ndarray)" args =>
       ArgsHMap "_mod(ndarray)" args -> IO [NDArrayHandle]
_mod args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_mod"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_mod_upd ::
         forall args . Fullfilled "_mod(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "_mod(ndarray)" args -> IO ()
_mod_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_mod"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_mod(ndarray)" = '[]

_backward_mod ::
              forall args . Fullfilled "_backward_mod(ndarray)" args =>
                ArgsHMap "_backward_mod(ndarray)" args -> IO [NDArrayHandle]
_backward_mod args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_mod"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_mod_upd ::
                  forall args . Fullfilled "_backward_mod(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_backward_mod(ndarray)" args -> IO ()
_backward_mod_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_mod"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_plus_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_plus_scalar ::
             forall args . Fullfilled "_plus_scalar(ndarray)" args =>
               ArgsHMap "_plus_scalar(ndarray)" args -> IO [NDArrayHandle]
_plus_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_plus_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_plus_scalar_upd ::
                 forall args . Fullfilled "_plus_scalar(ndarray)" args =>
                   [NDArrayHandle] -> ArgsHMap "_plus_scalar(ndarray)" args -> IO ()
_plus_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_plus_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_minus_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_minus_scalar ::
              forall args . Fullfilled "_minus_scalar(ndarray)" args =>
                ArgsHMap "_minus_scalar(ndarray)" args -> IO [NDArrayHandle]
_minus_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_minus_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_minus_scalar_upd ::
                  forall args . Fullfilled "_minus_scalar(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_minus_scalar(ndarray)" args -> IO ()
_minus_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_minus_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_rminus_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_rminus_scalar ::
               forall args . Fullfilled "_rminus_scalar(ndarray)" args =>
                 ArgsHMap "_rminus_scalar(ndarray)" args -> IO [NDArrayHandle]
_rminus_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_rminus_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_rminus_scalar_upd ::
                   forall args . Fullfilled "_rminus_scalar(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_rminus_scalar(ndarray)" args -> IO ()
_rminus_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_rminus_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_mul_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_mul_scalar ::
            forall args . Fullfilled "_mul_scalar(ndarray)" args =>
              ArgsHMap "_mul_scalar(ndarray)" args -> IO [NDArrayHandle]
_mul_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_mul_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_mul_scalar_upd ::
                forall args . Fullfilled "_mul_scalar(ndarray)" args =>
                  [NDArrayHandle] -> ArgsHMap "_mul_scalar(ndarray)" args -> IO ()
_mul_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_mul_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_mul_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_backward_mul_scalar ::
                     forall args . Fullfilled "_backward_mul_scalar(ndarray)" args =>
                       ArgsHMap "_backward_mul_scalar(ndarray)" args -> IO [NDArrayHandle]
_backward_mul_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_mul_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_mul_scalar_upd ::
                         forall args . Fullfilled "_backward_mul_scalar(ndarray)" args =>
                           [NDArrayHandle] ->
                             ArgsHMap "_backward_mul_scalar(ndarray)" args -> IO ()
_backward_mul_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_mul_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_div_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_div_scalar ::
            forall args . Fullfilled "_div_scalar(ndarray)" args =>
              ArgsHMap "_div_scalar(ndarray)" args -> IO [NDArrayHandle]
_div_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_div_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_div_scalar_upd ::
                forall args . Fullfilled "_div_scalar(ndarray)" args =>
                  [NDArrayHandle] -> ArgsHMap "_div_scalar(ndarray)" args -> IO ()
_div_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_div_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_div_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_backward_div_scalar ::
                     forall args . Fullfilled "_backward_div_scalar(ndarray)" args =>
                       ArgsHMap "_backward_div_scalar(ndarray)" args -> IO [NDArrayHandle]
_backward_div_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_div_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_div_scalar_upd ::
                         forall args . Fullfilled "_backward_div_scalar(ndarray)" args =>
                           [NDArrayHandle] ->
                             ArgsHMap "_backward_div_scalar(ndarray)" args -> IO ()
_backward_div_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_div_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_rdiv_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_rdiv_scalar ::
             forall args . Fullfilled "_rdiv_scalar(ndarray)" args =>
               ArgsHMap "_rdiv_scalar(ndarray)" args -> IO [NDArrayHandle]
_rdiv_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_rdiv_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_rdiv_scalar_upd ::
                 forall args . Fullfilled "_rdiv_scalar(ndarray)" args =>
                   [NDArrayHandle] -> ArgsHMap "_rdiv_scalar(ndarray)" args -> IO ()
_rdiv_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_rdiv_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_rdiv_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_rdiv_scalar ::
                      forall args . Fullfilled "_backward_rdiv_scalar(ndarray)" args =>
                        ArgsHMap "_backward_rdiv_scalar(ndarray)" args ->
                          IO [NDArrayHandle]
_backward_rdiv_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_rdiv_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_rdiv_scalar_upd ::
                          forall args . Fullfilled "_backward_rdiv_scalar(ndarray)" args =>
                            [NDArrayHandle] ->
                              ArgsHMap "_backward_rdiv_scalar(ndarray)" args -> IO ()
_backward_rdiv_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_rdiv_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_mod_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_mod_scalar ::
            forall args . Fullfilled "_mod_scalar(ndarray)" args =>
              ArgsHMap "_mod_scalar(ndarray)" args -> IO [NDArrayHandle]
_mod_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_mod_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_mod_scalar_upd ::
                forall args . Fullfilled "_mod_scalar(ndarray)" args =>
                  [NDArrayHandle] -> ArgsHMap "_mod_scalar(ndarray)" args -> IO ()
_mod_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_mod_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_mod_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_mod_scalar ::
                     forall args . Fullfilled "_backward_mod_scalar(ndarray)" args =>
                       ArgsHMap "_backward_mod_scalar(ndarray)" args -> IO [NDArrayHandle]
_backward_mod_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_mod_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_mod_scalar_upd ::
                         forall args . Fullfilled "_backward_mod_scalar(ndarray)" args =>
                           [NDArrayHandle] ->
                             ArgsHMap "_backward_mod_scalar(ndarray)" args -> IO ()
_backward_mod_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_mod_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_rmod_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_rmod_scalar ::
             forall args . Fullfilled "_rmod_scalar(ndarray)" args =>
               ArgsHMap "_rmod_scalar(ndarray)" args -> IO [NDArrayHandle]
_rmod_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_rmod_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_rmod_scalar_upd ::
                 forall args . Fullfilled "_rmod_scalar(ndarray)" args =>
                   [NDArrayHandle] -> ArgsHMap "_rmod_scalar(ndarray)" args -> IO ()
_rmod_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_rmod_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_rmod_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_rmod_scalar ::
                      forall args . Fullfilled "_backward_rmod_scalar(ndarray)" args =>
                        ArgsHMap "_backward_rmod_scalar(ndarray)" args ->
                          IO [NDArrayHandle]
_backward_rmod_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_rmod_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_rmod_scalar_upd ::
                          forall args . Fullfilled "_backward_rmod_scalar(ndarray)" args =>
                            [NDArrayHandle] ->
                              ArgsHMap "_backward_rmod_scalar(ndarray)" args -> IO ()
_backward_rmod_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_rmod_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_scatter_elemwise_div(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_scatter_elemwise_div ::
                      forall args . Fullfilled "_scatter_elemwise_div(ndarray)" args =>
                        ArgsHMap "_scatter_elemwise_div(ndarray)" args ->
                          IO [NDArrayHandle]
_scatter_elemwise_div args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_scatter_elemwise_div"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_scatter_elemwise_div_upd ::
                          forall args . Fullfilled "_scatter_elemwise_div(ndarray)" args =>
                            [NDArrayHandle] ->
                              ArgsHMap "_scatter_elemwise_div(ndarray)" args -> IO ()
_scatter_elemwise_div_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_scatter_elemwise_div"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_scatter_plus_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_scatter_plus_scalar ::
                     forall args . Fullfilled "_scatter_plus_scalar(ndarray)" args =>
                       ArgsHMap "_scatter_plus_scalar(ndarray)" args -> IO [NDArrayHandle]
_scatter_plus_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_scatter_plus_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_scatter_plus_scalar_upd ::
                         forall args . Fullfilled "_scatter_plus_scalar(ndarray)" args =>
                           [NDArrayHandle] ->
                             ArgsHMap "_scatter_plus_scalar(ndarray)" args -> IO ()
_scatter_plus_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_scatter_plus_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_scatter_minus_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_scatter_minus_scalar ::
                      forall args . Fullfilled "_scatter_minus_scalar(ndarray)" args =>
                        ArgsHMap "_scatter_minus_scalar(ndarray)" args ->
                          IO [NDArrayHandle]
_scatter_minus_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_scatter_minus_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_scatter_minus_scalar_upd ::
                          forall args . Fullfilled "_scatter_minus_scalar(ndarray)" args =>
                            [NDArrayHandle] ->
                              ArgsHMap "_scatter_minus_scalar(ndarray)" args -> IO ()
_scatter_minus_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_scatter_minus_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_Embedding(ndarray)" =
     '[ '("input_dim", AttrReq Int), '("output_dim", AttrReq Int),
        '("dtype",
          AttrOpt
            (EnumType '["float16", "float32", "float64", "int32", "uint8"])),
        '("data", AttrOpt NDArrayHandle),
        '("weight", AttrOpt NDArrayHandle)]

_Embedding ::
           forall args . Fullfilled "_Embedding(ndarray)" args =>
             ArgsHMap "_Embedding(ndarray)" args -> IO [NDArrayHandle]
_Embedding args
  = let scalarArgs
          = catMaybes
              [("input_dim",) . showValue <$> (args !? #input_dim :: Maybe Int),
               ("output_dim",) . showValue <$> (args !? #output_dim :: Maybe Int),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType '["float16", "float32", "float64", "int32", "uint8"]))]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("weight",) <$> (args !? #weight :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Embedding"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_Embedding_upd ::
               forall args . Fullfilled "_Embedding(ndarray)" args =>
                 [NDArrayHandle] -> ArgsHMap "_Embedding(ndarray)" args -> IO ()
_Embedding_upd outputs args
  = let scalarArgs
          = catMaybes
              [("input_dim",) . showValue <$> (args !? #input_dim :: Maybe Int),
               ("output_dim",) . showValue <$> (args !? #output_dim :: Maybe Int),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType '["float16", "float32", "float64", "int32", "uint8"]))]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("weight",) <$> (args !? #weight :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Embedding"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_contrib_SparseEmbedding(ndarray)" =
     '[ '("input_dim", AttrReq Int), '("output_dim", AttrReq Int),
        '("dtype",
          AttrOpt
            (EnumType '["float16", "float32", "float64", "int32", "uint8"])),
        '("data", AttrOpt NDArrayHandle),
        '("weight", AttrOpt NDArrayHandle)]

_contrib_SparseEmbedding ::
                         forall args .
                           Fullfilled "_contrib_SparseEmbedding(ndarray)" args =>
                           ArgsHMap "_contrib_SparseEmbedding(ndarray)" args ->
                             IO [NDArrayHandle]
_contrib_SparseEmbedding args
  = let scalarArgs
          = catMaybes
              [("input_dim",) . showValue <$> (args !? #input_dim :: Maybe Int),
               ("output_dim",) . showValue <$> (args !? #output_dim :: Maybe Int),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType '["float16", "float32", "float64", "int32", "uint8"]))]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("weight",) <$> (args !? #weight :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_SparseEmbedding"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_contrib_SparseEmbedding_upd ::
                             forall args .
                               Fullfilled "_contrib_SparseEmbedding(ndarray)" args =>
                               [NDArrayHandle] ->
                                 ArgsHMap "_contrib_SparseEmbedding(ndarray)" args -> IO ()
_contrib_SparseEmbedding_upd outputs args
  = let scalarArgs
          = catMaybes
              [("input_dim",) . showValue <$> (args !? #input_dim :: Maybe Int),
               ("output_dim",) . showValue <$> (args !? #output_dim :: Maybe Int),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType '["float16", "float32", "float64", "int32", "uint8"]))]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("weight",) <$> (args !? #weight :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_SparseEmbedding"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_Embedding(ndarray)" = '[]

_backward_Embedding ::
                    forall args . Fullfilled "_backward_Embedding(ndarray)" args =>
                      ArgsHMap "_backward_Embedding(ndarray)" args -> IO [NDArrayHandle]
_backward_Embedding args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Embedding"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_Embedding_upd ::
                        forall args . Fullfilled "_backward_Embedding(ndarray)" args =>
                          [NDArrayHandle] ->
                            ArgsHMap "_backward_Embedding(ndarray)" args -> IO ()
_backward_Embedding_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Embedding"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_SparseEmbedding(ndarray)" =
     '[]

_backward_SparseEmbedding ::
                          forall args .
                            Fullfilled "_backward_SparseEmbedding(ndarray)" args =>
                            ArgsHMap "_backward_SparseEmbedding(ndarray)" args ->
                              IO [NDArrayHandle]
_backward_SparseEmbedding args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SparseEmbedding"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_SparseEmbedding_upd ::
                              forall args .
                                Fullfilled "_backward_SparseEmbedding(ndarray)" args =>
                                [NDArrayHandle] ->
                                  ArgsHMap "_backward_SparseEmbedding(ndarray)" args -> IO ()
_backward_SparseEmbedding_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SparseEmbedding"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "take(ndarray)" =
     '[ '("axis", AttrOpt Int),
        '("mode", AttrOpt (EnumType '["clip", "raise", "wrap"])),
        '("a", AttrOpt NDArrayHandle), '("indices", AttrOpt NDArrayHandle)]

take ::
     forall args . Fullfilled "take(ndarray)" args =>
       ArgsHMap "take(ndarray)" args -> IO [NDArrayHandle]
take args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("mode",) . showValue <$>
                 (args !? #mode :: Maybe (EnumType '["clip", "raise", "wrap"]))]
        tensorArgs
          = catMaybes
              [("a",) <$> (args !? #a :: Maybe NDArrayHandle),
               ("indices",) <$> (args !? #indices :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "take"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

take_upd ::
         forall args . Fullfilled "take(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "take(ndarray)" args -> IO ()
take_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("mode",) . showValue <$>
                 (args !? #mode :: Maybe (EnumType '["clip", "raise", "wrap"]))]
        tensorArgs
          = catMaybes
              [("a",) <$> (args !? #a :: Maybe NDArrayHandle),
               ("indices",) <$> (args !? #indices :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "take"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_take(ndarray)" = '[]

_backward_take ::
               forall args . Fullfilled "_backward_take(ndarray)" args =>
                 ArgsHMap "_backward_take(ndarray)" args -> IO [NDArrayHandle]
_backward_take args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_take"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_take_upd ::
                   forall args . Fullfilled "_backward_take(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_backward_take(ndarray)" args -> IO ()
_backward_take_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_take"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "batch_take(ndarray)" =
     '[ '("a", AttrOpt NDArrayHandle),
        '("indices", AttrOpt NDArrayHandle)]

batch_take ::
           forall args . Fullfilled "batch_take(ndarray)" args =>
             ArgsHMap "batch_take(ndarray)" args -> IO [NDArrayHandle]
batch_take args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("a",) <$> (args !? #a :: Maybe NDArrayHandle),
               ("indices",) <$> (args !? #indices :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "batch_take"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

batch_take_upd ::
               forall args . Fullfilled "batch_take(ndarray)" args =>
                 [NDArrayHandle] -> ArgsHMap "batch_take(ndarray)" args -> IO ()
batch_take_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("a",) <$> (args !? #a :: Maybe NDArrayHandle),
               ("indices",) <$> (args !? #indices :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "batch_take"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "one_hot(ndarray)" =
     '[ '("depth", AttrReq Int), '("on_value", AttrOpt Double),
        '("off_value", AttrOpt Double),
        '("dtype",
          AttrOpt
            (EnumType '["float16", "float32", "float64", "int32", "uint8"])),
        '("indices", AttrOpt NDArrayHandle)]

one_hot ::
        forall args . Fullfilled "one_hot(ndarray)" args =>
          ArgsHMap "one_hot(ndarray)" args -> IO [NDArrayHandle]
one_hot args
  = let scalarArgs
          = catMaybes
              [("depth",) . showValue <$> (args !? #depth :: Maybe Int),
               ("on_value",) . showValue <$> (args !? #on_value :: Maybe Double),
               ("off_value",) . showValue <$>
                 (args !? #off_value :: Maybe Double),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType '["float16", "float32", "float64", "int32", "uint8"]))]
        tensorArgs
          = catMaybes
              [("indices",) <$> (args !? #indices :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "one_hot"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

one_hot_upd ::
            forall args . Fullfilled "one_hot(ndarray)" args =>
              [NDArrayHandle] -> ArgsHMap "one_hot(ndarray)" args -> IO ()
one_hot_upd outputs args
  = let scalarArgs
          = catMaybes
              [("depth",) . showValue <$> (args !? #depth :: Maybe Int),
               ("on_value",) . showValue <$> (args !? #on_value :: Maybe Double),
               ("off_value",) . showValue <$>
                 (args !? #off_value :: Maybe Double),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType '["float16", "float32", "float64", "int32", "uint8"]))]
        tensorArgs
          = catMaybes
              [("indices",) <$> (args !? #indices :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "one_hot"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "gather_nd(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle),
        '("indices", AttrOpt NDArrayHandle)]

gather_nd ::
          forall args . Fullfilled "gather_nd(ndarray)" args =>
            ArgsHMap "gather_nd(ndarray)" args -> IO [NDArrayHandle]
gather_nd args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("indices",) <$> (args !? #indices :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "gather_nd"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

gather_nd_upd ::
              forall args . Fullfilled "gather_nd(ndarray)" args =>
                [NDArrayHandle] -> ArgsHMap "gather_nd(ndarray)" args -> IO ()
gather_nd_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("indices",) <$> (args !? #indices :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "gather_nd"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "scatter_nd(ndarray)" =
     '[ '("shape", AttrReq [Int]), '("data", AttrOpt NDArrayHandle),
        '("indices", AttrOpt NDArrayHandle)]

scatter_nd ::
           forall args . Fullfilled "scatter_nd(ndarray)" args =>
             ArgsHMap "scatter_nd(ndarray)" args -> IO [NDArrayHandle]
scatter_nd args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("indices",) <$> (args !? #indices :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "scatter_nd"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

scatter_nd_upd ::
               forall args . Fullfilled "scatter_nd(ndarray)" args =>
                 [NDArrayHandle] -> ArgsHMap "scatter_nd(ndarray)" args -> IO ()
scatter_nd_upd outputs args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("indices",) <$> (args !? #indices :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "scatter_nd"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_scatter_set_nd(ndarray)" =
     '[ '("shape", AttrReq [Int]), '("data", AttrOpt NDArrayHandle),
        '("indices", AttrOpt NDArrayHandle)]

_scatter_set_nd ::
                forall args . Fullfilled "_scatter_set_nd(ndarray)" args =>
                  ArgsHMap "_scatter_set_nd(ndarray)" args -> IO [NDArrayHandle]
_scatter_set_nd args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("indices",) <$> (args !? #indices :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_scatter_set_nd"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_scatter_set_nd_upd ::
                    forall args . Fullfilled "_scatter_set_nd(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_scatter_set_nd(ndarray)" args -> IO ()
_scatter_set_nd_upd outputs args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("indices",) <$> (args !? #indices :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_scatter_set_nd"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "broadcast_equal(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

broadcast_equal ::
                forall args . Fullfilled "broadcast_equal(ndarray)" args =>
                  ArgsHMap "broadcast_equal(ndarray)" args -> IO [NDArrayHandle]
broadcast_equal args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_equal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

broadcast_equal_upd ::
                    forall args . Fullfilled "broadcast_equal(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "broadcast_equal(ndarray)" args -> IO ()
broadcast_equal_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_equal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "broadcast_not_equal(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

broadcast_not_equal ::
                    forall args . Fullfilled "broadcast_not_equal(ndarray)" args =>
                      ArgsHMap "broadcast_not_equal(ndarray)" args -> IO [NDArrayHandle]
broadcast_not_equal args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_not_equal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

broadcast_not_equal_upd ::
                        forall args . Fullfilled "broadcast_not_equal(ndarray)" args =>
                          [NDArrayHandle] ->
                            ArgsHMap "broadcast_not_equal(ndarray)" args -> IO ()
broadcast_not_equal_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_not_equal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "broadcast_greater(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

broadcast_greater ::
                  forall args . Fullfilled "broadcast_greater(ndarray)" args =>
                    ArgsHMap "broadcast_greater(ndarray)" args -> IO [NDArrayHandle]
broadcast_greater args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_greater"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

broadcast_greater_upd ::
                      forall args . Fullfilled "broadcast_greater(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "broadcast_greater(ndarray)" args -> IO ()
broadcast_greater_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_greater"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "broadcast_greater_equal(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

broadcast_greater_equal ::
                        forall args . Fullfilled "broadcast_greater_equal(ndarray)" args =>
                          ArgsHMap "broadcast_greater_equal(ndarray)" args ->
                            IO [NDArrayHandle]
broadcast_greater_equal args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_greater_equal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

broadcast_greater_equal_upd ::
                            forall args . Fullfilled "broadcast_greater_equal(ndarray)" args =>
                              [NDArrayHandle] ->
                                ArgsHMap "broadcast_greater_equal(ndarray)" args -> IO ()
broadcast_greater_equal_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_greater_equal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "broadcast_lesser(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

broadcast_lesser ::
                 forall args . Fullfilled "broadcast_lesser(ndarray)" args =>
                   ArgsHMap "broadcast_lesser(ndarray)" args -> IO [NDArrayHandle]
broadcast_lesser args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_lesser"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

broadcast_lesser_upd ::
                     forall args . Fullfilled "broadcast_lesser(ndarray)" args =>
                       [NDArrayHandle] ->
                         ArgsHMap "broadcast_lesser(ndarray)" args -> IO ()
broadcast_lesser_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_lesser"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "broadcast_lesser_equal(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

broadcast_lesser_equal ::
                       forall args . Fullfilled "broadcast_lesser_equal(ndarray)" args =>
                         ArgsHMap "broadcast_lesser_equal(ndarray)" args ->
                           IO [NDArrayHandle]
broadcast_lesser_equal args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_lesser_equal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

broadcast_lesser_equal_upd ::
                           forall args . Fullfilled "broadcast_lesser_equal(ndarray)" args =>
                             [NDArrayHandle] ->
                               ArgsHMap "broadcast_lesser_equal(ndarray)" args -> IO ()
broadcast_lesser_equal_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_lesser_equal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "argmax(ndarray)" =
     '[ '("axis", AttrOpt (Maybe Int)), '("keepdims", AttrOpt Bool),
        '("data", AttrOpt NDArrayHandle)]

argmax ::
       forall args . Fullfilled "argmax(ndarray)" args =>
         ArgsHMap "argmax(ndarray)" args -> IO [NDArrayHandle]
argmax args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "argmax"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

argmax_upd ::
           forall args . Fullfilled "argmax(ndarray)" args =>
             [NDArrayHandle] -> ArgsHMap "argmax(ndarray)" args -> IO ()
argmax_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "argmax"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "argmin(ndarray)" =
     '[ '("axis", AttrOpt (Maybe Int)), '("keepdims", AttrOpt Bool),
        '("data", AttrOpt NDArrayHandle)]

argmin ::
       forall args . Fullfilled "argmin(ndarray)" args =>
         ArgsHMap "argmin(ndarray)" args -> IO [NDArrayHandle]
argmin args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "argmin"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

argmin_upd ::
           forall args . Fullfilled "argmin(ndarray)" args =>
             [NDArrayHandle] -> ArgsHMap "argmin(ndarray)" args -> IO ()
argmin_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "argmin"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "argmax_channel(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

argmax_channel ::
               forall args . Fullfilled "argmax_channel(ndarray)" args =>
                 ArgsHMap "argmax_channel(ndarray)" args -> IO [NDArrayHandle]
argmax_channel args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "argmax_channel"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

argmax_channel_upd ::
                   forall args . Fullfilled "argmax_channel(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "argmax_channel(ndarray)" args -> IO ()
argmax_channel_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "argmax_channel"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "pick(ndarray)" =
     '[ '("axis", AttrOpt (Maybe Int)), '("keepdims", AttrOpt Bool),
        '("data", AttrOpt NDArrayHandle),
        '("index", AttrOpt NDArrayHandle)]

pick ::
     forall args . Fullfilled "pick(ndarray)" args =>
       ArgsHMap "pick(ndarray)" args -> IO [NDArrayHandle]
pick args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("index",) <$> (args !? #index :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "pick"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

pick_upd ::
         forall args . Fullfilled "pick(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "pick(ndarray)" args -> IO ()
pick_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("index",) <$> (args !? #index :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "pick"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_pick(ndarray)" = '[]

_backward_pick ::
               forall args . Fullfilled "_backward_pick(ndarray)" args =>
                 ArgsHMap "_backward_pick(ndarray)" args -> IO [NDArrayHandle]
_backward_pick args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_pick"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_pick_upd ::
                   forall args . Fullfilled "_backward_pick(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_backward_pick(ndarray)" args -> IO ()
_backward_pick_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_pick"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_sparse_retain(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle),
        '("indices", AttrOpt NDArrayHandle)]

_sparse_retain ::
               forall args . Fullfilled "_sparse_retain(ndarray)" args =>
                 ArgsHMap "_sparse_retain(ndarray)" args -> IO [NDArrayHandle]
_sparse_retain args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("indices",) <$> (args !? #indices :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sparse_retain"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_sparse_retain_upd ::
                   forall args . Fullfilled "_sparse_retain(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_sparse_retain(ndarray)" args -> IO ()
_sparse_retain_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("indices",) <$> (args !? #indices :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sparse_retain"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_sparse_retain(ndarray)" =
     '[]

_backward_sparse_retain ::
                        forall args . Fullfilled "_backward_sparse_retain(ndarray)" args =>
                          ArgsHMap "_backward_sparse_retain(ndarray)" args ->
                            IO [NDArrayHandle]
_backward_sparse_retain args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sparse_retain"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_sparse_retain_upd ::
                            forall args . Fullfilled "_backward_sparse_retain(ndarray)" args =>
                              [NDArrayHandle] ->
                                ArgsHMap "_backward_sparse_retain(ndarray)" args -> IO ()
_backward_sparse_retain_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sparse_retain"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_where(ndarray)" =
     '[ '("condition", AttrOpt NDArrayHandle),
        '("x", AttrOpt NDArrayHandle), '("y", AttrOpt NDArrayHandle)]

_where ::
       forall args . Fullfilled "_where(ndarray)" args =>
         ArgsHMap "_where(ndarray)" args -> IO [NDArrayHandle]
_where args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("condition",) <$> (args !? #condition :: Maybe NDArrayHandle),
               ("x",) <$> (args !? #x :: Maybe NDArrayHandle),
               ("y",) <$> (args !? #y :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "where"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_where_upd ::
           forall args . Fullfilled "_where(ndarray)" args =>
             [NDArrayHandle] -> ArgsHMap "_where(ndarray)" args -> IO ()
_where_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("condition",) <$> (args !? #condition :: Maybe NDArrayHandle),
               ("x",) <$> (args !? #x :: Maybe NDArrayHandle),
               ("y",) <$> (args !? #y :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "where"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_where(ndarray)" = '[]

_backward_where ::
                forall args . Fullfilled "_backward_where(ndarray)" args =>
                  ArgsHMap "_backward_where(ndarray)" args -> IO [NDArrayHandle]
_backward_where args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_where"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_where_upd ::
                    forall args . Fullfilled "_backward_where(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_backward_where(ndarray)" args -> IO ()
_backward_where_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_where"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_maximum_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_maximum_scalar ::
                forall args . Fullfilled "_maximum_scalar(ndarray)" args =>
                  ArgsHMap "_maximum_scalar(ndarray)" args -> IO [NDArrayHandle]
_maximum_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_maximum_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_maximum_scalar_upd ::
                    forall args . Fullfilled "_maximum_scalar(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_maximum_scalar(ndarray)" args -> IO ()
_maximum_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_maximum_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_maximum_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_maximum_scalar ::
                         forall args .
                           Fullfilled "_backward_maximum_scalar(ndarray)" args =>
                           ArgsHMap "_backward_maximum_scalar(ndarray)" args ->
                             IO [NDArrayHandle]
_backward_maximum_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_maximum_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_maximum_scalar_upd ::
                             forall args .
                               Fullfilled "_backward_maximum_scalar(ndarray)" args =>
                               [NDArrayHandle] ->
                                 ArgsHMap "_backward_maximum_scalar(ndarray)" args -> IO ()
_backward_maximum_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_maximum_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_minimum_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_minimum_scalar ::
                forall args . Fullfilled "_minimum_scalar(ndarray)" args =>
                  ArgsHMap "_minimum_scalar(ndarray)" args -> IO [NDArrayHandle]
_minimum_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_minimum_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_minimum_scalar_upd ::
                    forall args . Fullfilled "_minimum_scalar(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_minimum_scalar(ndarray)" args -> IO ()
_minimum_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_minimum_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_minimum_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_minimum_scalar ::
                         forall args .
                           Fullfilled "_backward_minimum_scalar(ndarray)" args =>
                           ArgsHMap "_backward_minimum_scalar(ndarray)" args ->
                             IO [NDArrayHandle]
_backward_minimum_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_minimum_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_minimum_scalar_upd ::
                             forall args .
                               Fullfilled "_backward_minimum_scalar(ndarray)" args =>
                               [NDArrayHandle] ->
                                 ArgsHMap "_backward_minimum_scalar(ndarray)" args -> IO ()
_backward_minimum_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_minimum_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_power_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_power_scalar ::
              forall args . Fullfilled "_power_scalar(ndarray)" args =>
                ArgsHMap "_power_scalar(ndarray)" args -> IO [NDArrayHandle]
_power_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_power_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_power_scalar_upd ::
                  forall args . Fullfilled "_power_scalar(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_power_scalar(ndarray)" args -> IO ()
_power_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_power_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_power_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_power_scalar ::
                       forall args . Fullfilled "_backward_power_scalar(ndarray)" args =>
                         ArgsHMap "_backward_power_scalar(ndarray)" args ->
                           IO [NDArrayHandle]
_backward_power_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_power_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_power_scalar_upd ::
                           forall args . Fullfilled "_backward_power_scalar(ndarray)" args =>
                             [NDArrayHandle] ->
                               ArgsHMap "_backward_power_scalar(ndarray)" args -> IO ()
_backward_power_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_power_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_rpower_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_rpower_scalar ::
               forall args . Fullfilled "_rpower_scalar(ndarray)" args =>
                 ArgsHMap "_rpower_scalar(ndarray)" args -> IO [NDArrayHandle]
_rpower_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_rpower_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_rpower_scalar_upd ::
                   forall args . Fullfilled "_rpower_scalar(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_rpower_scalar(ndarray)" args -> IO ()
_rpower_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_rpower_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_rpower_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_rpower_scalar ::
                        forall args . Fullfilled "_backward_rpower_scalar(ndarray)" args =>
                          ArgsHMap "_backward_rpower_scalar(ndarray)" args ->
                            IO [NDArrayHandle]
_backward_rpower_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_rpower_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_rpower_scalar_upd ::
                            forall args . Fullfilled "_backward_rpower_scalar(ndarray)" args =>
                              [NDArrayHandle] ->
                                ArgsHMap "_backward_rpower_scalar(ndarray)" args -> IO ()
_backward_rpower_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_rpower_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_hypot_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_hypot_scalar ::
              forall args . Fullfilled "_hypot_scalar(ndarray)" args =>
                ArgsHMap "_hypot_scalar(ndarray)" args -> IO [NDArrayHandle]
_hypot_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_hypot_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_hypot_scalar_upd ::
                  forall args . Fullfilled "_hypot_scalar(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_hypot_scalar(ndarray)" args -> IO ()
_hypot_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_hypot_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_hypot_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_hypot_scalar ::
                       forall args . Fullfilled "_backward_hypot_scalar(ndarray)" args =>
                         ArgsHMap "_backward_hypot_scalar(ndarray)" args ->
                           IO [NDArrayHandle]
_backward_hypot_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_hypot_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_hypot_scalar_upd ::
                           forall args . Fullfilled "_backward_hypot_scalar(ndarray)" args =>
                             [NDArrayHandle] ->
                               ArgsHMap "_backward_hypot_scalar(ndarray)" args -> IO ()
_backward_hypot_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_hypot_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "smooth_l1(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

smooth_l1 ::
          forall args . Fullfilled "smooth_l1(ndarray)" args =>
            ArgsHMap "smooth_l1(ndarray)" args -> IO [NDArrayHandle]
smooth_l1 args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "smooth_l1"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

smooth_l1_upd ::
              forall args . Fullfilled "smooth_l1(ndarray)" args =>
                [NDArrayHandle] -> ArgsHMap "smooth_l1(ndarray)" args -> IO ()
smooth_l1_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "smooth_l1"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_smooth_l1(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_smooth_l1 ::
                    forall args . Fullfilled "_backward_smooth_l1(ndarray)" args =>
                      ArgsHMap "_backward_smooth_l1(ndarray)" args -> IO [NDArrayHandle]
_backward_smooth_l1 args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_smooth_l1"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_smooth_l1_upd ::
                        forall args . Fullfilled "_backward_smooth_l1(ndarray)" args =>
                          [NDArrayHandle] ->
                            ArgsHMap "_backward_smooth_l1(ndarray)" args -> IO ()
_backward_smooth_l1_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_smooth_l1"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "topk(ndarray)" =
     '[ '("axis", AttrOpt (Maybe Int)), '("k", AttrOpt Int),
        '("ret_typ",
          AttrOpt (EnumType '["both", "indices", "mask", "value"])),
        '("is_ascend", AttrOpt Bool), '("data", AttrOpt NDArrayHandle)]

topk ::
     forall args . Fullfilled "topk(ndarray)" args =>
       ArgsHMap "topk(ndarray)" args -> IO [NDArrayHandle]
topk args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("k",) . showValue <$> (args !? #k :: Maybe Int),
               ("ret_typ",) . showValue <$>
                 (args !? #ret_typ ::
                    Maybe (EnumType '["both", "indices", "mask", "value"])),
               ("is_ascend",) . showValue <$> (args !? #is_ascend :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "topk"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

topk_upd ::
         forall args . Fullfilled "topk(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "topk(ndarray)" args -> IO ()
topk_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("k",) . showValue <$> (args !? #k :: Maybe Int),
               ("ret_typ",) . showValue <$>
                 (args !? #ret_typ ::
                    Maybe (EnumType '["both", "indices", "mask", "value"])),
               ("is_ascend",) . showValue <$> (args !? #is_ascend :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "topk"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_topk(ndarray)" = '[]

_backward_topk ::
               forall args . Fullfilled "_backward_topk(ndarray)" args =>
                 ArgsHMap "_backward_topk(ndarray)" args -> IO [NDArrayHandle]
_backward_topk args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_topk"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_topk_upd ::
                   forall args . Fullfilled "_backward_topk(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_backward_topk(ndarray)" args -> IO ()
_backward_topk_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_topk"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "sort(ndarray)" =
     '[ '("axis", AttrOpt (Maybe Int)), '("is_ascend", AttrOpt Bool),
        '("data", AttrOpt NDArrayHandle)]

sort ::
     forall args . Fullfilled "sort(ndarray)" args =>
       ArgsHMap "sort(ndarray)" args -> IO [NDArrayHandle]
sort args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("is_ascend",) . showValue <$> (args !? #is_ascend :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sort"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

sort_upd ::
         forall args . Fullfilled "sort(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "sort(ndarray)" args -> IO ()
sort_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("is_ascend",) . showValue <$> (args !? #is_ascend :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sort"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "argsort(ndarray)" =
     '[ '("axis", AttrOpt (Maybe Int)), '("is_ascend", AttrOpt Bool),
        '("data", AttrOpt NDArrayHandle)]

argsort ::
        forall args . Fullfilled "argsort(ndarray)" args =>
          ArgsHMap "argsort(ndarray)" args -> IO [NDArrayHandle]
argsort args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("is_ascend",) . showValue <$> (args !? #is_ascend :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "argsort"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

argsort_upd ::
            forall args . Fullfilled "argsort(ndarray)" args =>
              [NDArrayHandle] -> ArgsHMap "argsort(ndarray)" args -> IO ()
argsort_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("is_ascend",) . showValue <$> (args !? #is_ascend :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "argsort"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_Reshape(ndarray)" =
     '[ '("shape", AttrOpt [Int]), '("reverse", AttrOpt Bool),
        '("target_shape", AttrOpt [Int]), '("keep_highest", AttrOpt Bool),
        '("data", AttrOpt NDArrayHandle)]

_Reshape ::
         forall args . Fullfilled "_Reshape(ndarray)" args =>
           ArgsHMap "_Reshape(ndarray)" args -> IO [NDArrayHandle]
_Reshape args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("reverse",) . showValue <$> (args !? #reverse :: Maybe Bool),
               ("target_shape",) . showValue <$>
                 (args !? #target_shape :: Maybe [Int]),
               ("keep_highest",) . showValue <$>
                 (args !? #keep_highest :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Reshape"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_Reshape_upd ::
             forall args . Fullfilled "_Reshape(ndarray)" args =>
               [NDArrayHandle] -> ArgsHMap "_Reshape(ndarray)" args -> IO ()
_Reshape_upd outputs args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("reverse",) . showValue <$> (args !? #reverse :: Maybe Bool),
               ("target_shape",) . showValue <$>
                 (args !? #target_shape :: Maybe [Int]),
               ("keep_highest",) . showValue <$>
                 (args !? #keep_highest :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Reshape"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_Flatten(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

_Flatten ::
         forall args . Fullfilled "_Flatten(ndarray)" args =>
           ArgsHMap "_Flatten(ndarray)" args -> IO [NDArrayHandle]
_Flatten args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Flatten"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_Flatten_upd ::
             forall args . Fullfilled "_Flatten(ndarray)" args =>
               [NDArrayHandle] -> ArgsHMap "_Flatten(ndarray)" args -> IO ()
_Flatten_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Flatten"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "transpose(ndarray)" =
     '[ '("axes", AttrOpt [Int]), '("data", AttrOpt NDArrayHandle)]

transpose ::
          forall args . Fullfilled "transpose(ndarray)" args =>
            ArgsHMap "transpose(ndarray)" args -> IO [NDArrayHandle]
transpose args
  = let scalarArgs
          = catMaybes
              [("axes",) . showValue <$> (args !? #axes :: Maybe [Int])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "transpose"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

transpose_upd ::
              forall args . Fullfilled "transpose(ndarray)" args =>
                [NDArrayHandle] -> ArgsHMap "transpose(ndarray)" args -> IO ()
transpose_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axes",) . showValue <$> (args !? #axes :: Maybe [Int])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "transpose"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "expand_dims(ndarray)" =
     '[ '("axis", AttrReq Int), '("data", AttrOpt NDArrayHandle)]

expand_dims ::
            forall args . Fullfilled "expand_dims(ndarray)" args =>
              ArgsHMap "expand_dims(ndarray)" args -> IO [NDArrayHandle]
expand_dims args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "expand_dims"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

expand_dims_upd ::
                forall args . Fullfilled "expand_dims(ndarray)" args =>
                  [NDArrayHandle] -> ArgsHMap "expand_dims(ndarray)" args -> IO ()
expand_dims_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "expand_dims"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "slice(ndarray)" =
     '[ '("begin", AttrReq [Int]), '("end", AttrReq [Int]),
        '("step", AttrOpt [Int]), '("data", AttrOpt NDArrayHandle)]

slice ::
      forall args . Fullfilled "slice(ndarray)" args =>
        ArgsHMap "slice(ndarray)" args -> IO [NDArrayHandle]
slice args
  = let scalarArgs
          = catMaybes
              [("begin",) . showValue <$> (args !? #begin :: Maybe [Int]),
               ("end",) . showValue <$> (args !? #end :: Maybe [Int]),
               ("step",) . showValue <$> (args !? #step :: Maybe [Int])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "slice"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

slice_upd ::
          forall args . Fullfilled "slice(ndarray)" args =>
            [NDArrayHandle] -> ArgsHMap "slice(ndarray)" args -> IO ()
slice_upd outputs args
  = let scalarArgs
          = catMaybes
              [("begin",) . showValue <$> (args !? #begin :: Maybe [Int]),
               ("end",) . showValue <$> (args !? #end :: Maybe [Int]),
               ("step",) . showValue <$> (args !? #step :: Maybe [Int])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "slice"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_slice(ndarray)" = '[]

_backward_slice ::
                forall args . Fullfilled "_backward_slice(ndarray)" args =>
                  ArgsHMap "_backward_slice(ndarray)" args -> IO [NDArrayHandle]
_backward_slice args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_slice"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_slice_upd ::
                    forall args . Fullfilled "_backward_slice(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_backward_slice(ndarray)" args -> IO ()
_backward_slice_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_slice"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_slice_assign(ndarray)" =
     '[ '("begin", AttrReq [Int]), '("end", AttrReq [Int]),
        '("step", AttrOpt [Int]), '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_slice_assign ::
              forall args . Fullfilled "_slice_assign(ndarray)" args =>
                ArgsHMap "_slice_assign(ndarray)" args -> IO [NDArrayHandle]
_slice_assign args
  = let scalarArgs
          = catMaybes
              [("begin",) . showValue <$> (args !? #begin :: Maybe [Int]),
               ("end",) . showValue <$> (args !? #end :: Maybe [Int]),
               ("step",) . showValue <$> (args !? #step :: Maybe [Int])]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_slice_assign"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_slice_assign_upd ::
                  forall args . Fullfilled "_slice_assign(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_slice_assign(ndarray)" args -> IO ()
_slice_assign_upd outputs args
  = let scalarArgs
          = catMaybes
              [("begin",) . showValue <$> (args !? #begin :: Maybe [Int]),
               ("end",) . showValue <$> (args !? #end :: Maybe [Int]),
               ("step",) . showValue <$> (args !? #step :: Maybe [Int])]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_slice_assign"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_slice_assign_scalar(ndarray)" =
     '[ '("scalar", AttrOpt Float), '("begin", AttrReq [Int]),
        '("end", AttrReq [Int]), '("step", AttrOpt [Int]),
        '("data", AttrOpt NDArrayHandle)]

_slice_assign_scalar ::
                     forall args . Fullfilled "_slice_assign_scalar(ndarray)" args =>
                       ArgsHMap "_slice_assign_scalar(ndarray)" args -> IO [NDArrayHandle]
_slice_assign_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float),
               ("begin",) . showValue <$> (args !? #begin :: Maybe [Int]),
               ("end",) . showValue <$> (args !? #end :: Maybe [Int]),
               ("step",) . showValue <$> (args !? #step :: Maybe [Int])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_slice_assign_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_slice_assign_scalar_upd ::
                         forall args . Fullfilled "_slice_assign_scalar(ndarray)" args =>
                           [NDArrayHandle] ->
                             ArgsHMap "_slice_assign_scalar(ndarray)" args -> IO ()
_slice_assign_scalar_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float),
               ("begin",) . showValue <$> (args !? #begin :: Maybe [Int]),
               ("end",) . showValue <$> (args !? #end :: Maybe [Int]),
               ("step",) . showValue <$> (args !? #step :: Maybe [Int])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_slice_assign_scalar"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "slice_axis(ndarray)" =
     '[ '("axis", AttrReq Int), '("begin", AttrReq Int),
        '("end", AttrReq (Maybe Int)), '("data", AttrOpt NDArrayHandle)]

slice_axis ::
           forall args . Fullfilled "slice_axis(ndarray)" args =>
             ArgsHMap "slice_axis(ndarray)" args -> IO [NDArrayHandle]
slice_axis args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("begin",) . showValue <$> (args !? #begin :: Maybe Int),
               ("end",) . showValue <$> (args !? #end :: Maybe (Maybe Int))]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "slice_axis"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

slice_axis_upd ::
               forall args . Fullfilled "slice_axis(ndarray)" args =>
                 [NDArrayHandle] -> ArgsHMap "slice_axis(ndarray)" args -> IO ()
slice_axis_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("begin",) . showValue <$> (args !? #begin :: Maybe Int),
               ("end",) . showValue <$> (args !? #end :: Maybe (Maybe Int))]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "slice_axis"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_slice_axis(ndarray)" = '[]

_backward_slice_axis ::
                     forall args . Fullfilled "_backward_slice_axis(ndarray)" args =>
                       ArgsHMap "_backward_slice_axis(ndarray)" args -> IO [NDArrayHandle]
_backward_slice_axis args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_slice_axis"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_slice_axis_upd ::
                         forall args . Fullfilled "_backward_slice_axis(ndarray)" args =>
                           [NDArrayHandle] ->
                             ArgsHMap "_backward_slice_axis(ndarray)" args -> IO ()
_backward_slice_axis_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_slice_axis"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "clip(ndarray)" =
     '[ '("a_min", AttrReq Float), '("a_max", AttrReq Float),
        '("data", AttrOpt NDArrayHandle)]

clip ::
     forall args . Fullfilled "clip(ndarray)" args =>
       ArgsHMap "clip(ndarray)" args -> IO [NDArrayHandle]
clip args
  = let scalarArgs
          = catMaybes
              [("a_min",) . showValue <$> (args !? #a_min :: Maybe Float),
               ("a_max",) . showValue <$> (args !? #a_max :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "clip"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

clip_upd ::
         forall args . Fullfilled "clip(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "clip(ndarray)" args -> IO ()
clip_upd outputs args
  = let scalarArgs
          = catMaybes
              [("a_min",) . showValue <$> (args !? #a_min :: Maybe Float),
               ("a_max",) . showValue <$> (args !? #a_max :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "clip"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_clip(ndarray)" = '[]

_backward_clip ::
               forall args . Fullfilled "_backward_clip(ndarray)" args =>
                 ArgsHMap "_backward_clip(ndarray)" args -> IO [NDArrayHandle]
_backward_clip args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_clip"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_clip_upd ::
                   forall args . Fullfilled "_backward_clip(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_backward_clip(ndarray)" args -> IO ()
_backward_clip_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_clip"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "repeat(ndarray)" =
     '[ '("repeats", AttrReq Int), '("axis", AttrOpt (Maybe Int)),
        '("data", AttrOpt NDArrayHandle)]

repeat ::
       forall args . Fullfilled "repeat(ndarray)" args =>
         ArgsHMap "repeat(ndarray)" args -> IO [NDArrayHandle]
repeat args
  = let scalarArgs
          = catMaybes
              [("repeats",) . showValue <$> (args !? #repeats :: Maybe Int),
               ("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int))]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "repeat"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

repeat_upd ::
           forall args . Fullfilled "repeat(ndarray)" args =>
             [NDArrayHandle] -> ArgsHMap "repeat(ndarray)" args -> IO ()
repeat_upd outputs args
  = let scalarArgs
          = catMaybes
              [("repeats",) . showValue <$> (args !? #repeats :: Maybe Int),
               ("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int))]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "repeat"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_repeat(ndarray)" = '[]

_backward_repeat ::
                 forall args . Fullfilled "_backward_repeat(ndarray)" args =>
                   ArgsHMap "_backward_repeat(ndarray)" args -> IO [NDArrayHandle]
_backward_repeat args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_repeat"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_repeat_upd ::
                     forall args . Fullfilled "_backward_repeat(ndarray)" args =>
                       [NDArrayHandle] ->
                         ArgsHMap "_backward_repeat(ndarray)" args -> IO ()
_backward_repeat_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_repeat"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "tile(ndarray)" =
     '[ '("reps", AttrReq [Int]), '("data", AttrOpt NDArrayHandle)]

tile ::
     forall args . Fullfilled "tile(ndarray)" args =>
       ArgsHMap "tile(ndarray)" args -> IO [NDArrayHandle]
tile args
  = let scalarArgs
          = catMaybes
              [("reps",) . showValue <$> (args !? #reps :: Maybe [Int])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "tile"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

tile_upd ::
         forall args . Fullfilled "tile(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "tile(ndarray)" args -> IO ()
tile_upd outputs args
  = let scalarArgs
          = catMaybes
              [("reps",) . showValue <$> (args !? #reps :: Maybe [Int])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "tile"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_tile(ndarray)" = '[]

_backward_tile ::
               forall args . Fullfilled "_backward_tile(ndarray)" args =>
                 ArgsHMap "_backward_tile(ndarray)" args -> IO [NDArrayHandle]
_backward_tile args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_tile"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_tile_upd ::
                   forall args . Fullfilled "_backward_tile(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_backward_tile(ndarray)" args -> IO ()
_backward_tile_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_tile"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "reverse(ndarray)" =
     '[ '("axis", AttrReq [Int]), '("data", AttrOpt NDArrayHandle)]

reverse ::
        forall args . Fullfilled "reverse(ndarray)" args =>
          ArgsHMap "reverse(ndarray)" args -> IO [NDArrayHandle]
reverse args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "reverse"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

reverse_upd ::
            forall args . Fullfilled "reverse(ndarray)" args =>
              [NDArrayHandle] -> ArgsHMap "reverse(ndarray)" args -> IO ()
reverse_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "reverse"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_reverse(ndarray)" = '[]

_backward_reverse ::
                  forall args . Fullfilled "_backward_reverse(ndarray)" args =>
                    ArgsHMap "_backward_reverse(ndarray)" args -> IO [NDArrayHandle]
_backward_reverse args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_reverse"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_reverse_upd ::
                      forall args . Fullfilled "_backward_reverse(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "_backward_reverse(ndarray)" args -> IO ()
_backward_reverse_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_reverse"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "stack(ndarray)" =
     '[ '("axis", AttrOpt Int), '("num_args", AttrReq Int),
        '("data", AttrOpt [NDArrayHandle])]

stack ::
      forall args . Fullfilled "stack(ndarray)" args =>
        ArgsHMap "stack(ndarray)" args -> IO [NDArrayHandle]
stack args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("num_args",) . showValue <$> (args !? #num_args :: Maybe Int)]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
        array = fromMaybe [] (args !? #data :: Maybe [NDArrayHandle])
      in
      do op <- nnGetOpHandle "stack"
         let scalarArgs'
               = if hasKey args #num_args then scalarArgs else
                   (,) "num_args" (showValue (length array)) : scalarArgs
         listndarr <- mxImperativeInvoke (fromOpHandle op) array scalarArgs'
                        Nothing
         return listndarr

stack_upd ::
          forall args . Fullfilled "stack(ndarray)" args =>
            [NDArrayHandle] -> ArgsHMap "stack(ndarray)" args -> IO ()
stack_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("num_args",) . showValue <$> (args !? #num_args :: Maybe Int)]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
        array = fromMaybe [] (args !? #data :: Maybe [NDArrayHandle])
      in
      do op <- nnGetOpHandle "stack"
         let scalarArgs'
               = if hasKey args #num_args then scalarArgs else
                   (,) "num_args" (showValue (length array)) : scalarArgs
         listndarr <- mxImperativeInvoke (fromOpHandle op) array scalarArgs'
                        (Just outputs)
         return ()

type instance ParameterList "_backward_stack(ndarray)" = '[]

_backward_stack ::
                forall args . Fullfilled "_backward_stack(ndarray)" args =>
                  ArgsHMap "_backward_stack(ndarray)" args -> IO [NDArrayHandle]
_backward_stack args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_stack"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_stack_upd ::
                    forall args . Fullfilled "_backward_stack(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_backward_stack(ndarray)" args -> IO ()
_backward_stack_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_stack"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "dot(ndarray)" =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("lhs", AttrOpt NDArrayHandle), '("rhs", AttrOpt NDArrayHandle)]

dot ::
    forall args . Fullfilled "dot(ndarray)" args =>
      ArgsHMap "dot(ndarray)" args -> IO [NDArrayHandle]
dot args
  = let scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$>
                 (args !? #transpose_a :: Maybe Bool),
               ("transpose_b",) . showValue <$>
                 (args !? #transpose_b :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "dot"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

dot_upd ::
        forall args . Fullfilled "dot(ndarray)" args =>
          [NDArrayHandle] -> ArgsHMap "dot(ndarray)" args -> IO ()
dot_upd outputs args
  = let scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$>
                 (args !? #transpose_a :: Maybe Bool),
               ("transpose_b",) . showValue <$>
                 (args !? #transpose_b :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "dot"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_dot(ndarray)" =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool)]

_backward_dot ::
              forall args . Fullfilled "_backward_dot(ndarray)" args =>
                ArgsHMap "_backward_dot(ndarray)" args -> IO [NDArrayHandle]
_backward_dot args
  = let scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$>
                 (args !? #transpose_a :: Maybe Bool),
               ("transpose_b",) . showValue <$>
                 (args !? #transpose_b :: Maybe Bool)]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_dot"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_dot_upd ::
                  forall args . Fullfilled "_backward_dot(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_backward_dot(ndarray)" args -> IO ()
_backward_dot_upd outputs args
  = let scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$>
                 (args !? #transpose_a :: Maybe Bool),
               ("transpose_b",) . showValue <$>
                 (args !? #transpose_b :: Maybe Bool)]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_dot"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "batch_dot(ndarray)" =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("lhs", AttrOpt NDArrayHandle), '("rhs", AttrOpt NDArrayHandle)]

batch_dot ::
          forall args . Fullfilled "batch_dot(ndarray)" args =>
            ArgsHMap "batch_dot(ndarray)" args -> IO [NDArrayHandle]
batch_dot args
  = let scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$>
                 (args !? #transpose_a :: Maybe Bool),
               ("transpose_b",) . showValue <$>
                 (args !? #transpose_b :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "batch_dot"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

batch_dot_upd ::
              forall args . Fullfilled "batch_dot(ndarray)" args =>
                [NDArrayHandle] -> ArgsHMap "batch_dot(ndarray)" args -> IO ()
batch_dot_upd outputs args
  = let scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$>
                 (args !? #transpose_a :: Maybe Bool),
               ("transpose_b",) . showValue <$>
                 (args !? #transpose_b :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "batch_dot"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_batch_dot(ndarray)" = '[]

_backward_batch_dot ::
                    forall args . Fullfilled "_backward_batch_dot(ndarray)" args =>
                      ArgsHMap "_backward_batch_dot(ndarray)" args -> IO [NDArrayHandle]
_backward_batch_dot args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_batch_dot"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_batch_dot_upd ::
                        forall args . Fullfilled "_backward_batch_dot(ndarray)" args =>
                          [NDArrayHandle] ->
                            ArgsHMap "_backward_batch_dot(ndarray)" args -> IO ()
_backward_batch_dot_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_batch_dot"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "sin(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

sin ::
    forall args . Fullfilled "sin(ndarray)" args =>
      ArgsHMap "sin(ndarray)" args -> IO [NDArrayHandle]
sin args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sin"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

sin_upd ::
        forall args . Fullfilled "sin(ndarray)" args =>
          [NDArrayHandle] -> ArgsHMap "sin(ndarray)" args -> IO ()
sin_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sin"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_sin(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_sin ::
              forall args . Fullfilled "_backward_sin(ndarray)" args =>
                ArgsHMap "_backward_sin(ndarray)" args -> IO [NDArrayHandle]
_backward_sin args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sin"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_sin_upd ::
                  forall args . Fullfilled "_backward_sin(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_backward_sin(ndarray)" args -> IO ()
_backward_sin_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sin"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "cos(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

cos ::
    forall args . Fullfilled "cos(ndarray)" args =>
      ArgsHMap "cos(ndarray)" args -> IO [NDArrayHandle]
cos args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "cos"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

cos_upd ::
        forall args . Fullfilled "cos(ndarray)" args =>
          [NDArrayHandle] -> ArgsHMap "cos(ndarray)" args -> IO ()
cos_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "cos"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_cos(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_cos ::
              forall args . Fullfilled "_backward_cos(ndarray)" args =>
                ArgsHMap "_backward_cos(ndarray)" args -> IO [NDArrayHandle]
_backward_cos args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_cos"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_cos_upd ::
                  forall args . Fullfilled "_backward_cos(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_backward_cos(ndarray)" args -> IO ()
_backward_cos_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_cos"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "tan(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

tan ::
    forall args . Fullfilled "tan(ndarray)" args =>
      ArgsHMap "tan(ndarray)" args -> IO [NDArrayHandle]
tan args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "tan"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

tan_upd ::
        forall args . Fullfilled "tan(ndarray)" args =>
          [NDArrayHandle] -> ArgsHMap "tan(ndarray)" args -> IO ()
tan_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "tan"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_tan(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_tan ::
              forall args . Fullfilled "_backward_tan(ndarray)" args =>
                ArgsHMap "_backward_tan(ndarray)" args -> IO [NDArrayHandle]
_backward_tan args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_tan"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_tan_upd ::
                  forall args . Fullfilled "_backward_tan(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_backward_tan(ndarray)" args -> IO ()
_backward_tan_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_tan"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "arcsin(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

arcsin ::
       forall args . Fullfilled "arcsin(ndarray)" args =>
         ArgsHMap "arcsin(ndarray)" args -> IO [NDArrayHandle]
arcsin args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "arcsin"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

arcsin_upd ::
           forall args . Fullfilled "arcsin(ndarray)" args =>
             [NDArrayHandle] -> ArgsHMap "arcsin(ndarray)" args -> IO ()
arcsin_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "arcsin"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_arcsin(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_arcsin ::
                 forall args . Fullfilled "_backward_arcsin(ndarray)" args =>
                   ArgsHMap "_backward_arcsin(ndarray)" args -> IO [NDArrayHandle]
_backward_arcsin args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_arcsin"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_arcsin_upd ::
                     forall args . Fullfilled "_backward_arcsin(ndarray)" args =>
                       [NDArrayHandle] ->
                         ArgsHMap "_backward_arcsin(ndarray)" args -> IO ()
_backward_arcsin_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_arcsin"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "arccos(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

arccos ::
       forall args . Fullfilled "arccos(ndarray)" args =>
         ArgsHMap "arccos(ndarray)" args -> IO [NDArrayHandle]
arccos args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "arccos"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

arccos_upd ::
           forall args . Fullfilled "arccos(ndarray)" args =>
             [NDArrayHandle] -> ArgsHMap "arccos(ndarray)" args -> IO ()
arccos_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "arccos"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_arccos(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_arccos ::
                 forall args . Fullfilled "_backward_arccos(ndarray)" args =>
                   ArgsHMap "_backward_arccos(ndarray)" args -> IO [NDArrayHandle]
_backward_arccos args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_arccos"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_arccos_upd ::
                     forall args . Fullfilled "_backward_arccos(ndarray)" args =>
                       [NDArrayHandle] ->
                         ArgsHMap "_backward_arccos(ndarray)" args -> IO ()
_backward_arccos_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_arccos"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "arctan(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

arctan ::
       forall args . Fullfilled "arctan(ndarray)" args =>
         ArgsHMap "arctan(ndarray)" args -> IO [NDArrayHandle]
arctan args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "arctan"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

arctan_upd ::
           forall args . Fullfilled "arctan(ndarray)" args =>
             [NDArrayHandle] -> ArgsHMap "arctan(ndarray)" args -> IO ()
arctan_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "arctan"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_arctan(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_arctan ::
                 forall args . Fullfilled "_backward_arctan(ndarray)" args =>
                   ArgsHMap "_backward_arctan(ndarray)" args -> IO [NDArrayHandle]
_backward_arctan args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_arctan"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_arctan_upd ::
                     forall args . Fullfilled "_backward_arctan(ndarray)" args =>
                       [NDArrayHandle] ->
                         ArgsHMap "_backward_arctan(ndarray)" args -> IO ()
_backward_arctan_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_arctan"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "degrees(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

degrees ::
        forall args . Fullfilled "degrees(ndarray)" args =>
          ArgsHMap "degrees(ndarray)" args -> IO [NDArrayHandle]
degrees args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "degrees"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

degrees_upd ::
            forall args . Fullfilled "degrees(ndarray)" args =>
              [NDArrayHandle] -> ArgsHMap "degrees(ndarray)" args -> IO ()
degrees_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "degrees"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_degrees(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_degrees ::
                  forall args . Fullfilled "_backward_degrees(ndarray)" args =>
                    ArgsHMap "_backward_degrees(ndarray)" args -> IO [NDArrayHandle]
_backward_degrees args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_degrees"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_degrees_upd ::
                      forall args . Fullfilled "_backward_degrees(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "_backward_degrees(ndarray)" args -> IO ()
_backward_degrees_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_degrees"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "radians(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

radians ::
        forall args . Fullfilled "radians(ndarray)" args =>
          ArgsHMap "radians(ndarray)" args -> IO [NDArrayHandle]
radians args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "radians"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

radians_upd ::
            forall args . Fullfilled "radians(ndarray)" args =>
              [NDArrayHandle] -> ArgsHMap "radians(ndarray)" args -> IO ()
radians_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "radians"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_radians(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_radians ::
                  forall args . Fullfilled "_backward_radians(ndarray)" args =>
                    ArgsHMap "_backward_radians(ndarray)" args -> IO [NDArrayHandle]
_backward_radians args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_radians"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_radians_upd ::
                      forall args . Fullfilled "_backward_radians(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "_backward_radians(ndarray)" args -> IO ()
_backward_radians_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_radians"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "sinh(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

sinh ::
     forall args . Fullfilled "sinh(ndarray)" args =>
       ArgsHMap "sinh(ndarray)" args -> IO [NDArrayHandle]
sinh args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sinh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

sinh_upd ::
         forall args . Fullfilled "sinh(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "sinh(ndarray)" args -> IO ()
sinh_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sinh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_sinh(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_sinh ::
               forall args . Fullfilled "_backward_sinh(ndarray)" args =>
                 ArgsHMap "_backward_sinh(ndarray)" args -> IO [NDArrayHandle]
_backward_sinh args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sinh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_sinh_upd ::
                   forall args . Fullfilled "_backward_sinh(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_backward_sinh(ndarray)" args -> IO ()
_backward_sinh_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sinh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "cosh(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

cosh ::
     forall args . Fullfilled "cosh(ndarray)" args =>
       ArgsHMap "cosh(ndarray)" args -> IO [NDArrayHandle]
cosh args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "cosh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

cosh_upd ::
         forall args . Fullfilled "cosh(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "cosh(ndarray)" args -> IO ()
cosh_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "cosh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_cosh(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_cosh ::
               forall args . Fullfilled "_backward_cosh(ndarray)" args =>
                 ArgsHMap "_backward_cosh(ndarray)" args -> IO [NDArrayHandle]
_backward_cosh args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_cosh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_cosh_upd ::
                   forall args . Fullfilled "_backward_cosh(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_backward_cosh(ndarray)" args -> IO ()
_backward_cosh_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_cosh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "tanh(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

tanh ::
     forall args . Fullfilled "tanh(ndarray)" args =>
       ArgsHMap "tanh(ndarray)" args -> IO [NDArrayHandle]
tanh args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "tanh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

tanh_upd ::
         forall args . Fullfilled "tanh(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "tanh(ndarray)" args -> IO ()
tanh_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "tanh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_tanh(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_tanh ::
               forall args . Fullfilled "_backward_tanh(ndarray)" args =>
                 ArgsHMap "_backward_tanh(ndarray)" args -> IO [NDArrayHandle]
_backward_tanh args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_tanh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_tanh_upd ::
                   forall args . Fullfilled "_backward_tanh(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_backward_tanh(ndarray)" args -> IO ()
_backward_tanh_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_tanh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "arcsinh(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

arcsinh ::
        forall args . Fullfilled "arcsinh(ndarray)" args =>
          ArgsHMap "arcsinh(ndarray)" args -> IO [NDArrayHandle]
arcsinh args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "arcsinh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

arcsinh_upd ::
            forall args . Fullfilled "arcsinh(ndarray)" args =>
              [NDArrayHandle] -> ArgsHMap "arcsinh(ndarray)" args -> IO ()
arcsinh_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "arcsinh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_arcsinh(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_arcsinh ::
                  forall args . Fullfilled "_backward_arcsinh(ndarray)" args =>
                    ArgsHMap "_backward_arcsinh(ndarray)" args -> IO [NDArrayHandle]
_backward_arcsinh args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_arcsinh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_arcsinh_upd ::
                      forall args . Fullfilled "_backward_arcsinh(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "_backward_arcsinh(ndarray)" args -> IO ()
_backward_arcsinh_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_arcsinh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "arccosh(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

arccosh ::
        forall args . Fullfilled "arccosh(ndarray)" args =>
          ArgsHMap "arccosh(ndarray)" args -> IO [NDArrayHandle]
arccosh args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "arccosh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

arccosh_upd ::
            forall args . Fullfilled "arccosh(ndarray)" args =>
              [NDArrayHandle] -> ArgsHMap "arccosh(ndarray)" args -> IO ()
arccosh_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "arccosh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_arccosh(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_arccosh ::
                  forall args . Fullfilled "_backward_arccosh(ndarray)" args =>
                    ArgsHMap "_backward_arccosh(ndarray)" args -> IO [NDArrayHandle]
_backward_arccosh args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_arccosh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_arccosh_upd ::
                      forall args . Fullfilled "_backward_arccosh(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "_backward_arccosh(ndarray)" args -> IO ()
_backward_arccosh_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_arccosh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "arctanh(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

arctanh ::
        forall args . Fullfilled "arctanh(ndarray)" args =>
          ArgsHMap "arctanh(ndarray)" args -> IO [NDArrayHandle]
arctanh args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "arctanh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

arctanh_upd ::
            forall args . Fullfilled "arctanh(ndarray)" args =>
              [NDArrayHandle] -> ArgsHMap "arctanh(ndarray)" args -> IO ()
arctanh_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "arctanh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_arctanh(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_arctanh ::
                  forall args . Fullfilled "_backward_arctanh(ndarray)" args =>
                    ArgsHMap "_backward_arctanh(ndarray)" args -> IO [NDArrayHandle]
_backward_arctanh args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_arctanh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_arctanh_upd ::
                      forall args . Fullfilled "_backward_arctanh(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "_backward_arctanh(ndarray)" args -> IO ()
_backward_arctanh_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_arctanh"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_contrib_dequantize(ndarray)" =
     '[ '("out_type", AttrReq (EnumType '["float32"])),
        '("input", AttrOpt NDArrayHandle),
        '("min_range", AttrOpt NDArrayHandle),
        '("max_range", AttrOpt NDArrayHandle)]

_contrib_dequantize ::
                    forall args . Fullfilled "_contrib_dequantize(ndarray)" args =>
                      ArgsHMap "_contrib_dequantize(ndarray)" args -> IO [NDArrayHandle]
_contrib_dequantize args
  = let scalarArgs
          = catMaybes
              [("out_type",) . showValue <$>
                 (args !? #out_type :: Maybe (EnumType '["float32"]))]
        tensorArgs
          = catMaybes
              [("input",) <$> (args !? #input :: Maybe NDArrayHandle),
               ("min_range",) <$> (args !? #min_range :: Maybe NDArrayHandle),
               ("max_range",) <$> (args !? #max_range :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_dequantize"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_contrib_dequantize_upd ::
                        forall args . Fullfilled "_contrib_dequantize(ndarray)" args =>
                          [NDArrayHandle] ->
                            ArgsHMap "_contrib_dequantize(ndarray)" args -> IO ()
_contrib_dequantize_upd outputs args
  = let scalarArgs
          = catMaybes
              [("out_type",) . showValue <$>
                 (args !? #out_type :: Maybe (EnumType '["float32"]))]
        tensorArgs
          = catMaybes
              [("input",) <$> (args !? #input :: Maybe NDArrayHandle),
               ("min_range",) <$> (args !? #min_range :: Maybe NDArrayHandle),
               ("max_range",) <$> (args !? #max_range :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_dequantize"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_contrib_CTCLoss(ndarray)" =
     '[ '("use_data_lengths", AttrOpt Bool),
        '("use_label_lengths", AttrOpt Bool),
        '("blank_label", AttrOpt (EnumType '["first", "last"])),
        '("data", AttrOpt NDArrayHandle),
        '("label", AttrOpt NDArrayHandle),
        '("data_lengths", AttrOpt NDArrayHandle),
        '("label_lengths", AttrOpt NDArrayHandle)]

_contrib_CTCLoss ::
                 forall args . Fullfilled "_contrib_CTCLoss(ndarray)" args =>
                   ArgsHMap "_contrib_CTCLoss(ndarray)" args -> IO [NDArrayHandle]
_contrib_CTCLoss args
  = let scalarArgs
          = catMaybes
              [("use_data_lengths",) . showValue <$>
                 (args !? #use_data_lengths :: Maybe Bool),
               ("use_label_lengths",) . showValue <$>
                 (args !? #use_label_lengths :: Maybe Bool),
               ("blank_label",) . showValue <$>
                 (args !? #blank_label :: Maybe (EnumType '["first", "last"]))]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("label",) <$> (args !? #label :: Maybe NDArrayHandle),
               ("data_lengths",) <$>
                 (args !? #data_lengths :: Maybe NDArrayHandle),
               ("label_lengths",) <$>
                 (args !? #label_lengths :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_CTCLoss"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_contrib_CTCLoss_upd ::
                     forall args . Fullfilled "_contrib_CTCLoss(ndarray)" args =>
                       [NDArrayHandle] ->
                         ArgsHMap "_contrib_CTCLoss(ndarray)" args -> IO ()
_contrib_CTCLoss_upd outputs args
  = let scalarArgs
          = catMaybes
              [("use_data_lengths",) . showValue <$>
                 (args !? #use_data_lengths :: Maybe Bool),
               ("use_label_lengths",) . showValue <$>
                 (args !? #use_label_lengths :: Maybe Bool),
               ("blank_label",) . showValue <$>
                 (args !? #blank_label :: Maybe (EnumType '["first", "last"]))]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("label",) <$> (args !? #label :: Maybe NDArrayHandle),
               ("data_lengths",) <$>
                 (args !? #data_lengths :: Maybe NDArrayHandle),
               ("label_lengths",) <$>
                 (args !? #label_lengths :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_CTCLoss"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_contrib_quantize(ndarray)" =
     '[ '("out_type", AttrOpt (EnumType '["uint8"])),
        '("input", AttrOpt NDArrayHandle),
        '("min_range", AttrOpt NDArrayHandle),
        '("max_range", AttrOpt NDArrayHandle)]

_contrib_quantize ::
                  forall args . Fullfilled "_contrib_quantize(ndarray)" args =>
                    ArgsHMap "_contrib_quantize(ndarray)" args -> IO [NDArrayHandle]
_contrib_quantize args
  = let scalarArgs
          = catMaybes
              [("out_type",) . showValue <$>
                 (args !? #out_type :: Maybe (EnumType '["uint8"]))]
        tensorArgs
          = catMaybes
              [("input",) <$> (args !? #input :: Maybe NDArrayHandle),
               ("min_range",) <$> (args !? #min_range :: Maybe NDArrayHandle),
               ("max_range",) <$> (args !? #max_range :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_quantize"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_contrib_quantize_upd ::
                      forall args . Fullfilled "_contrib_quantize(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "_contrib_quantize(ndarray)" args -> IO ()
_contrib_quantize_upd outputs args
  = let scalarArgs
          = catMaybes
              [("out_type",) . showValue <$>
                 (args !? #out_type :: Maybe (EnumType '["uint8"]))]
        tensorArgs
          = catMaybes
              [("input",) <$> (args !? #input :: Maybe NDArrayHandle),
               ("min_range",) <$> (args !? #min_range :: Maybe NDArrayHandle),
               ("max_range",) <$> (args !? #max_range :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_quantize"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_Custom(ndarray)" = '[]

_backward_Custom ::
                 forall args . Fullfilled "_backward_Custom(ndarray)" args =>
                   ArgsHMap "_backward_Custom(ndarray)" args -> IO [NDArrayHandle]
_backward_Custom args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Custom"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_Custom_upd ::
                     forall args . Fullfilled "_backward_Custom(ndarray)" args =>
                       [NDArrayHandle] ->
                         ArgsHMap "_backward_Custom(ndarray)" args -> IO ()
_backward_Custom_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Custom"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_sample_multinomial(ndarray)" =
     '[ '("shape", AttrOpt [Int]), '("get_prob", AttrOpt Bool),
        '("dtype", AttrOpt (EnumType '["int32"])),
        '("data", AttrOpt NDArrayHandle)]

_sample_multinomial ::
                    forall args . Fullfilled "_sample_multinomial(ndarray)" args =>
                      ArgsHMap "_sample_multinomial(ndarray)" args -> IO [NDArrayHandle]
_sample_multinomial args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("get_prob",) . showValue <$> (args !? #get_prob :: Maybe Bool),
               ("dtype",) . showValue <$>
                 (args !? #dtype :: Maybe (EnumType '["int32"]))]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_multinomial"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_sample_multinomial_upd ::
                        forall args . Fullfilled "_sample_multinomial(ndarray)" args =>
                          [NDArrayHandle] ->
                            ArgsHMap "_sample_multinomial(ndarray)" args -> IO ()
_sample_multinomial_upd outputs args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("get_prob",) . showValue <$> (args !? #get_prob :: Maybe Bool),
               ("dtype",) . showValue <$>
                 (args !? #dtype :: Maybe (EnumType '["int32"]))]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_multinomial"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_sample_multinomial(ndarray)"
     = '[]

_backward_sample_multinomial ::
                             forall args .
                               Fullfilled "_backward_sample_multinomial(ndarray)" args =>
                               ArgsHMap "_backward_sample_multinomial(ndarray)" args ->
                                 IO [NDArrayHandle]
_backward_sample_multinomial args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sample_multinomial"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_sample_multinomial_upd ::
                                 forall args .
                                   Fullfilled "_backward_sample_multinomial(ndarray)" args =>
                                   [NDArrayHandle] ->
                                     ArgsHMap "_backward_sample_multinomial(ndarray)" args -> IO ()
_backward_sample_multinomial_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sample_multinomial"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_sample_uniform(ndarray)" =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("low", AttrOpt NDArrayHandle), '("high", AttrOpt NDArrayHandle)]

_sample_uniform ::
                forall args . Fullfilled "_sample_uniform(ndarray)" args =>
                  ArgsHMap "_sample_uniform(ndarray)" args -> IO [NDArrayHandle]
_sample_uniform args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs
          = catMaybes
              [("low",) <$> (args !? #low :: Maybe NDArrayHandle),
               ("high",) <$> (args !? #high :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_uniform"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_sample_uniform_upd ::
                    forall args . Fullfilled "_sample_uniform(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_sample_uniform(ndarray)" args -> IO ()
_sample_uniform_upd outputs args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs
          = catMaybes
              [("low",) <$> (args !? #low :: Maybe NDArrayHandle),
               ("high",) <$> (args !? #high :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_uniform"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_sample_normal(ndarray)" =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("mu", AttrOpt NDArrayHandle), '("sigma", AttrOpt NDArrayHandle)]

_sample_normal ::
               forall args . Fullfilled "_sample_normal(ndarray)" args =>
                 ArgsHMap "_sample_normal(ndarray)" args -> IO [NDArrayHandle]
_sample_normal args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs
          = catMaybes
              [("mu",) <$> (args !? #mu :: Maybe NDArrayHandle),
               ("sigma",) <$> (args !? #sigma :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_normal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_sample_normal_upd ::
                   forall args . Fullfilled "_sample_normal(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_sample_normal(ndarray)" args -> IO ()
_sample_normal_upd outputs args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs
          = catMaybes
              [("mu",) <$> (args !? #mu :: Maybe NDArrayHandle),
               ("sigma",) <$> (args !? #sigma :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_normal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_sample_gamma(ndarray)" =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("alpha", AttrOpt NDArrayHandle),
        '("beta", AttrOpt NDArrayHandle)]

_sample_gamma ::
              forall args . Fullfilled "_sample_gamma(ndarray)" args =>
                ArgsHMap "_sample_gamma(ndarray)" args -> IO [NDArrayHandle]
_sample_gamma args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs
          = catMaybes
              [("alpha",) <$> (args !? #alpha :: Maybe NDArrayHandle),
               ("beta",) <$> (args !? #beta :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_gamma"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_sample_gamma_upd ::
                  forall args . Fullfilled "_sample_gamma(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_sample_gamma(ndarray)" args -> IO ()
_sample_gamma_upd outputs args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs
          = catMaybes
              [("alpha",) <$> (args !? #alpha :: Maybe NDArrayHandle),
               ("beta",) <$> (args !? #beta :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_gamma"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_sample_exponential(ndarray)" =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("lam", AttrOpt NDArrayHandle)]

_sample_exponential ::
                    forall args . Fullfilled "_sample_exponential(ndarray)" args =>
                      ArgsHMap "_sample_exponential(ndarray)" args -> IO [NDArrayHandle]
_sample_exponential args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs
          = catMaybes [("lam",) <$> (args !? #lam :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_exponential"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_sample_exponential_upd ::
                        forall args . Fullfilled "_sample_exponential(ndarray)" args =>
                          [NDArrayHandle] ->
                            ArgsHMap "_sample_exponential(ndarray)" args -> IO ()
_sample_exponential_upd outputs args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs
          = catMaybes [("lam",) <$> (args !? #lam :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_exponential"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_sample_poisson(ndarray)" =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("lam", AttrOpt NDArrayHandle)]

_sample_poisson ::
                forall args . Fullfilled "_sample_poisson(ndarray)" args =>
                  ArgsHMap "_sample_poisson(ndarray)" args -> IO [NDArrayHandle]
_sample_poisson args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs
          = catMaybes [("lam",) <$> (args !? #lam :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_poisson"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_sample_poisson_upd ::
                    forall args . Fullfilled "_sample_poisson(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_sample_poisson(ndarray)" args -> IO ()
_sample_poisson_upd outputs args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs
          = catMaybes [("lam",) <$> (args !? #lam :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_poisson"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_sample_negative_binomial(ndarray)" =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("k", AttrOpt NDArrayHandle), '("p", AttrOpt NDArrayHandle)]

_sample_negative_binomial ::
                          forall args .
                            Fullfilled "_sample_negative_binomial(ndarray)" args =>
                            ArgsHMap "_sample_negative_binomial(ndarray)" args ->
                              IO [NDArrayHandle]
_sample_negative_binomial args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs
          = catMaybes
              [("k",) <$> (args !? #k :: Maybe NDArrayHandle),
               ("p",) <$> (args !? #p :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_negative_binomial"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_sample_negative_binomial_upd ::
                              forall args .
                                Fullfilled "_sample_negative_binomial(ndarray)" args =>
                                [NDArrayHandle] ->
                                  ArgsHMap "_sample_negative_binomial(ndarray)" args -> IO ()
_sample_negative_binomial_upd outputs args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs
          = catMaybes
              [("k",) <$> (args !? #k :: Maybe NDArrayHandle),
               ("p",) <$> (args !? #p :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_negative_binomial"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance
     ParameterList "_sample_generalized_negative_binomial(ndarray)" =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("mu", AttrOpt NDArrayHandle), '("alpha", AttrOpt NDArrayHandle)]

_sample_generalized_negative_binomial ::
                                      forall args .
                                        Fullfilled "_sample_generalized_negative_binomial(ndarray)"
                                          args =>
                                        ArgsHMap "_sample_generalized_negative_binomial(ndarray)"
                                          args
                                          -> IO [NDArrayHandle]
_sample_generalized_negative_binomial args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs
          = catMaybes
              [("mu",) <$> (args !? #mu :: Maybe NDArrayHandle),
               ("alpha",) <$> (args !? #alpha :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_generalized_negative_binomial"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_sample_generalized_negative_binomial_upd ::
                                          forall args .
                                            Fullfilled
                                              "_sample_generalized_negative_binomial(ndarray)"
                                              args =>
                                            [NDArrayHandle] ->
                                              ArgsHMap
                                                "_sample_generalized_negative_binomial(ndarray)"
                                                args
                                                -> IO ()
_sample_generalized_negative_binomial_upd outputs args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs
          = catMaybes
              [("mu",) <$> (args !? #mu :: Maybe NDArrayHandle),
               ("alpha",) <$> (args !? #alpha :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_generalized_negative_binomial"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_random_uniform(ndarray)" =
     '[ '("low", AttrOpt Float), '("high", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

_random_uniform ::
                forall args . Fullfilled "_random_uniform(ndarray)" args =>
                  ArgsHMap "_random_uniform(ndarray)" args -> IO [NDArrayHandle]
_random_uniform args
  = let scalarArgs
          = catMaybes
              [("low",) . showValue <$> (args !? #low :: Maybe Float),
               ("high",) . showValue <$> (args !? #high :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_random_uniform"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_random_uniform_upd ::
                    forall args . Fullfilled "_random_uniform(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_random_uniform(ndarray)" args -> IO ()
_random_uniform_upd outputs args
  = let scalarArgs
          = catMaybes
              [("low",) . showValue <$> (args !? #low :: Maybe Float),
               ("high",) . showValue <$> (args !? #high :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_random_uniform"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_random_normal(ndarray)" =
     '[ '("loc", AttrOpt Float), '("scale", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

_random_normal ::
               forall args . Fullfilled "_random_normal(ndarray)" args =>
                 ArgsHMap "_random_normal(ndarray)" args -> IO [NDArrayHandle]
_random_normal args
  = let scalarArgs
          = catMaybes
              [("loc",) . showValue <$> (args !? #loc :: Maybe Float),
               ("scale",) . showValue <$> (args !? #scale :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_random_normal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_random_normal_upd ::
                   forall args . Fullfilled "_random_normal(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_random_normal(ndarray)" args -> IO ()
_random_normal_upd outputs args
  = let scalarArgs
          = catMaybes
              [("loc",) . showValue <$> (args !? #loc :: Maybe Float),
               ("scale",) . showValue <$> (args !? #scale :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_random_normal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_random_gamma(ndarray)" =
     '[ '("alpha", AttrOpt Float), '("beta", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

_random_gamma ::
              forall args . Fullfilled "_random_gamma(ndarray)" args =>
                ArgsHMap "_random_gamma(ndarray)" args -> IO [NDArrayHandle]
_random_gamma args
  = let scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> (args !? #alpha :: Maybe Float),
               ("beta",) . showValue <$> (args !? #beta :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_random_gamma"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_random_gamma_upd ::
                  forall args . Fullfilled "_random_gamma(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_random_gamma(ndarray)" args -> IO ()
_random_gamma_upd outputs args
  = let scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> (args !? #alpha :: Maybe Float),
               ("beta",) . showValue <$> (args !? #beta :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_random_gamma"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_random_exponential(ndarray)" =
     '[ '("lam", AttrOpt Float), '("shape", AttrOpt [Int]),
        '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

_random_exponential ::
                    forall args . Fullfilled "_random_exponential(ndarray)" args =>
                      ArgsHMap "_random_exponential(ndarray)" args -> IO [NDArrayHandle]
_random_exponential args
  = let scalarArgs
          = catMaybes
              [("lam",) . showValue <$> (args !? #lam :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_random_exponential"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_random_exponential_upd ::
                        forall args . Fullfilled "_random_exponential(ndarray)" args =>
                          [NDArrayHandle] ->
                            ArgsHMap "_random_exponential(ndarray)" args -> IO ()
_random_exponential_upd outputs args
  = let scalarArgs
          = catMaybes
              [("lam",) . showValue <$> (args !? #lam :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_random_exponential"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_random_poisson(ndarray)" =
     '[ '("lam", AttrOpt Float), '("shape", AttrOpt [Int]),
        '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

_random_poisson ::
                forall args . Fullfilled "_random_poisson(ndarray)" args =>
                  ArgsHMap "_random_poisson(ndarray)" args -> IO [NDArrayHandle]
_random_poisson args
  = let scalarArgs
          = catMaybes
              [("lam",) . showValue <$> (args !? #lam :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_random_poisson"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_random_poisson_upd ::
                    forall args . Fullfilled "_random_poisson(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_random_poisson(ndarray)" args -> IO ()
_random_poisson_upd outputs args
  = let scalarArgs
          = catMaybes
              [("lam",) . showValue <$> (args !? #lam :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_random_poisson"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_random_negative_binomial(ndarray)" =
     '[ '("k", AttrOpt Int), '("p", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

_random_negative_binomial ::
                          forall args .
                            Fullfilled "_random_negative_binomial(ndarray)" args =>
                            ArgsHMap "_random_negative_binomial(ndarray)" args ->
                              IO [NDArrayHandle]
_random_negative_binomial args
  = let scalarArgs
          = catMaybes
              [("k",) . showValue <$> (args !? #k :: Maybe Int),
               ("p",) . showValue <$> (args !? #p :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_random_negative_binomial"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_random_negative_binomial_upd ::
                              forall args .
                                Fullfilled "_random_negative_binomial(ndarray)" args =>
                                [NDArrayHandle] ->
                                  ArgsHMap "_random_negative_binomial(ndarray)" args -> IO ()
_random_negative_binomial_upd outputs args
  = let scalarArgs
          = catMaybes
              [("k",) . showValue <$> (args !? #k :: Maybe Int),
               ("p",) . showValue <$> (args !? #p :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_random_negative_binomial"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance
     ParameterList "_random_generalized_negative_binomial(ndarray)" =
     '[ '("mu", AttrOpt Float), '("alpha", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

_random_generalized_negative_binomial ::
                                      forall args .
                                        Fullfilled "_random_generalized_negative_binomial(ndarray)"
                                          args =>
                                        ArgsHMap "_random_generalized_negative_binomial(ndarray)"
                                          args
                                          -> IO [NDArrayHandle]
_random_generalized_negative_binomial args
  = let scalarArgs
          = catMaybes
              [("mu",) . showValue <$> (args !? #mu :: Maybe Float),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_random_generalized_negative_binomial"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_random_generalized_negative_binomial_upd ::
                                          forall args .
                                            Fullfilled
                                              "_random_generalized_negative_binomial(ndarray)"
                                              args =>
                                            [NDArrayHandle] ->
                                              ArgsHMap
                                                "_random_generalized_negative_binomial(ndarray)"
                                                args
                                                -> IO ()
_random_generalized_negative_binomial_upd outputs args
  = let scalarArgs
          = catMaybes
              [("mu",) . showValue <$> (args !? #mu :: Maybe Float),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_random_generalized_negative_binomial"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "softmax(ndarray)" =
     '[ '("axis", AttrOpt Int), '("data", AttrOpt NDArrayHandle)]

softmax ::
        forall args . Fullfilled "softmax(ndarray)" args =>
          ArgsHMap "softmax(ndarray)" args -> IO [NDArrayHandle]
softmax args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "softmax"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

softmax_upd ::
            forall args . Fullfilled "softmax(ndarray)" args =>
              [NDArrayHandle] -> ArgsHMap "softmax(ndarray)" args -> IO ()
softmax_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "softmax"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_softmax(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_softmax ::
                  forall args . Fullfilled "_backward_softmax(ndarray)" args =>
                    ArgsHMap "_backward_softmax(ndarray)" args -> IO [NDArrayHandle]
_backward_softmax args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_softmax"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_softmax_upd ::
                      forall args . Fullfilled "_backward_softmax(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "_backward_softmax(ndarray)" args -> IO ()
_backward_softmax_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_softmax"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "log_softmax(ndarray)" =
     '[ '("axis", AttrOpt Int), '("data", AttrOpt NDArrayHandle)]

log_softmax ::
            forall args . Fullfilled "log_softmax(ndarray)" args =>
              ArgsHMap "log_softmax(ndarray)" args -> IO [NDArrayHandle]
log_softmax args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "log_softmax"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

log_softmax_upd ::
                forall args . Fullfilled "log_softmax(ndarray)" args =>
                  [NDArrayHandle] -> ArgsHMap "log_softmax(ndarray)" args -> IO ()
log_softmax_upd outputs args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "log_softmax"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_log_softmax(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_backward_log_softmax ::
                      forall args . Fullfilled "_backward_log_softmax(ndarray)" args =>
                        ArgsHMap "_backward_log_softmax(ndarray)" args ->
                          IO [NDArrayHandle]
_backward_log_softmax args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_log_softmax"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_log_softmax_upd ::
                          forall args . Fullfilled "_backward_log_softmax(ndarray)" args =>
                            [NDArrayHandle] ->
                              ArgsHMap "_backward_log_softmax(ndarray)" args -> IO ()
_backward_log_softmax_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_log_softmax"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_cvimdecode(ndarray)" =
     '[ '("flag", AttrOpt Int), '("to_rgb", AttrOpt Bool),
        '("buf", AttrOpt NDArrayHandle)]

_cvimdecode ::
            forall args . Fullfilled "_cvimdecode(ndarray)" args =>
              ArgsHMap "_cvimdecode(ndarray)" args -> IO [NDArrayHandle]
_cvimdecode args
  = let scalarArgs
          = catMaybes
              [("flag",) . showValue <$> (args !? #flag :: Maybe Int),
               ("to_rgb",) . showValue <$> (args !? #to_rgb :: Maybe Bool)]
        tensorArgs
          = catMaybes [("buf",) <$> (args !? #buf :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_cvimdecode"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_cvimdecode_upd ::
                forall args . Fullfilled "_cvimdecode(ndarray)" args =>
                  [NDArrayHandle] -> ArgsHMap "_cvimdecode(ndarray)" args -> IO ()
_cvimdecode_upd outputs args
  = let scalarArgs
          = catMaybes
              [("flag",) . showValue <$> (args !? #flag :: Maybe Int),
               ("to_rgb",) . showValue <$> (args !? #to_rgb :: Maybe Bool)]
        tensorArgs
          = catMaybes [("buf",) <$> (args !? #buf :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_cvimdecode"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_cvimread(ndarray)" =
     '[ '("filename", AttrReq String), '("flag", AttrOpt Int),
        '("to_rgb", AttrOpt Bool)]

_cvimread ::
          forall args . Fullfilled "_cvimread(ndarray)" args =>
            ArgsHMap "_cvimread(ndarray)" args -> IO [NDArrayHandle]
_cvimread args
  = let scalarArgs
          = catMaybes
              [("filename",) . showValue <$> (args !? #filename :: Maybe String),
               ("flag",) . showValue <$> (args !? #flag :: Maybe Int),
               ("to_rgb",) . showValue <$> (args !? #to_rgb :: Maybe Bool)]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_cvimread"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_cvimread_upd ::
              forall args . Fullfilled "_cvimread(ndarray)" args =>
                [NDArrayHandle] -> ArgsHMap "_cvimread(ndarray)" args -> IO ()
_cvimread_upd outputs args
  = let scalarArgs
          = catMaybes
              [("filename",) . showValue <$> (args !? #filename :: Maybe String),
               ("flag",) . showValue <$> (args !? #flag :: Maybe Int),
               ("to_rgb",) . showValue <$> (args !? #to_rgb :: Maybe Bool)]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_cvimread"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_cvimresize(ndarray)" =
     '[ '("w", AttrReq Int), '("h", AttrReq Int),
        '("interp", AttrOpt Int), '("src", AttrOpt NDArrayHandle)]

_cvimresize ::
            forall args . Fullfilled "_cvimresize(ndarray)" args =>
              ArgsHMap "_cvimresize(ndarray)" args -> IO [NDArrayHandle]
_cvimresize args
  = let scalarArgs
          = catMaybes
              [("w",) . showValue <$> (args !? #w :: Maybe Int),
               ("h",) . showValue <$> (args !? #h :: Maybe Int),
               ("interp",) . showValue <$> (args !? #interp :: Maybe Int)]
        tensorArgs
          = catMaybes [("src",) <$> (args !? #src :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_cvimresize"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_cvimresize_upd ::
                forall args . Fullfilled "_cvimresize(ndarray)" args =>
                  [NDArrayHandle] -> ArgsHMap "_cvimresize(ndarray)" args -> IO ()
_cvimresize_upd outputs args
  = let scalarArgs
          = catMaybes
              [("w",) . showValue <$> (args !? #w :: Maybe Int),
               ("h",) . showValue <$> (args !? #h :: Maybe Int),
               ("interp",) . showValue <$> (args !? #interp :: Maybe Int)]
        tensorArgs
          = catMaybes [("src",) <$> (args !? #src :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_cvimresize"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_cvcopyMakeBorder(ndarray)" =
     '[ '("top", AttrReq Int), '("bot", AttrReq Int),
        '("left", AttrReq Int), '("right", AttrReq Int),
        '("type", AttrOpt Int), '("value", AttrOpt Double),
        '("values", AttrOpt [Double]), '("src", AttrOpt NDArrayHandle)]

_cvcopyMakeBorder ::
                  forall args . Fullfilled "_cvcopyMakeBorder(ndarray)" args =>
                    ArgsHMap "_cvcopyMakeBorder(ndarray)" args -> IO [NDArrayHandle]
_cvcopyMakeBorder args
  = let scalarArgs
          = catMaybes
              [("top",) . showValue <$> (args !? #top :: Maybe Int),
               ("bot",) . showValue <$> (args !? #bot :: Maybe Int),
               ("left",) . showValue <$> (args !? #left :: Maybe Int),
               ("right",) . showValue <$> (args !? #right :: Maybe Int),
               ("type",) . showValue <$> (args !? #type :: Maybe Int),
               ("value",) . showValue <$> (args !? #value :: Maybe Double),
               ("values",) . showValue <$> (args !? #values :: Maybe [Double])]
        tensorArgs
          = catMaybes [("src",) <$> (args !? #src :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_cvcopyMakeBorder"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_cvcopyMakeBorder_upd ::
                      forall args . Fullfilled "_cvcopyMakeBorder(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "_cvcopyMakeBorder(ndarray)" args -> IO ()
_cvcopyMakeBorder_upd outputs args
  = let scalarArgs
          = catMaybes
              [("top",) . showValue <$> (args !? #top :: Maybe Int),
               ("bot",) . showValue <$> (args !? #bot :: Maybe Int),
               ("left",) . showValue <$> (args !? #left :: Maybe Int),
               ("right",) . showValue <$> (args !? #right :: Maybe Int),
               ("type",) . showValue <$> (args !? #type :: Maybe Int),
               ("value",) . showValue <$> (args !? #value :: Maybe Double),
               ("values",) . showValue <$> (args !? #values :: Maybe [Double])]
        tensorArgs
          = catMaybes [("src",) <$> (args !? #src :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_cvcopyMakeBorder"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_NoGradient(ndarray)" = '[]

_NoGradient ::
            forall args . Fullfilled "_NoGradient(ndarray)" args =>
              ArgsHMap "_NoGradient(ndarray)" args -> IO [NDArrayHandle]
_NoGradient args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_NoGradient"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_NoGradient_upd ::
                forall args . Fullfilled "_NoGradient(ndarray)" args =>
                  [NDArrayHandle] -> ArgsHMap "_NoGradient(ndarray)" args -> IO ()
_NoGradient_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_NoGradient"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_CachedOp(ndarray)" = '[]

_CachedOp ::
          forall args . Fullfilled "_CachedOp(ndarray)" args =>
            ArgsHMap "_CachedOp(ndarray)" args -> IO [NDArrayHandle]
_CachedOp args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_CachedOp"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_CachedOp_upd ::
              forall args . Fullfilled "_CachedOp(ndarray)" args =>
                [NDArrayHandle] -> ArgsHMap "_CachedOp(ndarray)" args -> IO ()
_CachedOp_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_CachedOp"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_CachedOp(ndarray)" = '[]

_backward_CachedOp ::
                   forall args . Fullfilled "_backward_CachedOp(ndarray)" args =>
                     ArgsHMap "_backward_CachedOp(ndarray)" args -> IO [NDArrayHandle]
_backward_CachedOp args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_CachedOp"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_CachedOp_upd ::
                       forall args . Fullfilled "_backward_CachedOp(ndarray)" args =>
                         [NDArrayHandle] ->
                           ArgsHMap "_backward_CachedOp(ndarray)" args -> IO ()
_backward_CachedOp_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_CachedOp"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_copyto(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle)]

_copyto ::
        forall args . Fullfilled "_copyto(ndarray)" args =>
          ArgsHMap "_copyto(ndarray)" args -> IO [NDArrayHandle]
_copyto args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_copyto"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_copyto_upd ::
            forall args . Fullfilled "_copyto(ndarray)" args =>
              [NDArrayHandle] -> ArgsHMap "_copyto(ndarray)" args -> IO ()
_copyto_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_copyto"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_SwapAxis(ndarray)" =
     '[ '("dim1", AttrOpt Int), '("dim2", AttrOpt Int),
        '("data", AttrOpt NDArrayHandle)]

_SwapAxis ::
          forall args . Fullfilled "_SwapAxis(ndarray)" args =>
            ArgsHMap "_SwapAxis(ndarray)" args -> IO [NDArrayHandle]
_SwapAxis args
  = let scalarArgs
          = catMaybes
              [("dim1",) . showValue <$> (args !? #dim1 :: Maybe Int),
               ("dim2",) . showValue <$> (args !? #dim2 :: Maybe Int)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SwapAxis"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_SwapAxis_upd ::
              forall args . Fullfilled "_SwapAxis(ndarray)" args =>
                [NDArrayHandle] -> ArgsHMap "_SwapAxis(ndarray)" args -> IO ()
_SwapAxis_upd outputs args
  = let scalarArgs
          = catMaybes
              [("dim1",) . showValue <$> (args !? #dim1 :: Maybe Int),
               ("dim2",) . showValue <$> (args !? #dim2 :: Maybe Int)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SwapAxis"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_Pad(ndarray)" =
     '[ '("mode", AttrReq (EnumType '["constant", "edge", "reflect"])),
        '("pad_width", AttrReq [Int]), '("constant_value", AttrOpt Double),
        '("data", AttrOpt NDArrayHandle)]

_Pad ::
     forall args . Fullfilled "_Pad(ndarray)" args =>
       ArgsHMap "_Pad(ndarray)" args -> IO [NDArrayHandle]
_Pad args
  = let scalarArgs
          = catMaybes
              [("mode",) . showValue <$>
                 (args !? #mode ::
                    Maybe (EnumType '["constant", "edge", "reflect"])),
               ("pad_width",) . showValue <$> (args !? #pad_width :: Maybe [Int]),
               ("constant_value",) . showValue <$>
                 (args !? #constant_value :: Maybe Double)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Pad"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_Pad_upd ::
         forall args . Fullfilled "_Pad(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "_Pad(ndarray)" args -> IO ()
_Pad_upd outputs args
  = let scalarArgs
          = catMaybes
              [("mode",) . showValue <$>
                 (args !? #mode ::
                    Maybe (EnumType '["constant", "edge", "reflect"])),
               ("pad_width",) . showValue <$> (args !? #pad_width :: Maybe [Int]),
               ("constant_value",) . showValue <$>
                 (args !? #constant_value :: Maybe Double)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Pad"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_BatchNorm_v1(ndarray)" =
     '[ '("eps", AttrOpt Float), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool),
        '("data", AttrOpt NDArrayHandle),
        '("gamma", AttrOpt NDArrayHandle),
        '("beta", AttrOpt NDArrayHandle)]

_BatchNorm_v1 ::
              forall args . Fullfilled "_BatchNorm_v1(ndarray)" args =>
                ArgsHMap "_BatchNorm_v1(ndarray)" args -> IO [NDArrayHandle]
_BatchNorm_v1 args
  = let scalarArgs
          = catMaybes
              [("eps",) . showValue <$> (args !? #eps :: Maybe Float),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float),
               ("fix_gamma",) . showValue <$> (args !? #fix_gamma :: Maybe Bool),
               ("use_global_stats",) . showValue <$>
                 (args !? #use_global_stats :: Maybe Bool),
               ("output_mean_var",) . showValue <$>
                 (args !? #output_mean_var :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("gamma",) <$> (args !? #gamma :: Maybe NDArrayHandle),
               ("beta",) <$> (args !? #beta :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "BatchNorm_v1"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_BatchNorm_v1_upd ::
                  forall args . Fullfilled "_BatchNorm_v1(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_BatchNorm_v1(ndarray)" args -> IO ()
_BatchNorm_v1_upd outputs args
  = let scalarArgs
          = catMaybes
              [("eps",) . showValue <$> (args !? #eps :: Maybe Float),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float),
               ("fix_gamma",) . showValue <$> (args !? #fix_gamma :: Maybe Bool),
               ("use_global_stats",) . showValue <$>
                 (args !? #use_global_stats :: Maybe Bool),
               ("output_mean_var",) . showValue <$>
                 (args !? #output_mean_var :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("gamma",) <$> (args !? #gamma :: Maybe NDArrayHandle),
               ("beta",) <$> (args !? #beta :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "BatchNorm_v1"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "sgd_update(ndarray)" =
     '[ '("lr", AttrReq Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("weight", AttrOpt NDArrayHandle),
        '("grad", AttrOpt NDArrayHandle)]

sgd_update ::
           forall args . Fullfilled "sgd_update(ndarray)" args =>
             ArgsHMap "sgd_update(ndarray)" args -> IO [NDArrayHandle]
sgd_update args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("grad",) <$> (args !? #grad :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sgd_update"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

sgd_update_upd ::
               forall args . Fullfilled "sgd_update(ndarray)" args =>
                 [NDArrayHandle] -> ArgsHMap "sgd_update(ndarray)" args -> IO ()
sgd_update_upd outputs args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("grad",) <$> (args !? #grad :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sgd_update"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "sgd_mom_update(ndarray)" =
     '[ '("lr", AttrReq Float), '("momentum", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("weight", AttrOpt NDArrayHandle),
        '("grad", AttrOpt NDArrayHandle), '("mom", AttrOpt NDArrayHandle)]

sgd_mom_update ::
               forall args . Fullfilled "sgd_mom_update(ndarray)" args =>
                 ArgsHMap "sgd_mom_update(ndarray)" args -> IO [NDArrayHandle]
sgd_mom_update args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("grad",) <$> (args !? #grad :: Maybe NDArrayHandle),
               ("mom",) <$> (args !? #mom :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sgd_mom_update"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

sgd_mom_update_upd ::
                   forall args . Fullfilled "sgd_mom_update(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "sgd_mom_update(ndarray)" args -> IO ()
sgd_mom_update_upd outputs args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("grad",) <$> (args !? #grad :: Maybe NDArrayHandle),
               ("mom",) <$> (args !? #mom :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sgd_mom_update"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "mp_sgd_update(ndarray)" =
     '[ '("lr", AttrReq Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("weight", AttrOpt NDArrayHandle),
        '("grad", AttrOpt NDArrayHandle),
        '("weight32", AttrOpt NDArrayHandle)]

mp_sgd_update ::
              forall args . Fullfilled "mp_sgd_update(ndarray)" args =>
                ArgsHMap "mp_sgd_update(ndarray)" args -> IO [NDArrayHandle]
mp_sgd_update args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("grad",) <$> (args !? #grad :: Maybe NDArrayHandle),
               ("weight32",) <$> (args !? #weight32 :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "mp_sgd_update"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

mp_sgd_update_upd ::
                  forall args . Fullfilled "mp_sgd_update(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "mp_sgd_update(ndarray)" args -> IO ()
mp_sgd_update_upd outputs args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("grad",) <$> (args !? #grad :: Maybe NDArrayHandle),
               ("weight32",) <$> (args !? #weight32 :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "mp_sgd_update"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "mp_sgd_mom_update(ndarray)" =
     '[ '("lr", AttrReq Float), '("momentum", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("weight", AttrOpt NDArrayHandle),
        '("grad", AttrOpt NDArrayHandle), '("mom", AttrOpt NDArrayHandle),
        '("weight32", AttrOpt NDArrayHandle)]

mp_sgd_mom_update ::
                  forall args . Fullfilled "mp_sgd_mom_update(ndarray)" args =>
                    ArgsHMap "mp_sgd_mom_update(ndarray)" args -> IO [NDArrayHandle]
mp_sgd_mom_update args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("grad",) <$> (args !? #grad :: Maybe NDArrayHandle),
               ("mom",) <$> (args !? #mom :: Maybe NDArrayHandle),
               ("weight32",) <$> (args !? #weight32 :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "mp_sgd_mom_update"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

mp_sgd_mom_update_upd ::
                      forall args . Fullfilled "mp_sgd_mom_update(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "mp_sgd_mom_update(ndarray)" args -> IO ()
mp_sgd_mom_update_upd outputs args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("grad",) <$> (args !? #grad :: Maybe NDArrayHandle),
               ("mom",) <$> (args !? #mom :: Maybe NDArrayHandle),
               ("weight32",) <$> (args !? #weight32 :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "mp_sgd_mom_update"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "adam_update(ndarray)" =
     '[ '("lr", AttrReq Float), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("weight", AttrOpt NDArrayHandle),
        '("grad", AttrOpt NDArrayHandle), '("mean", AttrOpt NDArrayHandle),
        '("var", AttrOpt NDArrayHandle)]

adam_update ::
            forall args . Fullfilled "adam_update(ndarray)" args =>
              ArgsHMap "adam_update(ndarray)" args -> IO [NDArrayHandle]
adam_update args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("beta1",) . showValue <$> (args !? #beta1 :: Maybe Float),
               ("beta2",) . showValue <$> (args !? #beta2 :: Maybe Float),
               ("epsilon",) . showValue <$> (args !? #epsilon :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("grad",) <$> (args !? #grad :: Maybe NDArrayHandle),
               ("mean",) <$> (args !? #mean :: Maybe NDArrayHandle),
               ("var",) <$> (args !? #var :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "adam_update"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

adam_update_upd ::
                forall args . Fullfilled "adam_update(ndarray)" args =>
                  [NDArrayHandle] -> ArgsHMap "adam_update(ndarray)" args -> IO ()
adam_update_upd outputs args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("beta1",) . showValue <$> (args !? #beta1 :: Maybe Float),
               ("beta2",) . showValue <$> (args !? #beta2 :: Maybe Float),
               ("epsilon",) . showValue <$> (args !? #epsilon :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("grad",) <$> (args !? #grad :: Maybe NDArrayHandle),
               ("mean",) <$> (args !? #mean :: Maybe NDArrayHandle),
               ("var",) <$> (args !? #var :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "adam_update"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "rmsprop_update(ndarray)" =
     '[ '("lr", AttrReq Float), '("gamma1", AttrOpt Float),
        '("epsilon", AttrOpt Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("clip_weights", AttrOpt Float),
        '("weight", AttrOpt NDArrayHandle),
        '("grad", AttrOpt NDArrayHandle), '("n", AttrOpt NDArrayHandle)]

rmsprop_update ::
               forall args . Fullfilled "rmsprop_update(ndarray)" args =>
                 ArgsHMap "rmsprop_update(ndarray)" args -> IO [NDArrayHandle]
rmsprop_update args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("gamma1",) . showValue <$> (args !? #gamma1 :: Maybe Float),
               ("epsilon",) . showValue <$> (args !? #epsilon :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("clip_weights",) . showValue <$>
                 (args !? #clip_weights :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("grad",) <$> (args !? #grad :: Maybe NDArrayHandle),
               ("n",) <$> (args !? #n :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "rmsprop_update"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

rmsprop_update_upd ::
                   forall args . Fullfilled "rmsprop_update(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "rmsprop_update(ndarray)" args -> IO ()
rmsprop_update_upd outputs args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("gamma1",) . showValue <$> (args !? #gamma1 :: Maybe Float),
               ("epsilon",) . showValue <$> (args !? #epsilon :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("clip_weights",) . showValue <$>
                 (args !? #clip_weights :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("grad",) <$> (args !? #grad :: Maybe NDArrayHandle),
               ("n",) <$> (args !? #n :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "rmsprop_update"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "rmspropalex_update(ndarray)" =
     '[ '("lr", AttrReq Float), '("gamma1", AttrOpt Float),
        '("gamma2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("clip_weights", AttrOpt Float),
        '("weight", AttrOpt NDArrayHandle),
        '("grad", AttrOpt NDArrayHandle), '("n", AttrOpt NDArrayHandle),
        '("g", AttrOpt NDArrayHandle), '("delta", AttrOpt NDArrayHandle)]

rmspropalex_update ::
                   forall args . Fullfilled "rmspropalex_update(ndarray)" args =>
                     ArgsHMap "rmspropalex_update(ndarray)" args -> IO [NDArrayHandle]
rmspropalex_update args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("gamma1",) . showValue <$> (args !? #gamma1 :: Maybe Float),
               ("gamma2",) . showValue <$> (args !? #gamma2 :: Maybe Float),
               ("epsilon",) . showValue <$> (args !? #epsilon :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("clip_weights",) . showValue <$>
                 (args !? #clip_weights :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("grad",) <$> (args !? #grad :: Maybe NDArrayHandle),
               ("n",) <$> (args !? #n :: Maybe NDArrayHandle),
               ("g",) <$> (args !? #g :: Maybe NDArrayHandle),
               ("delta",) <$> (args !? #delta :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "rmspropalex_update"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

rmspropalex_update_upd ::
                       forall args . Fullfilled "rmspropalex_update(ndarray)" args =>
                         [NDArrayHandle] ->
                           ArgsHMap "rmspropalex_update(ndarray)" args -> IO ()
rmspropalex_update_upd outputs args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("gamma1",) . showValue <$> (args !? #gamma1 :: Maybe Float),
               ("gamma2",) . showValue <$> (args !? #gamma2 :: Maybe Float),
               ("epsilon",) . showValue <$> (args !? #epsilon :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("clip_weights",) . showValue <$>
                 (args !? #clip_weights :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("grad",) <$> (args !? #grad :: Maybe NDArrayHandle),
               ("n",) <$> (args !? #n :: Maybe NDArrayHandle),
               ("g",) <$> (args !? #g :: Maybe NDArrayHandle),
               ("delta",) <$> (args !? #delta :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "rmspropalex_update"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "ftrl_update(ndarray)" =
     '[ '("lr", AttrReq Float), '("lamda1", AttrOpt Float),
        '("beta", AttrOpt Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("weight", AttrOpt NDArrayHandle),
        '("grad", AttrOpt NDArrayHandle), '("z", AttrOpt NDArrayHandle),
        '("n", AttrOpt NDArrayHandle)]

ftrl_update ::
            forall args . Fullfilled "ftrl_update(ndarray)" args =>
              ArgsHMap "ftrl_update(ndarray)" args -> IO [NDArrayHandle]
ftrl_update args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("lamda1",) . showValue <$> (args !? #lamda1 :: Maybe Float),
               ("beta",) . showValue <$> (args !? #beta :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("grad",) <$> (args !? #grad :: Maybe NDArrayHandle),
               ("z",) <$> (args !? #z :: Maybe NDArrayHandle),
               ("n",) <$> (args !? #n :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "ftrl_update"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

ftrl_update_upd ::
                forall args . Fullfilled "ftrl_update(ndarray)" args =>
                  [NDArrayHandle] -> ArgsHMap "ftrl_update(ndarray)" args -> IO ()
ftrl_update_upd outputs args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("lamda1",) . showValue <$> (args !? #lamda1 :: Maybe Float),
               ("beta",) . showValue <$> (args !? #beta :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("grad",) <$> (args !? #grad :: Maybe NDArrayHandle),
               ("z",) <$> (args !? #z :: Maybe NDArrayHandle),
               ("n",) <$> (args !? #n :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "ftrl_update"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_LeakyReLU(ndarray)" =
     '[ '("act_type",
          AttrOpt (EnumType '["elu", "leaky", "prelu", "rrelu"])),
        '("slope", AttrOpt Float), '("lower_bound", AttrOpt Float),
        '("upper_bound", AttrOpt Float), '("data", AttrOpt NDArrayHandle)]

_LeakyReLU ::
           forall args . Fullfilled "_LeakyReLU(ndarray)" args =>
             ArgsHMap "_LeakyReLU(ndarray)" args -> IO [NDArrayHandle]
_LeakyReLU args
  = let scalarArgs
          = catMaybes
              [("act_type",) . showValue <$>
                 (args !? #act_type ::
                    Maybe (EnumType '["elu", "leaky", "prelu", "rrelu"])),
               ("slope",) . showValue <$> (args !? #slope :: Maybe Float),
               ("lower_bound",) . showValue <$>
                 (args !? #lower_bound :: Maybe Float),
               ("upper_bound",) . showValue <$>
                 (args !? #upper_bound :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "LeakyReLU"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_LeakyReLU_upd ::
               forall args . Fullfilled "_LeakyReLU(ndarray)" args =>
                 [NDArrayHandle] -> ArgsHMap "_LeakyReLU(ndarray)" args -> IO ()
_LeakyReLU_upd outputs args
  = let scalarArgs
          = catMaybes
              [("act_type",) . showValue <$>
                 (args !? #act_type ::
                    Maybe (EnumType '["elu", "leaky", "prelu", "rrelu"])),
               ("slope",) . showValue <$> (args !? #slope :: Maybe Float),
               ("lower_bound",) . showValue <$>
                 (args !? #lower_bound :: Maybe Float),
               ("upper_bound",) . showValue <$>
                 (args !? #upper_bound :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "LeakyReLU"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_IdentityAttachKLSparseReg(ndarray)" =
     '[ '("sparseness_target", AttrOpt Float),
        '("penalty", AttrOpt Float), '("momentum", AttrOpt Float),
        '("data", AttrOpt NDArrayHandle)]

_IdentityAttachKLSparseReg ::
                           forall args .
                             Fullfilled "_IdentityAttachKLSparseReg(ndarray)" args =>
                             ArgsHMap "_IdentityAttachKLSparseReg(ndarray)" args ->
                               IO [NDArrayHandle]
_IdentityAttachKLSparseReg args
  = let scalarArgs
          = catMaybes
              [("sparseness_target",) . showValue <$>
                 (args !? #sparseness_target :: Maybe Float),
               ("penalty",) . showValue <$> (args !? #penalty :: Maybe Float),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "IdentityAttachKLSparseReg"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_IdentityAttachKLSparseReg_upd ::
                               forall args .
                                 Fullfilled "_IdentityAttachKLSparseReg(ndarray)" args =>
                                 [NDArrayHandle] ->
                                   ArgsHMap "_IdentityAttachKLSparseReg(ndarray)" args -> IO ()
_IdentityAttachKLSparseReg_upd outputs args
  = let scalarArgs
          = catMaybes
              [("sparseness_target",) . showValue <$>
                 (args !? #sparseness_target :: Maybe Float),
               ("penalty",) . showValue <$> (args !? #penalty :: Maybe Float),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "IdentityAttachKLSparseReg"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_UpSampling(ndarray)" =
     '[ '("scale", AttrReq Int), '("num_filter", AttrOpt Int),
        '("sample_type", AttrReq (EnumType '["bilinear", "nearest"])),
        '("multi_input_mode", AttrOpt (EnumType '["concat", "sum"])),
        '("num_args", AttrReq Int), '("workspace", AttrOpt Int),
        '("data", AttrOpt [NDArrayHandle])]

_UpSampling ::
            forall args . Fullfilled "_UpSampling(ndarray)" args =>
              ArgsHMap "_UpSampling(ndarray)" args -> IO [NDArrayHandle]
_UpSampling args
  = let scalarArgs
          = catMaybes
              [("scale",) . showValue <$> (args !? #scale :: Maybe Int),
               ("num_filter",) . showValue <$> (args !? #num_filter :: Maybe Int),
               ("sample_type",) . showValue <$>
                 (args !? #sample_type ::
                    Maybe (EnumType '["bilinear", "nearest"])),
               ("multi_input_mode",) . showValue <$>
                 (args !? #multi_input_mode :: Maybe (EnumType '["concat", "sum"])),
               ("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("workspace",) . showValue <$> (args !? #workspace :: Maybe Int)]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
        array = fromMaybe [] (args !? #data :: Maybe [NDArrayHandle])
      in
      do op <- nnGetOpHandle "UpSampling"
         let scalarArgs'
               = if hasKey args #num_args then scalarArgs else
                   (,) "num_args" (showValue (length array)) : scalarArgs
         listndarr <- mxImperativeInvoke (fromOpHandle op) array scalarArgs'
                        Nothing
         return listndarr

_UpSampling_upd ::
                forall args . Fullfilled "_UpSampling(ndarray)" args =>
                  [NDArrayHandle] -> ArgsHMap "_UpSampling(ndarray)" args -> IO ()
_UpSampling_upd outputs args
  = let scalarArgs
          = catMaybes
              [("scale",) . showValue <$> (args !? #scale :: Maybe Int),
               ("num_filter",) . showValue <$> (args !? #num_filter :: Maybe Int),
               ("sample_type",) . showValue <$>
                 (args !? #sample_type ::
                    Maybe (EnumType '["bilinear", "nearest"])),
               ("multi_input_mode",) . showValue <$>
                 (args !? #multi_input_mode :: Maybe (EnumType '["concat", "sum"])),
               ("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("workspace",) . showValue <$> (args !? #workspace :: Maybe Int)]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
        array = fromMaybe [] (args !? #data :: Maybe [NDArrayHandle])
      in
      do op <- nnGetOpHandle "UpSampling"
         let scalarArgs'
               = if hasKey args #num_args then scalarArgs else
                   (,) "num_args" (showValue (length array)) : scalarArgs
         listndarr <- mxImperativeInvoke (fromOpHandle op) array scalarArgs'
                        (Just outputs)
         return ()

type instance ParameterList "_SliceChannel(ndarray)" =
     '[ '("num_outputs", AttrReq Int), '("axis", AttrOpt Int),
        '("squeeze_axis", AttrOpt Bool), '("data", AttrOpt NDArrayHandle)]

_SliceChannel ::
              forall args . Fullfilled "_SliceChannel(ndarray)" args =>
                ArgsHMap "_SliceChannel(ndarray)" args -> IO [NDArrayHandle]
_SliceChannel args
  = let scalarArgs
          = catMaybes
              [("num_outputs",) . showValue <$>
                 (args !? #num_outputs :: Maybe Int),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("squeeze_axis",) . showValue <$>
                 (args !? #squeeze_axis :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SliceChannel"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_SliceChannel_upd ::
                  forall args . Fullfilled "_SliceChannel(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_SliceChannel(ndarray)" args -> IO ()
_SliceChannel_upd outputs args
  = let scalarArgs
          = catMaybes
              [("num_outputs",) . showValue <$>
                 (args !? #num_outputs :: Maybe Int),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("squeeze_axis",) . showValue <$>
                 (args !? #squeeze_axis :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SliceChannel"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_CuDNNBatchNorm(ndarray)" =
     '[ '("eps", AttrOpt Double), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool), '("axis", AttrOpt Int),
        '("cudnn_off", AttrOpt Bool), '("data", AttrOpt NDArrayHandle)]

_CuDNNBatchNorm ::
                forall args . Fullfilled "_CuDNNBatchNorm(ndarray)" args =>
                  ArgsHMap "_CuDNNBatchNorm(ndarray)" args -> IO [NDArrayHandle]
_CuDNNBatchNorm args
  = let scalarArgs
          = catMaybes
              [("eps",) . showValue <$> (args !? #eps :: Maybe Double),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float),
               ("fix_gamma",) . showValue <$> (args !? #fix_gamma :: Maybe Bool),
               ("use_global_stats",) . showValue <$>
                 (args !? #use_global_stats :: Maybe Bool),
               ("output_mean_var",) . showValue <$>
                 (args !? #output_mean_var :: Maybe Bool),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("cudnn_off",) . showValue <$> (args !? #cudnn_off :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "CuDNNBatchNorm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_CuDNNBatchNorm_upd ::
                    forall args . Fullfilled "_CuDNNBatchNorm(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_CuDNNBatchNorm(ndarray)" args -> IO ()
_CuDNNBatchNorm_upd outputs args
  = let scalarArgs
          = catMaybes
              [("eps",) . showValue <$> (args !? #eps :: Maybe Double),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float),
               ("fix_gamma",) . showValue <$> (args !? #fix_gamma :: Maybe Bool),
               ("use_global_stats",) . showValue <$>
                 (args !? #use_global_stats :: Maybe Bool),
               ("output_mean_var",) . showValue <$>
                 (args !? #output_mean_var :: Maybe Bool),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("cudnn_off",) . showValue <$> (args !? #cudnn_off :: Maybe Bool)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "CuDNNBatchNorm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "softmax_cross_entropy(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle),
        '("label", AttrOpt NDArrayHandle)]

softmax_cross_entropy ::
                      forall args . Fullfilled "softmax_cross_entropy(ndarray)" args =>
                        ArgsHMap "softmax_cross_entropy(ndarray)" args ->
                          IO [NDArrayHandle]
softmax_cross_entropy args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("label",) <$> (args !? #label :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "softmax_cross_entropy"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

softmax_cross_entropy_upd ::
                          forall args . Fullfilled "softmax_cross_entropy(ndarray)" args =>
                            [NDArrayHandle] ->
                              ArgsHMap "softmax_cross_entropy(ndarray)" args -> IO ()
softmax_cross_entropy_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("label",) <$> (args !? #label :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "softmax_cross_entropy"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance
     ParameterList "_backward_softmax_cross_entropy(ndarray)" = '[]

_backward_softmax_cross_entropy ::
                                forall args .
                                  Fullfilled "_backward_softmax_cross_entropy(ndarray)" args =>
                                  ArgsHMap "_backward_softmax_cross_entropy(ndarray)" args ->
                                    IO [NDArrayHandle]
_backward_softmax_cross_entropy args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_softmax_cross_entropy"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_softmax_cross_entropy_upd ::
                                    forall args .
                                      Fullfilled "_backward_softmax_cross_entropy(ndarray)" args =>
                                      [NDArrayHandle] ->
                                        ArgsHMap "_backward_softmax_cross_entropy(ndarray)" args ->
                                          IO ()
_backward_softmax_cross_entropy_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_softmax_cross_entropy"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_Concat(ndarray)" =
     '[ '("num_args", AttrReq Int), '("dim", AttrOpt Int),
        '("data", AttrOpt [NDArrayHandle])]

_Concat ::
        forall args . Fullfilled "_Concat(ndarray)" args =>
          ArgsHMap "_Concat(ndarray)" args -> IO [NDArrayHandle]
_Concat args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("dim",) . showValue <$> (args !? #dim :: Maybe Int)]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
        array = fromMaybe [] (args !? #data :: Maybe [NDArrayHandle])
      in
      do op <- nnGetOpHandle "Concat"
         let scalarArgs'
               = if hasKey args #num_args then scalarArgs else
                   (,) "num_args" (showValue (length array)) : scalarArgs
         listndarr <- mxImperativeInvoke (fromOpHandle op) array scalarArgs'
                        Nothing
         return listndarr

_Concat_upd ::
            forall args . Fullfilled "_Concat(ndarray)" args =>
              [NDArrayHandle] -> ArgsHMap "_Concat(ndarray)" args -> IO ()
_Concat_upd outputs args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("dim",) . showValue <$> (args !? #dim :: Maybe Int)]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
        array = fromMaybe [] (args !? #data :: Maybe [NDArrayHandle])
      in
      do op <- nnGetOpHandle "Concat"
         let scalarArgs'
               = if hasKey args #num_args then scalarArgs else
                   (,) "num_args" (showValue (length array)) : scalarArgs
         listndarr <- mxImperativeInvoke (fromOpHandle op) array scalarArgs'
                        (Just outputs)
         return ()

type instance ParameterList "_BatchNorm(ndarray)" =
     '[ '("eps", AttrOpt Double), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool), '("axis", AttrOpt Int),
        '("cudnn_off", AttrOpt Bool), '("data", AttrOpt NDArrayHandle),
        '("gamma", AttrOpt NDArrayHandle),
        '("beta", AttrOpt NDArrayHandle),
        '("moving_mean", AttrOpt NDArrayHandle),
        '("moving_var", AttrOpt NDArrayHandle)]

_BatchNorm ::
           forall args . Fullfilled "_BatchNorm(ndarray)" args =>
             ArgsHMap "_BatchNorm(ndarray)" args -> IO [NDArrayHandle]
_BatchNorm args
  = let scalarArgs
          = catMaybes
              [("eps",) . showValue <$> (args !? #eps :: Maybe Double),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float),
               ("fix_gamma",) . showValue <$> (args !? #fix_gamma :: Maybe Bool),
               ("use_global_stats",) . showValue <$>
                 (args !? #use_global_stats :: Maybe Bool),
               ("output_mean_var",) . showValue <$>
                 (args !? #output_mean_var :: Maybe Bool),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("cudnn_off",) . showValue <$> (args !? #cudnn_off :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("gamma",) <$> (args !? #gamma :: Maybe NDArrayHandle),
               ("beta",) <$> (args !? #beta :: Maybe NDArrayHandle),
               ("moving_mean",) <$> (args !? #moving_mean :: Maybe NDArrayHandle),
               ("moving_var",) <$> (args !? #moving_var :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "BatchNorm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_BatchNorm_upd ::
               forall args . Fullfilled "_BatchNorm(ndarray)" args =>
                 [NDArrayHandle] -> ArgsHMap "_BatchNorm(ndarray)" args -> IO ()
_BatchNorm_upd outputs args
  = let scalarArgs
          = catMaybes
              [("eps",) . showValue <$> (args !? #eps :: Maybe Double),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float),
               ("fix_gamma",) . showValue <$> (args !? #fix_gamma :: Maybe Bool),
               ("use_global_stats",) . showValue <$>
                 (args !? #use_global_stats :: Maybe Bool),
               ("output_mean_var",) . showValue <$>
                 (args !? #output_mean_var :: Maybe Bool),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("cudnn_off",) . showValue <$> (args !? #cudnn_off :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("gamma",) <$> (args !? #gamma :: Maybe NDArrayHandle),
               ("beta",) <$> (args !? #beta :: Maybe NDArrayHandle),
               ("moving_mean",) <$> (args !? #moving_mean :: Maybe NDArrayHandle),
               ("moving_var",) <$> (args !? #moving_var :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "BatchNorm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_CustomFunction(ndarray)" = '[]

_CustomFunction ::
                forall args . Fullfilled "_CustomFunction(ndarray)" args =>
                  ArgsHMap "_CustomFunction(ndarray)" args -> IO [NDArrayHandle]
_CustomFunction args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_CustomFunction"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_CustomFunction_upd ::
                    forall args . Fullfilled "_CustomFunction(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_CustomFunction(ndarray)" args -> IO ()
_CustomFunction_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_CustomFunction"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_CustomFunction(ndarray)" =
     '[]

_backward_CustomFunction ::
                         forall args .
                           Fullfilled "_backward_CustomFunction(ndarray)" args =>
                           ArgsHMap "_backward_CustomFunction(ndarray)" args ->
                             IO [NDArrayHandle]
_backward_CustomFunction args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_CustomFunction"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_CustomFunction_upd ::
                             forall args .
                               Fullfilled "_backward_CustomFunction(ndarray)" args =>
                               [NDArrayHandle] ->
                                 ArgsHMap "_backward_CustomFunction(ndarray)" args -> IO ()
_backward_CustomFunction_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_CustomFunction"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_contrib_MultiBoxTarget(ndarray)" =
     '[ '("overlap_threshold", AttrOpt Float),
        '("ignore_label", AttrOpt Float),
        '("negative_mining_ratio", AttrOpt Float),
        '("negative_mining_thresh", AttrOpt Float),
        '("minimum_negative_samples", AttrOpt Int),
        '("variances", AttrOpt [Float]),
        '("anchor", AttrOpt NDArrayHandle),
        '("label", AttrOpt NDArrayHandle),
        '("cls_pred", AttrOpt NDArrayHandle)]

_contrib_MultiBoxTarget ::
                        forall args . Fullfilled "_contrib_MultiBoxTarget(ndarray)" args =>
                          ArgsHMap "_contrib_MultiBoxTarget(ndarray)" args ->
                            IO [NDArrayHandle]
_contrib_MultiBoxTarget args
  = let scalarArgs
          = catMaybes
              [("overlap_threshold",) . showValue <$>
                 (args !? #overlap_threshold :: Maybe Float),
               ("ignore_label",) . showValue <$>
                 (args !? #ignore_label :: Maybe Float),
               ("negative_mining_ratio",) . showValue <$>
                 (args !? #negative_mining_ratio :: Maybe Float),
               ("negative_mining_thresh",) . showValue <$>
                 (args !? #negative_mining_thresh :: Maybe Float),
               ("minimum_negative_samples",) . showValue <$>
                 (args !? #minimum_negative_samples :: Maybe Int),
               ("variances",) . showValue <$>
                 (args !? #variances :: Maybe [Float])]
        tensorArgs
          = catMaybes
              [("anchor",) <$> (args !? #anchor :: Maybe NDArrayHandle),
               ("label",) <$> (args !? #label :: Maybe NDArrayHandle),
               ("cls_pred",) <$> (args !? #cls_pred :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_MultiBoxTarget"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_contrib_MultiBoxTarget_upd ::
                            forall args . Fullfilled "_contrib_MultiBoxTarget(ndarray)" args =>
                              [NDArrayHandle] ->
                                ArgsHMap "_contrib_MultiBoxTarget(ndarray)" args -> IO ()
_contrib_MultiBoxTarget_upd outputs args
  = let scalarArgs
          = catMaybes
              [("overlap_threshold",) . showValue <$>
                 (args !? #overlap_threshold :: Maybe Float),
               ("ignore_label",) . showValue <$>
                 (args !? #ignore_label :: Maybe Float),
               ("negative_mining_ratio",) . showValue <$>
                 (args !? #negative_mining_ratio :: Maybe Float),
               ("negative_mining_thresh",) . showValue <$>
                 (args !? #negative_mining_thresh :: Maybe Float),
               ("minimum_negative_samples",) . showValue <$>
                 (args !? #minimum_negative_samples :: Maybe Int),
               ("variances",) . showValue <$>
                 (args !? #variances :: Maybe [Float])]
        tensorArgs
          = catMaybes
              [("anchor",) <$> (args !? #anchor :: Maybe NDArrayHandle),
               ("label",) <$> (args !? #label :: Maybe NDArrayHandle),
               ("cls_pred",) <$> (args !? #cls_pred :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_MultiBoxTarget"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance
     ParameterList "_backward__contrib_MultiBoxTarget(ndarray)" = '[]

_backward__contrib_MultiBoxTarget ::
                                  forall args .
                                    Fullfilled "_backward__contrib_MultiBoxTarget(ndarray)" args =>
                                    ArgsHMap "_backward__contrib_MultiBoxTarget(ndarray)" args ->
                                      IO [NDArrayHandle]
_backward__contrib_MultiBoxTarget args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_MultiBoxTarget"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward__contrib_MultiBoxTarget_upd ::
                                      forall args .
                                        Fullfilled "_backward__contrib_MultiBoxTarget(ndarray)"
                                          args =>
                                        [NDArrayHandle] ->
                                          ArgsHMap "_backward__contrib_MultiBoxTarget(ndarray)" args
                                            -> IO ()
_backward__contrib_MultiBoxTarget_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_MultiBoxTarget"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_contrib_Proposal(ndarray)" =
     '[ '("rpn_pre_nms_top_n", AttrOpt Int),
        '("rpn_post_nms_top_n", AttrOpt Int),
        '("threshold", AttrOpt Float), '("rpn_min_size", AttrOpt Int),
        '("scales", AttrOpt [Float]), '("ratios", AttrOpt [Float]),
        '("feature_stride", AttrOpt Int), '("output_score", AttrOpt Bool),
        '("iou_loss", AttrOpt Bool), '("cls_score", AttrOpt NDArrayHandle),
        '("bbox_pred", AttrOpt NDArrayHandle),
        '("im_info", AttrOpt NDArrayHandle)]

_contrib_Proposal ::
                  forall args . Fullfilled "_contrib_Proposal(ndarray)" args =>
                    ArgsHMap "_contrib_Proposal(ndarray)" args -> IO [NDArrayHandle]
_contrib_Proposal args
  = let scalarArgs
          = catMaybes
              [("rpn_pre_nms_top_n",) . showValue <$>
                 (args !? #rpn_pre_nms_top_n :: Maybe Int),
               ("rpn_post_nms_top_n",) . showValue <$>
                 (args !? #rpn_post_nms_top_n :: Maybe Int),
               ("threshold",) . showValue <$> (args !? #threshold :: Maybe Float),
               ("rpn_min_size",) . showValue <$>
                 (args !? #rpn_min_size :: Maybe Int),
               ("scales",) . showValue <$> (args !? #scales :: Maybe [Float]),
               ("ratios",) . showValue <$> (args !? #ratios :: Maybe [Float]),
               ("feature_stride",) . showValue <$>
                 (args !? #feature_stride :: Maybe Int),
               ("output_score",) . showValue <$>
                 (args !? #output_score :: Maybe Bool),
               ("iou_loss",) . showValue <$> (args !? #iou_loss :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("cls_score",) <$> (args !? #cls_score :: Maybe NDArrayHandle),
               ("bbox_pred",) <$> (args !? #bbox_pred :: Maybe NDArrayHandle),
               ("im_info",) <$> (args !? #im_info :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_Proposal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_contrib_Proposal_upd ::
                      forall args . Fullfilled "_contrib_Proposal(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "_contrib_Proposal(ndarray)" args -> IO ()
_contrib_Proposal_upd outputs args
  = let scalarArgs
          = catMaybes
              [("rpn_pre_nms_top_n",) . showValue <$>
                 (args !? #rpn_pre_nms_top_n :: Maybe Int),
               ("rpn_post_nms_top_n",) . showValue <$>
                 (args !? #rpn_post_nms_top_n :: Maybe Int),
               ("threshold",) . showValue <$> (args !? #threshold :: Maybe Float),
               ("rpn_min_size",) . showValue <$>
                 (args !? #rpn_min_size :: Maybe Int),
               ("scales",) . showValue <$> (args !? #scales :: Maybe [Float]),
               ("ratios",) . showValue <$> (args !? #ratios :: Maybe [Float]),
               ("feature_stride",) . showValue <$>
                 (args !? #feature_stride :: Maybe Int),
               ("output_score",) . showValue <$>
                 (args !? #output_score :: Maybe Bool),
               ("iou_loss",) . showValue <$> (args !? #iou_loss :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("cls_score",) <$> (args !? #cls_score :: Maybe NDArrayHandle),
               ("bbox_pred",) <$> (args !? #bbox_pred :: Maybe NDArrayHandle),
               ("im_info",) <$> (args !? #im_info :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_Proposal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward__contrib_Proposal(ndarray)"
     = '[]

_backward__contrib_Proposal ::
                            forall args .
                              Fullfilled "_backward__contrib_Proposal(ndarray)" args =>
                              ArgsHMap "_backward__contrib_Proposal(ndarray)" args ->
                                IO [NDArrayHandle]
_backward__contrib_Proposal args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_Proposal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward__contrib_Proposal_upd ::
                                forall args .
                                  Fullfilled "_backward__contrib_Proposal(ndarray)" args =>
                                  [NDArrayHandle] ->
                                    ArgsHMap "_backward__contrib_Proposal(ndarray)" args -> IO ()
_backward__contrib_Proposal_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_Proposal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_contrib_count_sketch(ndarray)" =
     '[ '("out_dim", AttrReq Int),
        '("processing_batch_size", AttrOpt Int),
        '("data", AttrOpt NDArrayHandle), '("h", AttrOpt NDArrayHandle),
        '("s", AttrOpt NDArrayHandle)]

_contrib_count_sketch ::
                      forall args . Fullfilled "_contrib_count_sketch(ndarray)" args =>
                        ArgsHMap "_contrib_count_sketch(ndarray)" args ->
                          IO [NDArrayHandle]
_contrib_count_sketch args
  = let scalarArgs
          = catMaybes
              [("out_dim",) . showValue <$> (args !? #out_dim :: Maybe Int),
               ("processing_batch_size",) . showValue <$>
                 (args !? #processing_batch_size :: Maybe Int)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("h",) <$> (args !? #h :: Maybe NDArrayHandle),
               ("s",) <$> (args !? #s :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_count_sketch"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_contrib_count_sketch_upd ::
                          forall args . Fullfilled "_contrib_count_sketch(ndarray)" args =>
                            [NDArrayHandle] ->
                              ArgsHMap "_contrib_count_sketch(ndarray)" args -> IO ()
_contrib_count_sketch_upd outputs args
  = let scalarArgs
          = catMaybes
              [("out_dim",) . showValue <$> (args !? #out_dim :: Maybe Int),
               ("processing_batch_size",) . showValue <$>
                 (args !? #processing_batch_size :: Maybe Int)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("h",) <$> (args !? #h :: Maybe NDArrayHandle),
               ("s",) <$> (args !? #s :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_count_sketch"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance
     ParameterList "_backward__contrib_count_sketch(ndarray)" = '[]

_backward__contrib_count_sketch ::
                                forall args .
                                  Fullfilled "_backward__contrib_count_sketch(ndarray)" args =>
                                  ArgsHMap "_backward__contrib_count_sketch(ndarray)" args ->
                                    IO [NDArrayHandle]
_backward__contrib_count_sketch args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_count_sketch"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward__contrib_count_sketch_upd ::
                                    forall args .
                                      Fullfilled "_backward__contrib_count_sketch(ndarray)" args =>
                                      [NDArrayHandle] ->
                                        ArgsHMap "_backward__contrib_count_sketch(ndarray)" args ->
                                          IO ()
_backward__contrib_count_sketch_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_count_sketch"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_contrib_MultiBoxDetection(ndarray)" =
     '[ '("clip", AttrOpt Bool), '("threshold", AttrOpt Float),
        '("background_id", AttrOpt Int), '("nms_threshold", AttrOpt Float),
        '("force_suppress", AttrOpt Bool), '("variances", AttrOpt [Float]),
        '("nms_topk", AttrOpt Int), '("cls_prob", AttrOpt NDArrayHandle),
        '("loc_pred", AttrOpt NDArrayHandle),
        '("anchor", AttrOpt NDArrayHandle)]

_contrib_MultiBoxDetection ::
                           forall args .
                             Fullfilled "_contrib_MultiBoxDetection(ndarray)" args =>
                             ArgsHMap "_contrib_MultiBoxDetection(ndarray)" args ->
                               IO [NDArrayHandle]
_contrib_MultiBoxDetection args
  = let scalarArgs
          = catMaybes
              [("clip",) . showValue <$> (args !? #clip :: Maybe Bool),
               ("threshold",) . showValue <$> (args !? #threshold :: Maybe Float),
               ("background_id",) . showValue <$>
                 (args !? #background_id :: Maybe Int),
               ("nms_threshold",) . showValue <$>
                 (args !? #nms_threshold :: Maybe Float),
               ("force_suppress",) . showValue <$>
                 (args !? #force_suppress :: Maybe Bool),
               ("variances",) . showValue <$>
                 (args !? #variances :: Maybe [Float]),
               ("nms_topk",) . showValue <$> (args !? #nms_topk :: Maybe Int)]
        tensorArgs
          = catMaybes
              [("cls_prob",) <$> (args !? #cls_prob :: Maybe NDArrayHandle),
               ("loc_pred",) <$> (args !? #loc_pred :: Maybe NDArrayHandle),
               ("anchor",) <$> (args !? #anchor :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_MultiBoxDetection"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_contrib_MultiBoxDetection_upd ::
                               forall args .
                                 Fullfilled "_contrib_MultiBoxDetection(ndarray)" args =>
                                 [NDArrayHandle] ->
                                   ArgsHMap "_contrib_MultiBoxDetection(ndarray)" args -> IO ()
_contrib_MultiBoxDetection_upd outputs args
  = let scalarArgs
          = catMaybes
              [("clip",) . showValue <$> (args !? #clip :: Maybe Bool),
               ("threshold",) . showValue <$> (args !? #threshold :: Maybe Float),
               ("background_id",) . showValue <$>
                 (args !? #background_id :: Maybe Int),
               ("nms_threshold",) . showValue <$>
                 (args !? #nms_threshold :: Maybe Float),
               ("force_suppress",) . showValue <$>
                 (args !? #force_suppress :: Maybe Bool),
               ("variances",) . showValue <$>
                 (args !? #variances :: Maybe [Float]),
               ("nms_topk",) . showValue <$> (args !? #nms_topk :: Maybe Int)]
        tensorArgs
          = catMaybes
              [("cls_prob",) <$> (args !? #cls_prob :: Maybe NDArrayHandle),
               ("loc_pred",) <$> (args !? #loc_pred :: Maybe NDArrayHandle),
               ("anchor",) <$> (args !? #anchor :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_MultiBoxDetection"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance
     ParameterList "_backward__contrib_MultiBoxDetection(ndarray)" = '[]

_backward__contrib_MultiBoxDetection ::
                                     forall args .
                                       Fullfilled "_backward__contrib_MultiBoxDetection(ndarray)"
                                         args =>
                                       ArgsHMap "_backward__contrib_MultiBoxDetection(ndarray)" args
                                         -> IO [NDArrayHandle]
_backward__contrib_MultiBoxDetection args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_MultiBoxDetection"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward__contrib_MultiBoxDetection_upd ::
                                         forall args .
                                           Fullfilled
                                             "_backward__contrib_MultiBoxDetection(ndarray)" args =>
                                           [NDArrayHandle] ->
                                             ArgsHMap
                                               "_backward__contrib_MultiBoxDetection(ndarray)"
                                               args
                                               -> IO ()
_backward__contrib_MultiBoxDetection_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_MultiBoxDetection"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance
     ParameterList "_backward__contrib_PSROIPooling(ndarray)" = '[]

_backward__contrib_PSROIPooling ::
                                forall args .
                                  Fullfilled "_backward__contrib_PSROIPooling(ndarray)" args =>
                                  ArgsHMap "_backward__contrib_PSROIPooling(ndarray)" args ->
                                    IO [NDArrayHandle]
_backward__contrib_PSROIPooling args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_PSROIPooling"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward__contrib_PSROIPooling_upd ::
                                    forall args .
                                      Fullfilled "_backward__contrib_PSROIPooling(ndarray)" args =>
                                      [NDArrayHandle] ->
                                        ArgsHMap "_backward__contrib_PSROIPooling(ndarray)" args ->
                                          IO ()
_backward__contrib_PSROIPooling_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_PSROIPooling"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance
     ParameterList "_backward__contrib_DeformablePSROIPooling(ndarray)"
     = '[]

_backward__contrib_DeformablePSROIPooling ::
                                          forall args .
                                            Fullfilled
                                              "_backward__contrib_DeformablePSROIPooling(ndarray)"
                                              args =>
                                            ArgsHMap
                                              "_backward__contrib_DeformablePSROIPooling(ndarray)"
                                              args
                                              -> IO [NDArrayHandle]
_backward__contrib_DeformablePSROIPooling args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_DeformablePSROIPooling"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward__contrib_DeformablePSROIPooling_upd ::
                                              forall args .
                                                Fullfilled
                                                  "_backward__contrib_DeformablePSROIPooling(ndarray)"
                                                  args =>
                                                [NDArrayHandle] ->
                                                  ArgsHMap
                                                    "_backward__contrib_DeformablePSROIPooling(ndarray)"
                                                    args
                                                    -> IO ()
_backward__contrib_DeformablePSROIPooling_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_DeformablePSROIPooling"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward__contrib_CTCLoss(ndarray)" =
     '[]

_backward__contrib_CTCLoss ::
                           forall args .
                             Fullfilled "_backward__contrib_CTCLoss(ndarray)" args =>
                             ArgsHMap "_backward__contrib_CTCLoss(ndarray)" args ->
                               IO [NDArrayHandle]
_backward__contrib_CTCLoss args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_CTCLoss"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward__contrib_CTCLoss_upd ::
                               forall args .
                                 Fullfilled "_backward__contrib_CTCLoss(ndarray)" args =>
                                 [NDArrayHandle] ->
                                   ArgsHMap "_backward__contrib_CTCLoss(ndarray)" args -> IO ()
_backward__contrib_CTCLoss_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_CTCLoss"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_contrib_MultiProposal(ndarray)" =
     '[ '("rpn_pre_nms_top_n", AttrOpt Int),
        '("rpn_post_nms_top_n", AttrOpt Int),
        '("threshold", AttrOpt Float), '("rpn_min_size", AttrOpt Int),
        '("scales", AttrOpt [Float]), '("ratios", AttrOpt [Float]),
        '("feature_stride", AttrOpt Int), '("output_score", AttrOpt Bool),
        '("iou_loss", AttrOpt Bool), '("cls_score", AttrOpt NDArrayHandle),
        '("bbox_pred", AttrOpt NDArrayHandle),
        '("im_info", AttrOpt NDArrayHandle)]

_contrib_MultiProposal ::
                       forall args . Fullfilled "_contrib_MultiProposal(ndarray)" args =>
                         ArgsHMap "_contrib_MultiProposal(ndarray)" args ->
                           IO [NDArrayHandle]
_contrib_MultiProposal args
  = let scalarArgs
          = catMaybes
              [("rpn_pre_nms_top_n",) . showValue <$>
                 (args !? #rpn_pre_nms_top_n :: Maybe Int),
               ("rpn_post_nms_top_n",) . showValue <$>
                 (args !? #rpn_post_nms_top_n :: Maybe Int),
               ("threshold",) . showValue <$> (args !? #threshold :: Maybe Float),
               ("rpn_min_size",) . showValue <$>
                 (args !? #rpn_min_size :: Maybe Int),
               ("scales",) . showValue <$> (args !? #scales :: Maybe [Float]),
               ("ratios",) . showValue <$> (args !? #ratios :: Maybe [Float]),
               ("feature_stride",) . showValue <$>
                 (args !? #feature_stride :: Maybe Int),
               ("output_score",) . showValue <$>
                 (args !? #output_score :: Maybe Bool),
               ("iou_loss",) . showValue <$> (args !? #iou_loss :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("cls_score",) <$> (args !? #cls_score :: Maybe NDArrayHandle),
               ("bbox_pred",) <$> (args !? #bbox_pred :: Maybe NDArrayHandle),
               ("im_info",) <$> (args !? #im_info :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_MultiProposal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_contrib_MultiProposal_upd ::
                           forall args . Fullfilled "_contrib_MultiProposal(ndarray)" args =>
                             [NDArrayHandle] ->
                               ArgsHMap "_contrib_MultiProposal(ndarray)" args -> IO ()
_contrib_MultiProposal_upd outputs args
  = let scalarArgs
          = catMaybes
              [("rpn_pre_nms_top_n",) . showValue <$>
                 (args !? #rpn_pre_nms_top_n :: Maybe Int),
               ("rpn_post_nms_top_n",) . showValue <$>
                 (args !? #rpn_post_nms_top_n :: Maybe Int),
               ("threshold",) . showValue <$> (args !? #threshold :: Maybe Float),
               ("rpn_min_size",) . showValue <$>
                 (args !? #rpn_min_size :: Maybe Int),
               ("scales",) . showValue <$> (args !? #scales :: Maybe [Float]),
               ("ratios",) . showValue <$> (args !? #ratios :: Maybe [Float]),
               ("feature_stride",) . showValue <$>
                 (args !? #feature_stride :: Maybe Int),
               ("output_score",) . showValue <$>
                 (args !? #output_score :: Maybe Bool),
               ("iou_loss",) . showValue <$> (args !? #iou_loss :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("cls_score",) <$> (args !? #cls_score :: Maybe NDArrayHandle),
               ("bbox_pred",) <$> (args !? #bbox_pred :: Maybe NDArrayHandle),
               ("im_info",) <$> (args !? #im_info :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_MultiProposal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance
     ParameterList "_backward__contrib_MultiProposal(ndarray)" = '[]

_backward__contrib_MultiProposal ::
                                 forall args .
                                   Fullfilled "_backward__contrib_MultiProposal(ndarray)" args =>
                                   ArgsHMap "_backward__contrib_MultiProposal(ndarray)" args ->
                                     IO [NDArrayHandle]
_backward__contrib_MultiProposal args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_MultiProposal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward__contrib_MultiProposal_upd ::
                                     forall args .
                                       Fullfilled "_backward__contrib_MultiProposal(ndarray)"
                                         args =>
                                       [NDArrayHandle] ->
                                         ArgsHMap "_backward__contrib_MultiProposal(ndarray)" args
                                           -> IO ()
_backward__contrib_MultiProposal_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_MultiProposal"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_contrib_MultiBoxPrior(ndarray)" =
     '[ '("sizes", AttrOpt [Float]), '("ratios", AttrOpt [Float]),
        '("clip", AttrOpt Bool), '("steps", AttrOpt [Float]),
        '("offsets", AttrOpt [Float]), '("data", AttrOpt NDArrayHandle)]

_contrib_MultiBoxPrior ::
                       forall args . Fullfilled "_contrib_MultiBoxPrior(ndarray)" args =>
                         ArgsHMap "_contrib_MultiBoxPrior(ndarray)" args ->
                           IO [NDArrayHandle]
_contrib_MultiBoxPrior args
  = let scalarArgs
          = catMaybes
              [("sizes",) . showValue <$> (args !? #sizes :: Maybe [Float]),
               ("ratios",) . showValue <$> (args !? #ratios :: Maybe [Float]),
               ("clip",) . showValue <$> (args !? #clip :: Maybe Bool),
               ("steps",) . showValue <$> (args !? #steps :: Maybe [Float]),
               ("offsets",) . showValue <$> (args !? #offsets :: Maybe [Float])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_MultiBoxPrior"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_contrib_MultiBoxPrior_upd ::
                           forall args . Fullfilled "_contrib_MultiBoxPrior(ndarray)" args =>
                             [NDArrayHandle] ->
                               ArgsHMap "_contrib_MultiBoxPrior(ndarray)" args -> IO ()
_contrib_MultiBoxPrior_upd outputs args
  = let scalarArgs
          = catMaybes
              [("sizes",) . showValue <$> (args !? #sizes :: Maybe [Float]),
               ("ratios",) . showValue <$> (args !? #ratios :: Maybe [Float]),
               ("clip",) . showValue <$> (args !? #clip :: Maybe Bool),
               ("steps",) . showValue <$> (args !? #steps :: Maybe [Float]),
               ("offsets",) . showValue <$> (args !? #offsets :: Maybe [Float])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_MultiBoxPrior"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance
     ParameterList "_backward__contrib_MultiBoxPrior(ndarray)" = '[]

_backward__contrib_MultiBoxPrior ::
                                 forall args .
                                   Fullfilled "_backward__contrib_MultiBoxPrior(ndarray)" args =>
                                   ArgsHMap "_backward__contrib_MultiBoxPrior(ndarray)" args ->
                                     IO [NDArrayHandle]
_backward__contrib_MultiBoxPrior args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_MultiBoxPrior"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward__contrib_MultiBoxPrior_upd ::
                                     forall args .
                                       Fullfilled "_backward__contrib_MultiBoxPrior(ndarray)"
                                         args =>
                                       [NDArrayHandle] ->
                                         ArgsHMap "_backward__contrib_MultiBoxPrior(ndarray)" args
                                           -> IO ()
_backward__contrib_MultiBoxPrior_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_MultiBoxPrior"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_contrib_fft(ndarray)" =
     '[ '("compute_size", AttrOpt Int),
        '("data", AttrOpt NDArrayHandle)]

_contrib_fft ::
             forall args . Fullfilled "_contrib_fft(ndarray)" args =>
               ArgsHMap "_contrib_fft(ndarray)" args -> IO [NDArrayHandle]
_contrib_fft args
  = let scalarArgs
          = catMaybes
              [("compute_size",) . showValue <$>
                 (args !? #compute_size :: Maybe Int)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_fft"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_contrib_fft_upd ::
                 forall args . Fullfilled "_contrib_fft(ndarray)" args =>
                   [NDArrayHandle] -> ArgsHMap "_contrib_fft(ndarray)" args -> IO ()
_contrib_fft_upd outputs args
  = let scalarArgs
          = catMaybes
              [("compute_size",) . showValue <$>
                 (args !? #compute_size :: Maybe Int)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_fft"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward__contrib_fft(ndarray)" = '[]

_backward__contrib_fft ::
                       forall args . Fullfilled "_backward__contrib_fft(ndarray)" args =>
                         ArgsHMap "_backward__contrib_fft(ndarray)" args ->
                           IO [NDArrayHandle]
_backward__contrib_fft args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_fft"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward__contrib_fft_upd ::
                           forall args . Fullfilled "_backward__contrib_fft(ndarray)" args =>
                             [NDArrayHandle] ->
                               ArgsHMap "_backward__contrib_fft(ndarray)" args -> IO ()
_backward__contrib_fft_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_fft"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance
     ParameterList "_contrib_DeformableConvolution(ndarray)" =
     '[ '("kernel", AttrReq [Int]), '("stride", AttrOpt [Int]),
        '("dilate", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("num_filter", AttrReq Int), '("num_group", AttrOpt Int),
        '("num_deformable_group", AttrOpt Int),
        '("workspace", AttrOpt Int), '("no_bias", AttrOpt Bool),
        '("layout", AttrOpt (Maybe (EnumType '["NCDHW", "NCHW", "NCW"]))),
        '("data", AttrOpt NDArrayHandle),
        '("offset", AttrOpt NDArrayHandle),
        '("weight", AttrOpt NDArrayHandle),
        '("bias", AttrOpt NDArrayHandle)]

_contrib_DeformableConvolution ::
                               forall args .
                                 Fullfilled "_contrib_DeformableConvolution(ndarray)" args =>
                                 ArgsHMap "_contrib_DeformableConvolution(ndarray)" args ->
                                   IO [NDArrayHandle]
_contrib_DeformableConvolution args
  = let scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> (args !? #kernel :: Maybe [Int]),
               ("stride",) . showValue <$> (args !? #stride :: Maybe [Int]),
               ("dilate",) . showValue <$> (args !? #dilate :: Maybe [Int]),
               ("pad",) . showValue <$> (args !? #pad :: Maybe [Int]),
               ("num_filter",) . showValue <$> (args !? #num_filter :: Maybe Int),
               ("num_group",) . showValue <$> (args !? #num_group :: Maybe Int),
               ("num_deformable_group",) . showValue <$>
                 (args !? #num_deformable_group :: Maybe Int),
               ("workspace",) . showValue <$> (args !? #workspace :: Maybe Int),
               ("no_bias",) . showValue <$> (args !? #no_bias :: Maybe Bool),
               ("layout",) . showValue <$>
                 (args !? #layout ::
                    Maybe (Maybe (EnumType '["NCDHW", "NCHW", "NCW"])))]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("offset",) <$> (args !? #offset :: Maybe NDArrayHandle),
               ("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("bias",) <$> (args !? #bias :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_DeformableConvolution"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_contrib_DeformableConvolution_upd ::
                                   forall args .
                                     Fullfilled "_contrib_DeformableConvolution(ndarray)" args =>
                                     [NDArrayHandle] ->
                                       ArgsHMap "_contrib_DeformableConvolution(ndarray)" args ->
                                         IO ()
_contrib_DeformableConvolution_upd outputs args
  = let scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> (args !? #kernel :: Maybe [Int]),
               ("stride",) . showValue <$> (args !? #stride :: Maybe [Int]),
               ("dilate",) . showValue <$> (args !? #dilate :: Maybe [Int]),
               ("pad",) . showValue <$> (args !? #pad :: Maybe [Int]),
               ("num_filter",) . showValue <$> (args !? #num_filter :: Maybe Int),
               ("num_group",) . showValue <$> (args !? #num_group :: Maybe Int),
               ("num_deformable_group",) . showValue <$>
                 (args !? #num_deformable_group :: Maybe Int),
               ("workspace",) . showValue <$> (args !? #workspace :: Maybe Int),
               ("no_bias",) . showValue <$> (args !? #no_bias :: Maybe Bool),
               ("layout",) . showValue <$>
                 (args !? #layout ::
                    Maybe (Maybe (EnumType '["NCDHW", "NCHW", "NCW"])))]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("offset",) <$> (args !? #offset :: Maybe NDArrayHandle),
               ("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("bias",) <$> (args !? #bias :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_DeformableConvolution"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance
     ParameterList "_backward__contrib_DeformableConvolution(ndarray)" =
     '[]

_backward__contrib_DeformableConvolution ::
                                         forall args .
                                           Fullfilled
                                             "_backward__contrib_DeformableConvolution(ndarray)"
                                             args =>
                                           ArgsHMap
                                             "_backward__contrib_DeformableConvolution(ndarray)"
                                             args
                                             -> IO [NDArrayHandle]
_backward__contrib_DeformableConvolution args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_DeformableConvolution"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward__contrib_DeformableConvolution_upd ::
                                             forall args .
                                               Fullfilled
                                                 "_backward__contrib_DeformableConvolution(ndarray)"
                                                 args =>
                                               [NDArrayHandle] ->
                                                 ArgsHMap
                                                   "_backward__contrib_DeformableConvolution(ndarray)"
                                                   args
                                                   -> IO ()
_backward__contrib_DeformableConvolution_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_DeformableConvolution"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_contrib_ifft(ndarray)" =
     '[ '("compute_size", AttrOpt Int),
        '("data", AttrOpt NDArrayHandle)]

_contrib_ifft ::
              forall args . Fullfilled "_contrib_ifft(ndarray)" args =>
                ArgsHMap "_contrib_ifft(ndarray)" args -> IO [NDArrayHandle]
_contrib_ifft args
  = let scalarArgs
          = catMaybes
              [("compute_size",) . showValue <$>
                 (args !? #compute_size :: Maybe Int)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_ifft"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_contrib_ifft_upd ::
                  forall args . Fullfilled "_contrib_ifft(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_contrib_ifft(ndarray)" args -> IO ()
_contrib_ifft_upd outputs args
  = let scalarArgs
          = catMaybes
              [("compute_size",) . showValue <$>
                 (args !? #compute_size :: Maybe Int)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_ifft"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward__contrib_ifft(ndarray)" =
     '[]

_backward__contrib_ifft ::
                        forall args . Fullfilled "_backward__contrib_ifft(ndarray)" args =>
                          ArgsHMap "_backward__contrib_ifft(ndarray)" args ->
                            IO [NDArrayHandle]
_backward__contrib_ifft args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_ifft"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward__contrib_ifft_upd ::
                            forall args . Fullfilled "_backward__contrib_ifft(ndarray)" args =>
                              [NDArrayHandle] ->
                                ArgsHMap "_backward__contrib_ifft(ndarray)" args -> IO ()
_backward__contrib_ifft_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_ifft"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward__Native(ndarray)" = '[]

_backward__Native ::
                  forall args . Fullfilled "_backward__Native(ndarray)" args =>
                    ArgsHMap "_backward__Native(ndarray)" args -> IO [NDArrayHandle]
_backward__Native args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__Native"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward__Native_upd ::
                      forall args . Fullfilled "_backward__Native(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "_backward__Native(ndarray)" args -> IO ()
_backward__Native_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__Native"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward__NDArray(ndarray)" = '[]

_backward__NDArray ::
                   forall args . Fullfilled "_backward__NDArray(ndarray)" args =>
                     ArgsHMap "_backward__NDArray(ndarray)" args -> IO [NDArrayHandle]
_backward__NDArray args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__NDArray"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward__NDArray_upd ::
                       forall args . Fullfilled "_backward__NDArray(ndarray)" args =>
                         [NDArrayHandle] ->
                           ArgsHMap "_backward__NDArray(ndarray)" args -> IO ()
_backward__NDArray_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__NDArray"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_InstanceNorm(ndarray)" =
     '[ '("eps", AttrOpt Float), '("data", AttrOpt NDArrayHandle),
        '("gamma", AttrOpt NDArrayHandle),
        '("beta", AttrOpt NDArrayHandle)]

_InstanceNorm ::
              forall args . Fullfilled "_InstanceNorm(ndarray)" args =>
                ArgsHMap "_InstanceNorm(ndarray)" args -> IO [NDArrayHandle]
_InstanceNorm args
  = let scalarArgs
          = catMaybes
              [("eps",) . showValue <$> (args !? #eps :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("gamma",) <$> (args !? #gamma :: Maybe NDArrayHandle),
               ("beta",) <$> (args !? #beta :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "InstanceNorm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_InstanceNorm_upd ::
                  forall args . Fullfilled "_InstanceNorm(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_InstanceNorm(ndarray)" args -> IO ()
_InstanceNorm_upd outputs args
  = let scalarArgs
          = catMaybes
              [("eps",) . showValue <$> (args !? #eps :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("gamma",) <$> (args !? #gamma :: Maybe NDArrayHandle),
               ("beta",) <$> (args !? #beta :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "InstanceNorm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_InstanceNorm(ndarray)" = '[]

_backward_InstanceNorm ::
                       forall args . Fullfilled "_backward_InstanceNorm(ndarray)" args =>
                         ArgsHMap "_backward_InstanceNorm(ndarray)" args ->
                           IO [NDArrayHandle]
_backward_InstanceNorm args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_InstanceNorm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_InstanceNorm_upd ::
                           forall args . Fullfilled "_backward_InstanceNorm(ndarray)" args =>
                             [NDArrayHandle] ->
                               ArgsHMap "_backward_InstanceNorm(ndarray)" args -> IO ()
_backward_InstanceNorm_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_InstanceNorm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_SVMOutput(ndarray)" =
     '[ '("margin", AttrOpt Float),
        '("regularization_coefficient", AttrOpt Float),
        '("use_linear", AttrOpt Bool), '("data", AttrOpt NDArrayHandle),
        '("label", AttrOpt NDArrayHandle)]

_SVMOutput ::
           forall args . Fullfilled "_SVMOutput(ndarray)" args =>
             ArgsHMap "_SVMOutput(ndarray)" args -> IO [NDArrayHandle]
_SVMOutput args
  = let scalarArgs
          = catMaybes
              [("margin",) . showValue <$> (args !? #margin :: Maybe Float),
               ("regularization_coefficient",) . showValue <$>
                 (args !? #regularization_coefficient :: Maybe Float),
               ("use_linear",) . showValue <$>
                 (args !? #use_linear :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("label",) <$> (args !? #label :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SVMOutput"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_SVMOutput_upd ::
               forall args . Fullfilled "_SVMOutput(ndarray)" args =>
                 [NDArrayHandle] -> ArgsHMap "_SVMOutput(ndarray)" args -> IO ()
_SVMOutput_upd outputs args
  = let scalarArgs
          = catMaybes
              [("margin",) . showValue <$> (args !? #margin :: Maybe Float),
               ("regularization_coefficient",) . showValue <$>
                 (args !? #regularization_coefficient :: Maybe Float),
               ("use_linear",) . showValue <$>
                 (args !? #use_linear :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("label",) <$> (args !? #label :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SVMOutput"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_SVMOutput(ndarray)" = '[]

_backward_SVMOutput ::
                    forall args . Fullfilled "_backward_SVMOutput(ndarray)" args =>
                      ArgsHMap "_backward_SVMOutput(ndarray)" args -> IO [NDArrayHandle]
_backward_SVMOutput args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SVMOutput"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_SVMOutput_upd ::
                        forall args . Fullfilled "_backward_SVMOutput(ndarray)" args =>
                          [NDArrayHandle] ->
                            ArgsHMap "_backward_SVMOutput(ndarray)" args -> IO ()
_backward_SVMOutput_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SVMOutput"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_Pooling(ndarray)" =
     '[ '("global_pool", AttrOpt Bool), '("cudnn_off", AttrOpt Bool),
        '("kernel", AttrReq [Int]),
        '("pool_type", AttrReq (EnumType '["avg", "max", "sum"])),
        '("pooling_convention", AttrOpt (EnumType '["full", "valid"])),
        '("stride", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("data", AttrOpt NDArrayHandle)]

_Pooling ::
         forall args . Fullfilled "_Pooling(ndarray)" args =>
           ArgsHMap "_Pooling(ndarray)" args -> IO [NDArrayHandle]
_Pooling args
  = let scalarArgs
          = catMaybes
              [("global_pool",) . showValue <$>
                 (args !? #global_pool :: Maybe Bool),
               ("cudnn_off",) . showValue <$> (args !? #cudnn_off :: Maybe Bool),
               ("kernel",) . showValue <$> (args !? #kernel :: Maybe [Int]),
               ("pool_type",) . showValue <$>
                 (args !? #pool_type :: Maybe (EnumType '["avg", "max", "sum"])),
               ("pooling_convention",) . showValue <$>
                 (args !? #pooling_convention ::
                    Maybe (EnumType '["full", "valid"])),
               ("stride",) . showValue <$> (args !? #stride :: Maybe [Int]),
               ("pad",) . showValue <$> (args !? #pad :: Maybe [Int])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Pooling"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_Pooling_upd ::
             forall args . Fullfilled "_Pooling(ndarray)" args =>
               [NDArrayHandle] -> ArgsHMap "_Pooling(ndarray)" args -> IO ()
_Pooling_upd outputs args
  = let scalarArgs
          = catMaybes
              [("global_pool",) . showValue <$>
                 (args !? #global_pool :: Maybe Bool),
               ("cudnn_off",) . showValue <$> (args !? #cudnn_off :: Maybe Bool),
               ("kernel",) . showValue <$> (args !? #kernel :: Maybe [Int]),
               ("pool_type",) . showValue <$>
                 (args !? #pool_type :: Maybe (EnumType '["avg", "max", "sum"])),
               ("pooling_convention",) . showValue <$>
                 (args !? #pooling_convention ::
                    Maybe (EnumType '["full", "valid"])),
               ("stride",) . showValue <$> (args !? #stride :: Maybe [Int]),
               ("pad",) . showValue <$> (args !? #pad :: Maybe [Int])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Pooling"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_Pooling(ndarray)" = '[]

_backward_Pooling ::
                  forall args . Fullfilled "_backward_Pooling(ndarray)" args =>
                    ArgsHMap "_backward_Pooling(ndarray)" args -> IO [NDArrayHandle]
_backward_Pooling args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Pooling"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_Pooling_upd ::
                      forall args . Fullfilled "_backward_Pooling(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "_backward_Pooling(ndarray)" args -> IO ()
_backward_Pooling_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Pooling"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_Convolution_v1(ndarray)" =
     '[ '("kernel", AttrReq [Int]), '("stride", AttrOpt [Int]),
        '("dilate", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("num_filter", AttrReq Int), '("num_group", AttrOpt Int),
        '("workspace", AttrOpt Int), '("no_bias", AttrOpt Bool),
        '("cudnn_tune",
          AttrOpt
            (Maybe (EnumType '["fastest", "limited_workspace", "off"]))),
        '("cudnn_off", AttrOpt Bool),
        '("layout",
          AttrOpt (Maybe (EnumType '["NCDHW", "NCHW", "NDHWC", "NHWC"]))),
        '("data", AttrOpt NDArrayHandle),
        '("weight", AttrOpt NDArrayHandle),
        '("bias", AttrOpt NDArrayHandle)]

_Convolution_v1 ::
                forall args . Fullfilled "_Convolution_v1(ndarray)" args =>
                  ArgsHMap "_Convolution_v1(ndarray)" args -> IO [NDArrayHandle]
_Convolution_v1 args
  = let scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> (args !? #kernel :: Maybe [Int]),
               ("stride",) . showValue <$> (args !? #stride :: Maybe [Int]),
               ("dilate",) . showValue <$> (args !? #dilate :: Maybe [Int]),
               ("pad",) . showValue <$> (args !? #pad :: Maybe [Int]),
               ("num_filter",) . showValue <$> (args !? #num_filter :: Maybe Int),
               ("num_group",) . showValue <$> (args !? #num_group :: Maybe Int),
               ("workspace",) . showValue <$> (args !? #workspace :: Maybe Int),
               ("no_bias",) . showValue <$> (args !? #no_bias :: Maybe Bool),
               ("cudnn_tune",) . showValue <$>
                 (args !? #cudnn_tune ::
                    Maybe (Maybe (EnumType '["fastest", "limited_workspace", "off"]))),
               ("cudnn_off",) . showValue <$> (args !? #cudnn_off :: Maybe Bool),
               ("layout",) . showValue <$>
                 (args !? #layout ::
                    Maybe (Maybe (EnumType '["NCDHW", "NCHW", "NDHWC", "NHWC"])))]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("bias",) <$> (args !? #bias :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Convolution_v1"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_Convolution_v1_upd ::
                    forall args . Fullfilled "_Convolution_v1(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_Convolution_v1(ndarray)" args -> IO ()
_Convolution_v1_upd outputs args
  = let scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> (args !? #kernel :: Maybe [Int]),
               ("stride",) . showValue <$> (args !? #stride :: Maybe [Int]),
               ("dilate",) . showValue <$> (args !? #dilate :: Maybe [Int]),
               ("pad",) . showValue <$> (args !? #pad :: Maybe [Int]),
               ("num_filter",) . showValue <$> (args !? #num_filter :: Maybe Int),
               ("num_group",) . showValue <$> (args !? #num_group :: Maybe Int),
               ("workspace",) . showValue <$> (args !? #workspace :: Maybe Int),
               ("no_bias",) . showValue <$> (args !? #no_bias :: Maybe Bool),
               ("cudnn_tune",) . showValue <$>
                 (args !? #cudnn_tune ::
                    Maybe (Maybe (EnumType '["fastest", "limited_workspace", "off"]))),
               ("cudnn_off",) . showValue <$> (args !? #cudnn_off :: Maybe Bool),
               ("layout",) . showValue <$>
                 (args !? #layout ::
                    Maybe (Maybe (EnumType '["NCDHW", "NCHW", "NDHWC", "NHWC"])))]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("bias",) <$> (args !? #bias :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Convolution_v1"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_Convolution_v1(ndarray)" =
     '[]

_backward_Convolution_v1 ::
                         forall args .
                           Fullfilled "_backward_Convolution_v1(ndarray)" args =>
                           ArgsHMap "_backward_Convolution_v1(ndarray)" args ->
                             IO [NDArrayHandle]
_backward_Convolution_v1 args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Convolution_v1"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_Convolution_v1_upd ::
                             forall args .
                               Fullfilled "_backward_Convolution_v1(ndarray)" args =>
                               [NDArrayHandle] ->
                                 ArgsHMap "_backward_Convolution_v1(ndarray)" args -> IO ()
_backward_Convolution_v1_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Convolution_v1"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_Crop(ndarray)" =
     '[ '("num_args", AttrReq Int), '("offset", AttrOpt [Int]),
        '("h_w", AttrOpt [Int]), '("center_crop", AttrOpt Bool),
        '("data", AttrOpt [NDArrayHandle])]

_Crop ::
      forall args . Fullfilled "_Crop(ndarray)" args =>
        ArgsHMap "_Crop(ndarray)" args -> IO [NDArrayHandle]
_Crop args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("offset",) . showValue <$> (args !? #offset :: Maybe [Int]),
               ("h_w",) . showValue <$> (args !? #h_w :: Maybe [Int]),
               ("center_crop",) . showValue <$>
                 (args !? #center_crop :: Maybe Bool)]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
        array = fromMaybe [] (args !? #data :: Maybe [NDArrayHandle])
      in
      do op <- nnGetOpHandle "Crop"
         let scalarArgs'
               = if hasKey args #num_args then scalarArgs else
                   (,) "num_args" (showValue (length array)) : scalarArgs
         listndarr <- mxImperativeInvoke (fromOpHandle op) array scalarArgs'
                        Nothing
         return listndarr

_Crop_upd ::
          forall args . Fullfilled "_Crop(ndarray)" args =>
            [NDArrayHandle] -> ArgsHMap "_Crop(ndarray)" args -> IO ()
_Crop_upd outputs args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("offset",) . showValue <$> (args !? #offset :: Maybe [Int]),
               ("h_w",) . showValue <$> (args !? #h_w :: Maybe [Int]),
               ("center_crop",) . showValue <$>
                 (args !? #center_crop :: Maybe Bool)]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
        array = fromMaybe [] (args !? #data :: Maybe [NDArrayHandle])
      in
      do op <- nnGetOpHandle "Crop"
         let scalarArgs'
               = if hasKey args #num_args then scalarArgs else
                   (,) "num_args" (showValue (length array)) : scalarArgs
         listndarr <- mxImperativeInvoke (fromOpHandle op) array scalarArgs'
                        (Just outputs)
         return ()

type instance ParameterList "_backward_Crop(ndarray)" = '[]

_backward_Crop ::
               forall args . Fullfilled "_backward_Crop(ndarray)" args =>
                 ArgsHMap "_backward_Crop(ndarray)" args -> IO [NDArrayHandle]
_backward_Crop args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Crop"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_Crop_upd ::
                   forall args . Fullfilled "_backward_Crop(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_backward_Crop(ndarray)" args -> IO ()
_backward_Crop_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Crop"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_SpatialTransformer(ndarray)" =
     '[ '("target_shape", AttrOpt [Int]),
        '("transform_type", AttrReq (EnumType '["affine"])),
        '("sampler_type", AttrReq (EnumType '["bilinear"])),
        '("data", AttrOpt NDArrayHandle), '("loc", AttrOpt NDArrayHandle)]

_SpatialTransformer ::
                    forall args . Fullfilled "_SpatialTransformer(ndarray)" args =>
                      ArgsHMap "_SpatialTransformer(ndarray)" args -> IO [NDArrayHandle]
_SpatialTransformer args
  = let scalarArgs
          = catMaybes
              [("target_shape",) . showValue <$>
                 (args !? #target_shape :: Maybe [Int]),
               ("transform_type",) . showValue <$>
                 (args !? #transform_type :: Maybe (EnumType '["affine"])),
               ("sampler_type",) . showValue <$>
                 (args !? #sampler_type :: Maybe (EnumType '["bilinear"]))]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("loc",) <$> (args !? #loc :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SpatialTransformer"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_SpatialTransformer_upd ::
                        forall args . Fullfilled "_SpatialTransformer(ndarray)" args =>
                          [NDArrayHandle] ->
                            ArgsHMap "_SpatialTransformer(ndarray)" args -> IO ()
_SpatialTransformer_upd outputs args
  = let scalarArgs
          = catMaybes
              [("target_shape",) . showValue <$>
                 (args !? #target_shape :: Maybe [Int]),
               ("transform_type",) . showValue <$>
                 (args !? #transform_type :: Maybe (EnumType '["affine"])),
               ("sampler_type",) . showValue <$>
                 (args !? #sampler_type :: Maybe (EnumType '["bilinear"]))]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("loc",) <$> (args !? #loc :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SpatialTransformer"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_SpatialTransformer(ndarray)"
     = '[]

_backward_SpatialTransformer ::
                             forall args .
                               Fullfilled "_backward_SpatialTransformer(ndarray)" args =>
                               ArgsHMap "_backward_SpatialTransformer(ndarray)" args ->
                                 IO [NDArrayHandle]
_backward_SpatialTransformer args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SpatialTransformer"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_SpatialTransformer_upd ::
                                 forall args .
                                   Fullfilled "_backward_SpatialTransformer(ndarray)" args =>
                                   [NDArrayHandle] ->
                                     ArgsHMap "_backward_SpatialTransformer(ndarray)" args -> IO ()
_backward_SpatialTransformer_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SpatialTransformer"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_SwapAxis(ndarray)" = '[]

_backward_SwapAxis ::
                   forall args . Fullfilled "_backward_SwapAxis(ndarray)" args =>
                     ArgsHMap "_backward_SwapAxis(ndarray)" args -> IO [NDArrayHandle]
_backward_SwapAxis args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SwapAxis"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_SwapAxis_upd ::
                       forall args . Fullfilled "_backward_SwapAxis(ndarray)" args =>
                         [NDArrayHandle] ->
                           ArgsHMap "_backward_SwapAxis(ndarray)" args -> IO ()
_backward_SwapAxis_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SwapAxis"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_LinearRegressionOutput(ndarray)" =
     '[ '("grad_scale", AttrOpt Float),
        '("data", AttrOpt NDArrayHandle),
        '("label", AttrOpt NDArrayHandle)]

_LinearRegressionOutput ::
                        forall args . Fullfilled "_LinearRegressionOutput(ndarray)" args =>
                          ArgsHMap "_LinearRegressionOutput(ndarray)" args ->
                            IO [NDArrayHandle]
_LinearRegressionOutput args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("label",) <$> (args !? #label :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "LinearRegressionOutput"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_LinearRegressionOutput_upd ::
                            forall args . Fullfilled "_LinearRegressionOutput(ndarray)" args =>
                              [NDArrayHandle] ->
                                ArgsHMap "_LinearRegressionOutput(ndarray)" args -> IO ()
_LinearRegressionOutput_upd outputs args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("label",) <$> (args !? #label :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "LinearRegressionOutput"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance
     ParameterList "_backward_LinearRegressionOutput(ndarray)" = '[]

_backward_LinearRegressionOutput ::
                                 forall args .
                                   Fullfilled "_backward_LinearRegressionOutput(ndarray)" args =>
                                   ArgsHMap "_backward_LinearRegressionOutput(ndarray)" args ->
                                     IO [NDArrayHandle]
_backward_LinearRegressionOutput args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_LinearRegressionOutput"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_LinearRegressionOutput_upd ::
                                     forall args .
                                       Fullfilled "_backward_LinearRegressionOutput(ndarray)"
                                         args =>
                                       [NDArrayHandle] ->
                                         ArgsHMap "_backward_LinearRegressionOutput(ndarray)" args
                                           -> IO ()
_backward_LinearRegressionOutput_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_LinearRegressionOutput"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_MAERegressionOutput(ndarray)" =
     '[ '("grad_scale", AttrOpt Float),
        '("data", AttrOpt NDArrayHandle),
        '("label", AttrOpt NDArrayHandle)]

_MAERegressionOutput ::
                     forall args . Fullfilled "_MAERegressionOutput(ndarray)" args =>
                       ArgsHMap "_MAERegressionOutput(ndarray)" args -> IO [NDArrayHandle]
_MAERegressionOutput args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("label",) <$> (args !? #label :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "MAERegressionOutput"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_MAERegressionOutput_upd ::
                         forall args . Fullfilled "_MAERegressionOutput(ndarray)" args =>
                           [NDArrayHandle] ->
                             ArgsHMap "_MAERegressionOutput(ndarray)" args -> IO ()
_MAERegressionOutput_upd outputs args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("label",) <$> (args !? #label :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "MAERegressionOutput"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance
     ParameterList "_backward_MAERegressionOutput(ndarray)" = '[]

_backward_MAERegressionOutput ::
                              forall args .
                                Fullfilled "_backward_MAERegressionOutput(ndarray)" args =>
                                ArgsHMap "_backward_MAERegressionOutput(ndarray)" args ->
                                  IO [NDArrayHandle]
_backward_MAERegressionOutput args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_MAERegressionOutput"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_MAERegressionOutput_upd ::
                                  forall args .
                                    Fullfilled "_backward_MAERegressionOutput(ndarray)" args =>
                                    [NDArrayHandle] ->
                                      ArgsHMap "_backward_MAERegressionOutput(ndarray)" args ->
                                        IO ()
_backward_MAERegressionOutput_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_MAERegressionOutput"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_LogisticRegressionOutput(ndarray)" =
     '[ '("grad_scale", AttrOpt Float),
        '("data", AttrOpt NDArrayHandle),
        '("label", AttrOpt NDArrayHandle)]

_LogisticRegressionOutput ::
                          forall args .
                            Fullfilled "_LogisticRegressionOutput(ndarray)" args =>
                            ArgsHMap "_LogisticRegressionOutput(ndarray)" args ->
                              IO [NDArrayHandle]
_LogisticRegressionOutput args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("label",) <$> (args !? #label :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "LogisticRegressionOutput"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_LogisticRegressionOutput_upd ::
                              forall args .
                                Fullfilled "_LogisticRegressionOutput(ndarray)" args =>
                                [NDArrayHandle] ->
                                  ArgsHMap "_LogisticRegressionOutput(ndarray)" args -> IO ()
_LogisticRegressionOutput_upd outputs args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("label",) <$> (args !? #label :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "LogisticRegressionOutput"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance
     ParameterList "_backward_LogisticRegressionOutput(ndarray)" = '[]

_backward_LogisticRegressionOutput ::
                                   forall args .
                                     Fullfilled "_backward_LogisticRegressionOutput(ndarray)"
                                       args =>
                                     ArgsHMap "_backward_LogisticRegressionOutput(ndarray)" args ->
                                       IO [NDArrayHandle]
_backward_LogisticRegressionOutput args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_LogisticRegressionOutput"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_LogisticRegressionOutput_upd ::
                                       forall args .
                                         Fullfilled "_backward_LogisticRegressionOutput(ndarray)"
                                           args =>
                                         [NDArrayHandle] ->
                                           ArgsHMap "_backward_LogisticRegressionOutput(ndarray)"
                                             args
                                             -> IO ()
_backward_LogisticRegressionOutput_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_LogisticRegressionOutput"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_Pad(ndarray)" = '[]

_backward_Pad ::
              forall args . Fullfilled "_backward_Pad(ndarray)" args =>
                ArgsHMap "_backward_Pad(ndarray)" args -> IO [NDArrayHandle]
_backward_Pad args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Pad"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_Pad_upd ::
                  forall args . Fullfilled "_backward_Pad(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_backward_Pad(ndarray)" args -> IO ()
_backward_Pad_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Pad"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_SoftmaxOutput(ndarray)" =
     '[ '("grad_scale", AttrOpt Float),
        '("ignore_label", AttrOpt Float), '("multi_output", AttrOpt Bool),
        '("use_ignore", AttrOpt Bool), '("preserve_shape", AttrOpt Bool),
        '("normalization", AttrOpt (EnumType '["batch", "null", "valid"])),
        '("out_grad", AttrOpt Bool), '("smooth_alpha", AttrOpt Float),
        '("data", AttrOpt NDArrayHandle),
        '("label", AttrOpt NDArrayHandle)]

_SoftmaxOutput ::
               forall args . Fullfilled "_SoftmaxOutput(ndarray)" args =>
                 ArgsHMap "_SoftmaxOutput(ndarray)" args -> IO [NDArrayHandle]
_SoftmaxOutput args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float),
               ("ignore_label",) . showValue <$>
                 (args !? #ignore_label :: Maybe Float),
               ("multi_output",) . showValue <$>
                 (args !? #multi_output :: Maybe Bool),
               ("use_ignore",) . showValue <$>
                 (args !? #use_ignore :: Maybe Bool),
               ("preserve_shape",) . showValue <$>
                 (args !? #preserve_shape :: Maybe Bool),
               ("normalization",) . showValue <$>
                 (args !? #normalization ::
                    Maybe (EnumType '["batch", "null", "valid"])),
               ("out_grad",) . showValue <$> (args !? #out_grad :: Maybe Bool),
               ("smooth_alpha",) . showValue <$>
                 (args !? #smooth_alpha :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("label",) <$> (args !? #label :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SoftmaxOutput"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_SoftmaxOutput_upd ::
                   forall args . Fullfilled "_SoftmaxOutput(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_SoftmaxOutput(ndarray)" args -> IO ()
_SoftmaxOutput_upd outputs args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float),
               ("ignore_label",) . showValue <$>
                 (args !? #ignore_label :: Maybe Float),
               ("multi_output",) . showValue <$>
                 (args !? #multi_output :: Maybe Bool),
               ("use_ignore",) . showValue <$>
                 (args !? #use_ignore :: Maybe Bool),
               ("preserve_shape",) . showValue <$>
                 (args !? #preserve_shape :: Maybe Bool),
               ("normalization",) . showValue <$>
                 (args !? #normalization ::
                    Maybe (EnumType '["batch", "null", "valid"])),
               ("out_grad",) . showValue <$> (args !? #out_grad :: Maybe Bool),
               ("smooth_alpha",) . showValue <$>
                 (args !? #smooth_alpha :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("label",) <$> (args !? #label :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SoftmaxOutput"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_SoftmaxOutput(ndarray)" =
     '[]

_backward_SoftmaxOutput ::
                        forall args . Fullfilled "_backward_SoftmaxOutput(ndarray)" args =>
                          ArgsHMap "_backward_SoftmaxOutput(ndarray)" args ->
                            IO [NDArrayHandle]
_backward_SoftmaxOutput args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SoftmaxOutput"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_SoftmaxOutput_upd ::
                            forall args . Fullfilled "_backward_SoftmaxOutput(ndarray)" args =>
                              [NDArrayHandle] ->
                                ArgsHMap "_backward_SoftmaxOutput(ndarray)" args -> IO ()
_backward_SoftmaxOutput_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SoftmaxOutput"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_Softmax(ndarray)" =
     '[ '("grad_scale", AttrOpt Float),
        '("ignore_label", AttrOpt Float), '("multi_output", AttrOpt Bool),
        '("use_ignore", AttrOpt Bool), '("preserve_shape", AttrOpt Bool),
        '("normalization", AttrOpt (EnumType '["batch", "null", "valid"])),
        '("out_grad", AttrOpt Bool), '("smooth_alpha", AttrOpt Float),
        '("data", AttrOpt NDArrayHandle)]

_Softmax ::
         forall args . Fullfilled "_Softmax(ndarray)" args =>
           ArgsHMap "_Softmax(ndarray)" args -> IO [NDArrayHandle]
_Softmax args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float),
               ("ignore_label",) . showValue <$>
                 (args !? #ignore_label :: Maybe Float),
               ("multi_output",) . showValue <$>
                 (args !? #multi_output :: Maybe Bool),
               ("use_ignore",) . showValue <$>
                 (args !? #use_ignore :: Maybe Bool),
               ("preserve_shape",) . showValue <$>
                 (args !? #preserve_shape :: Maybe Bool),
               ("normalization",) . showValue <$>
                 (args !? #normalization ::
                    Maybe (EnumType '["batch", "null", "valid"])),
               ("out_grad",) . showValue <$> (args !? #out_grad :: Maybe Bool),
               ("smooth_alpha",) . showValue <$>
                 (args !? #smooth_alpha :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Softmax"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_Softmax_upd ::
             forall args . Fullfilled "_Softmax(ndarray)" args =>
               [NDArrayHandle] -> ArgsHMap "_Softmax(ndarray)" args -> IO ()
_Softmax_upd outputs args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float),
               ("ignore_label",) . showValue <$>
                 (args !? #ignore_label :: Maybe Float),
               ("multi_output",) . showValue <$>
                 (args !? #multi_output :: Maybe Bool),
               ("use_ignore",) . showValue <$>
                 (args !? #use_ignore :: Maybe Bool),
               ("preserve_shape",) . showValue <$>
                 (args !? #preserve_shape :: Maybe Bool),
               ("normalization",) . showValue <$>
                 (args !? #normalization ::
                    Maybe (EnumType '["batch", "null", "valid"])),
               ("out_grad",) . showValue <$> (args !? #out_grad :: Maybe Bool),
               ("smooth_alpha",) . showValue <$>
                 (args !? #smooth_alpha :: Maybe Float)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Softmax"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_Softmax(ndarray)" = '[]

_backward_Softmax ::
                  forall args . Fullfilled "_backward_Softmax(ndarray)" args =>
                    ArgsHMap "_backward_Softmax(ndarray)" args -> IO [NDArrayHandle]
_backward_Softmax args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Softmax"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_Softmax_upd ::
                      forall args . Fullfilled "_backward_Softmax(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "_backward_Softmax(ndarray)" args -> IO ()
_backward_Softmax_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Softmax"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_SequenceReverse(ndarray)" =
     '[ '("use_sequence_length", AttrOpt Bool),
        '("data", AttrOpt NDArrayHandle),
        '("sequence_length", AttrOpt NDArrayHandle)]

_SequenceReverse ::
                 forall args . Fullfilled "_SequenceReverse(ndarray)" args =>
                   ArgsHMap "_SequenceReverse(ndarray)" args -> IO [NDArrayHandle]
_SequenceReverse args
  = let scalarArgs
          = catMaybes
              [("use_sequence_length",) . showValue <$>
                 (args !? #use_sequence_length :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("sequence_length",) <$>
                 (args !? #sequence_length :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SequenceReverse"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_SequenceReverse_upd ::
                     forall args . Fullfilled "_SequenceReverse(ndarray)" args =>
                       [NDArrayHandle] ->
                         ArgsHMap "_SequenceReverse(ndarray)" args -> IO ()
_SequenceReverse_upd outputs args
  = let scalarArgs
          = catMaybes
              [("use_sequence_length",) . showValue <$>
                 (args !? #use_sequence_length :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("sequence_length",) <$>
                 (args !? #sequence_length :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SequenceReverse"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_SequenceReverse(ndarray)" =
     '[]

_backward_SequenceReverse ::
                          forall args .
                            Fullfilled "_backward_SequenceReverse(ndarray)" args =>
                            ArgsHMap "_backward_SequenceReverse(ndarray)" args ->
                              IO [NDArrayHandle]
_backward_SequenceReverse args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SequenceReverse"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_SequenceReverse_upd ::
                              forall args .
                                Fullfilled "_backward_SequenceReverse(ndarray)" args =>
                                [NDArrayHandle] ->
                                  ArgsHMap "_backward_SequenceReverse(ndarray)" args -> IO ()
_backward_SequenceReverse_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SequenceReverse"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_BatchNorm_v1(ndarray)" = '[]

_backward_BatchNorm_v1 ::
                       forall args . Fullfilled "_backward_BatchNorm_v1(ndarray)" args =>
                         ArgsHMap "_backward_BatchNorm_v1(ndarray)" args ->
                           IO [NDArrayHandle]
_backward_BatchNorm_v1 args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_BatchNorm_v1"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_BatchNorm_v1_upd ::
                           forall args . Fullfilled "_backward_BatchNorm_v1(ndarray)" args =>
                             [NDArrayHandle] ->
                               ArgsHMap "_backward_BatchNorm_v1(ndarray)" args -> IO ()
_backward_BatchNorm_v1_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_BatchNorm_v1"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_SequenceLast(ndarray)" =
     '[ '("use_sequence_length", AttrOpt Bool),
        '("data", AttrOpt NDArrayHandle),
        '("sequence_length", AttrOpt NDArrayHandle)]

_SequenceLast ::
              forall args . Fullfilled "_SequenceLast(ndarray)" args =>
                ArgsHMap "_SequenceLast(ndarray)" args -> IO [NDArrayHandle]
_SequenceLast args
  = let scalarArgs
          = catMaybes
              [("use_sequence_length",) . showValue <$>
                 (args !? #use_sequence_length :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("sequence_length",) <$>
                 (args !? #sequence_length :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SequenceLast"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_SequenceLast_upd ::
                  forall args . Fullfilled "_SequenceLast(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_SequenceLast(ndarray)" args -> IO ()
_SequenceLast_upd outputs args
  = let scalarArgs
          = catMaybes
              [("use_sequence_length",) . showValue <$>
                 (args !? #use_sequence_length :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("sequence_length",) <$>
                 (args !? #sequence_length :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SequenceLast"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_SequenceLast(ndarray)" = '[]

_backward_SequenceLast ::
                       forall args . Fullfilled "_backward_SequenceLast(ndarray)" args =>
                         ArgsHMap "_backward_SequenceLast(ndarray)" args ->
                           IO [NDArrayHandle]
_backward_SequenceLast args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SequenceLast"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_SequenceLast_upd ::
                           forall args . Fullfilled "_backward_SequenceLast(ndarray)" args =>
                             [NDArrayHandle] ->
                               ArgsHMap "_backward_SequenceLast(ndarray)" args -> IO ()
_backward_SequenceLast_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SequenceLast"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_Correlation(ndarray)" =
     '[ '("kernel_size", AttrOpt Int),
        '("max_displacement", AttrOpt Int), '("stride1", AttrOpt Int),
        '("stride2", AttrOpt Int), '("pad_size", AttrOpt Int),
        '("is_multiply", AttrOpt Bool), '("data1", AttrOpt NDArrayHandle),
        '("data2", AttrOpt NDArrayHandle)]

_Correlation ::
             forall args . Fullfilled "_Correlation(ndarray)" args =>
               ArgsHMap "_Correlation(ndarray)" args -> IO [NDArrayHandle]
_Correlation args
  = let scalarArgs
          = catMaybes
              [("kernel_size",) . showValue <$>
                 (args !? #kernel_size :: Maybe Int),
               ("max_displacement",) . showValue <$>
                 (args !? #max_displacement :: Maybe Int),
               ("stride1",) . showValue <$> (args !? #stride1 :: Maybe Int),
               ("stride2",) . showValue <$> (args !? #stride2 :: Maybe Int),
               ("pad_size",) . showValue <$> (args !? #pad_size :: Maybe Int),
               ("is_multiply",) . showValue <$>
                 (args !? #is_multiply :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("data1",) <$> (args !? #data1 :: Maybe NDArrayHandle),
               ("data2",) <$> (args !? #data2 :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Correlation"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_Correlation_upd ::
                 forall args . Fullfilled "_Correlation(ndarray)" args =>
                   [NDArrayHandle] -> ArgsHMap "_Correlation(ndarray)" args -> IO ()
_Correlation_upd outputs args
  = let scalarArgs
          = catMaybes
              [("kernel_size",) . showValue <$>
                 (args !? #kernel_size :: Maybe Int),
               ("max_displacement",) . showValue <$>
                 (args !? #max_displacement :: Maybe Int),
               ("stride1",) . showValue <$> (args !? #stride1 :: Maybe Int),
               ("stride2",) . showValue <$> (args !? #stride2 :: Maybe Int),
               ("pad_size",) . showValue <$> (args !? #pad_size :: Maybe Int),
               ("is_multiply",) . showValue <$>
                 (args !? #is_multiply :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("data1",) <$> (args !? #data1 :: Maybe NDArrayHandle),
               ("data2",) <$> (args !? #data2 :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Correlation"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_Correlation(ndarray)" = '[]

_backward_Correlation ::
                      forall args . Fullfilled "_backward_Correlation(ndarray)" args =>
                        ArgsHMap "_backward_Correlation(ndarray)" args ->
                          IO [NDArrayHandle]
_backward_Correlation args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Correlation"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_Correlation_upd ::
                          forall args . Fullfilled "_backward_Correlation(ndarray)" args =>
                            [NDArrayHandle] ->
                              ArgsHMap "_backward_Correlation(ndarray)" args -> IO ()
_backward_Correlation_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Correlation"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_MakeLoss(ndarray)" =
     '[ '("grad_scale", AttrOpt Float),
        '("valid_thresh", AttrOpt Float),
        '("normalization", AttrOpt (EnumType '["batch", "null", "valid"])),
        '("data", AttrOpt NDArrayHandle)]

_MakeLoss ::
          forall args . Fullfilled "_MakeLoss(ndarray)" args =>
            ArgsHMap "_MakeLoss(ndarray)" args -> IO [NDArrayHandle]
_MakeLoss args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float),
               ("valid_thresh",) . showValue <$>
                 (args !? #valid_thresh :: Maybe Float),
               ("normalization",) . showValue <$>
                 (args !? #normalization ::
                    Maybe (EnumType '["batch", "null", "valid"]))]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "MakeLoss"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_MakeLoss_upd ::
              forall args . Fullfilled "_MakeLoss(ndarray)" args =>
                [NDArrayHandle] -> ArgsHMap "_MakeLoss(ndarray)" args -> IO ()
_MakeLoss_upd outputs args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float),
               ("valid_thresh",) . showValue <$>
                 (args !? #valid_thresh :: Maybe Float),
               ("normalization",) . showValue <$>
                 (args !? #normalization ::
                    Maybe (EnumType '["batch", "null", "valid"]))]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "MakeLoss"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_MakeLoss(ndarray)" = '[]

_backward_MakeLoss ::
                   forall args . Fullfilled "_backward_MakeLoss(ndarray)" args =>
                     ArgsHMap "_backward_MakeLoss(ndarray)" args -> IO [NDArrayHandle]
_backward_MakeLoss args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_MakeLoss"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_MakeLoss_upd ::
                       forall args . Fullfilled "_backward_MakeLoss(ndarray)" args =>
                         [NDArrayHandle] ->
                           ArgsHMap "_backward_MakeLoss(ndarray)" args -> IO ()
_backward_MakeLoss_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_MakeLoss"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_LRN(ndarray)" =
     '[ '("alpha", AttrOpt Float), '("beta", AttrOpt Float),
        '("knorm", AttrOpt Float), '("nsize", AttrReq Int),
        '("data", AttrOpt NDArrayHandle)]

_LRN ::
     forall args . Fullfilled "_LRN(ndarray)" args =>
       ArgsHMap "_LRN(ndarray)" args -> IO [NDArrayHandle]
_LRN args
  = let scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> (args !? #alpha :: Maybe Float),
               ("beta",) . showValue <$> (args !? #beta :: Maybe Float),
               ("knorm",) . showValue <$> (args !? #knorm :: Maybe Float),
               ("nsize",) . showValue <$> (args !? #nsize :: Maybe Int)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "LRN"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_LRN_upd ::
         forall args . Fullfilled "_LRN(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "_LRN(ndarray)" args -> IO ()
_LRN_upd outputs args
  = let scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> (args !? #alpha :: Maybe Float),
               ("beta",) . showValue <$> (args !? #beta :: Maybe Float),
               ("knorm",) . showValue <$> (args !? #knorm :: Maybe Float),
               ("nsize",) . showValue <$> (args !? #nsize :: Maybe Int)]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "LRN"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_LRN(ndarray)" = '[]

_backward_LRN ::
              forall args . Fullfilled "_backward_LRN(ndarray)" args =>
                ArgsHMap "_backward_LRN(ndarray)" args -> IO [NDArrayHandle]
_backward_LRN args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_LRN"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_LRN_upd ::
                  forall args . Fullfilled "_backward_LRN(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_backward_LRN(ndarray)" args -> IO ()
_backward_LRN_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_LRN"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_SequenceMask(ndarray)" =
     '[ '("use_sequence_length", AttrOpt Bool),
        '("value", AttrOpt Float), '("data", AttrOpt NDArrayHandle),
        '("sequence_length", AttrOpt NDArrayHandle)]

_SequenceMask ::
              forall args . Fullfilled "_SequenceMask(ndarray)" args =>
                ArgsHMap "_SequenceMask(ndarray)" args -> IO [NDArrayHandle]
_SequenceMask args
  = let scalarArgs
          = catMaybes
              [("use_sequence_length",) . showValue <$>
                 (args !? #use_sequence_length :: Maybe Bool),
               ("value",) . showValue <$> (args !? #value :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("sequence_length",) <$>
                 (args !? #sequence_length :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SequenceMask"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_SequenceMask_upd ::
                  forall args . Fullfilled "_SequenceMask(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_SequenceMask(ndarray)" args -> IO ()
_SequenceMask_upd outputs args
  = let scalarArgs
          = catMaybes
              [("use_sequence_length",) . showValue <$>
                 (args !? #use_sequence_length :: Maybe Bool),
               ("value",) . showValue <$> (args !? #value :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("sequence_length",) <$>
                 (args !? #sequence_length :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SequenceMask"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_SequenceMask(ndarray)" = '[]

_backward_SequenceMask ::
                       forall args . Fullfilled "_backward_SequenceMask(ndarray)" args =>
                         ArgsHMap "_backward_SequenceMask(ndarray)" args ->
                           IO [NDArrayHandle]
_backward_SequenceMask args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SequenceMask"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_SequenceMask_upd ::
                           forall args . Fullfilled "_backward_SequenceMask(ndarray)" args =>
                             [NDArrayHandle] ->
                               ArgsHMap "_backward_SequenceMask(ndarray)" args -> IO ()
_backward_SequenceMask_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SequenceMask"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_GridGenerator(ndarray)" =
     '[ '("transform_type", AttrReq (EnumType '["affine", "warp"])),
        '("target_shape", AttrOpt [Int]), '("data", AttrOpt NDArrayHandle)]

_GridGenerator ::
               forall args . Fullfilled "_GridGenerator(ndarray)" args =>
                 ArgsHMap "_GridGenerator(ndarray)" args -> IO [NDArrayHandle]
_GridGenerator args
  = let scalarArgs
          = catMaybes
              [("transform_type",) . showValue <$>
                 (args !? #transform_type :: Maybe (EnumType '["affine", "warp"])),
               ("target_shape",) . showValue <$>
                 (args !? #target_shape :: Maybe [Int])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "GridGenerator"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_GridGenerator_upd ::
                   forall args . Fullfilled "_GridGenerator(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_GridGenerator(ndarray)" args -> IO ()
_GridGenerator_upd outputs args
  = let scalarArgs
          = catMaybes
              [("transform_type",) . showValue <$>
                 (args !? #transform_type :: Maybe (EnumType '["affine", "warp"])),
               ("target_shape",) . showValue <$>
                 (args !? #target_shape :: Maybe [Int])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "GridGenerator"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_GridGenerator(ndarray)" =
     '[]

_backward_GridGenerator ::
                        forall args . Fullfilled "_backward_GridGenerator(ndarray)" args =>
                          ArgsHMap "_backward_GridGenerator(ndarray)" args ->
                            IO [NDArrayHandle]
_backward_GridGenerator args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_GridGenerator"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_GridGenerator_upd ::
                            forall args . Fullfilled "_backward_GridGenerator(ndarray)" args =>
                              [NDArrayHandle] ->
                                ArgsHMap "_backward_GridGenerator(ndarray)" args -> IO ()
_backward_GridGenerator_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_GridGenerator"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_Pooling_v1(ndarray)" =
     '[ '("global_pool", AttrOpt Bool), '("kernel", AttrReq [Int]),
        '("pool_type", AttrReq (EnumType '["avg", "max", "sum"])),
        '("pooling_convention", AttrOpt (EnumType '["full", "valid"])),
        '("stride", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("data", AttrOpt NDArrayHandle)]

_Pooling_v1 ::
            forall args . Fullfilled "_Pooling_v1(ndarray)" args =>
              ArgsHMap "_Pooling_v1(ndarray)" args -> IO [NDArrayHandle]
_Pooling_v1 args
  = let scalarArgs
          = catMaybes
              [("global_pool",) . showValue <$>
                 (args !? #global_pool :: Maybe Bool),
               ("kernel",) . showValue <$> (args !? #kernel :: Maybe [Int]),
               ("pool_type",) . showValue <$>
                 (args !? #pool_type :: Maybe (EnumType '["avg", "max", "sum"])),
               ("pooling_convention",) . showValue <$>
                 (args !? #pooling_convention ::
                    Maybe (EnumType '["full", "valid"])),
               ("stride",) . showValue <$> (args !? #stride :: Maybe [Int]),
               ("pad",) . showValue <$> (args !? #pad :: Maybe [Int])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Pooling_v1"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_Pooling_v1_upd ::
                forall args . Fullfilled "_Pooling_v1(ndarray)" args =>
                  [NDArrayHandle] -> ArgsHMap "_Pooling_v1(ndarray)" args -> IO ()
_Pooling_v1_upd outputs args
  = let scalarArgs
          = catMaybes
              [("global_pool",) . showValue <$>
                 (args !? #global_pool :: Maybe Bool),
               ("kernel",) . showValue <$> (args !? #kernel :: Maybe [Int]),
               ("pool_type",) . showValue <$>
                 (args !? #pool_type :: Maybe (EnumType '["avg", "max", "sum"])),
               ("pooling_convention",) . showValue <$>
                 (args !? #pooling_convention ::
                    Maybe (EnumType '["full", "valid"])),
               ("stride",) . showValue <$> (args !? #stride :: Maybe [Int]),
               ("pad",) . showValue <$> (args !? #pad :: Maybe [Int])]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Pooling_v1"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_Pooling_v1(ndarray)" = '[]

_backward_Pooling_v1 ::
                     forall args . Fullfilled "_backward_Pooling_v1(ndarray)" args =>
                       ArgsHMap "_backward_Pooling_v1(ndarray)" args -> IO [NDArrayHandle]
_backward_Pooling_v1 args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Pooling_v1"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_Pooling_v1_upd ::
                         forall args . Fullfilled "_backward_Pooling_v1(ndarray)" args =>
                           [NDArrayHandle] ->
                             ArgsHMap "_backward_Pooling_v1(ndarray)" args -> IO ()
_backward_Pooling_v1_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Pooling_v1"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_LeakyReLU(ndarray)" = '[]

_backward_LeakyReLU ::
                    forall args . Fullfilled "_backward_LeakyReLU(ndarray)" args =>
                      ArgsHMap "_backward_LeakyReLU(ndarray)" args -> IO [NDArrayHandle]
_backward_LeakyReLU args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_LeakyReLU"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_LeakyReLU_upd ::
                        forall args . Fullfilled "_backward_LeakyReLU(ndarray)" args =>
                          [NDArrayHandle] ->
                            ArgsHMap "_backward_LeakyReLU(ndarray)" args -> IO ()
_backward_LeakyReLU_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_LeakyReLU"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance
     ParameterList "_backward_IdentityAttachKLSparseReg(ndarray)" = '[]

_backward_IdentityAttachKLSparseReg ::
                                    forall args .
                                      Fullfilled "_backward_IdentityAttachKLSparseReg(ndarray)"
                                        args =>
                                      ArgsHMap "_backward_IdentityAttachKLSparseReg(ndarray)" args
                                        -> IO [NDArrayHandle]
_backward_IdentityAttachKLSparseReg args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_IdentityAttachKLSparseReg"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_IdentityAttachKLSparseReg_upd ::
                                        forall args .
                                          Fullfilled "_backward_IdentityAttachKLSparseReg(ndarray)"
                                            args =>
                                          [NDArrayHandle] ->
                                            ArgsHMap "_backward_IdentityAttachKLSparseReg(ndarray)"
                                              args
                                              -> IO ()
_backward_IdentityAttachKLSparseReg_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_IdentityAttachKLSparseReg"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_Activation(ndarray)" =
     '[ '("act_type",
          AttrReq (EnumType '["relu", "sigmoid", "softrelu", "tanh"])),
        '("data", AttrOpt NDArrayHandle)]

_Activation ::
            forall args . Fullfilled "_Activation(ndarray)" args =>
              ArgsHMap "_Activation(ndarray)" args -> IO [NDArrayHandle]
_Activation args
  = let scalarArgs
          = catMaybes
              [("act_type",) . showValue <$>
                 (args !? #act_type ::
                    Maybe (EnumType '["relu", "sigmoid", "softrelu", "tanh"]))]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Activation"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_Activation_upd ::
                forall args . Fullfilled "_Activation(ndarray)" args =>
                  [NDArrayHandle] -> ArgsHMap "_Activation(ndarray)" args -> IO ()
_Activation_upd outputs args
  = let scalarArgs
          = catMaybes
              [("act_type",) . showValue <$>
                 (args !? #act_type ::
                    Maybe (EnumType '["relu", "sigmoid", "softrelu", "tanh"]))]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Activation"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_Activation(ndarray)" = '[]

_backward_Activation ::
                     forall args . Fullfilled "_backward_Activation(ndarray)" args =>
                       ArgsHMap "_backward_Activation(ndarray)" args -> IO [NDArrayHandle]
_backward_Activation args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Activation"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_Activation_upd ::
                         forall args . Fullfilled "_backward_Activation(ndarray)" args =>
                           [NDArrayHandle] ->
                             ArgsHMap "_backward_Activation(ndarray)" args -> IO ()
_backward_Activation_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Activation"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_UpSampling(ndarray)" = '[]

_backward_UpSampling ::
                     forall args . Fullfilled "_backward_UpSampling(ndarray)" args =>
                       ArgsHMap "_backward_UpSampling(ndarray)" args -> IO [NDArrayHandle]
_backward_UpSampling args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_UpSampling"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_UpSampling_upd ::
                         forall args . Fullfilled "_backward_UpSampling(ndarray)" args =>
                           [NDArrayHandle] ->
                             ArgsHMap "_backward_UpSampling(ndarray)" args -> IO ()
_backward_UpSampling_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_UpSampling"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_Deconvolution(ndarray)" =
     '[ '("kernel", AttrReq [Int]), '("stride", AttrOpt [Int]),
        '("dilate", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("adj", AttrOpt [Int]), '("target_shape", AttrOpt [Int]),
        '("num_filter", AttrReq Int), '("num_group", AttrOpt Int),
        '("workspace", AttrOpt Int), '("no_bias", AttrOpt Bool),
        '("cudnn_tune",
          AttrOpt
            (Maybe (EnumType '["fastest", "limited_workspace", "off"]))),
        '("cudnn_off", AttrOpt Bool),
        '("layout",
          AttrOpt
            (Maybe (EnumType '["NCDHW", "NCHW", "NCW", "NDHWC", "NHWC"]))),
        '("data", AttrOpt NDArrayHandle),
        '("weight", AttrOpt NDArrayHandle),
        '("bias", AttrOpt NDArrayHandle)]

_Deconvolution ::
               forall args . Fullfilled "_Deconvolution(ndarray)" args =>
                 ArgsHMap "_Deconvolution(ndarray)" args -> IO [NDArrayHandle]
_Deconvolution args
  = let scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> (args !? #kernel :: Maybe [Int]),
               ("stride",) . showValue <$> (args !? #stride :: Maybe [Int]),
               ("dilate",) . showValue <$> (args !? #dilate :: Maybe [Int]),
               ("pad",) . showValue <$> (args !? #pad :: Maybe [Int]),
               ("adj",) . showValue <$> (args !? #adj :: Maybe [Int]),
               ("target_shape",) . showValue <$>
                 (args !? #target_shape :: Maybe [Int]),
               ("num_filter",) . showValue <$> (args !? #num_filter :: Maybe Int),
               ("num_group",) . showValue <$> (args !? #num_group :: Maybe Int),
               ("workspace",) . showValue <$> (args !? #workspace :: Maybe Int),
               ("no_bias",) . showValue <$> (args !? #no_bias :: Maybe Bool),
               ("cudnn_tune",) . showValue <$>
                 (args !? #cudnn_tune ::
                    Maybe (Maybe (EnumType '["fastest", "limited_workspace", "off"]))),
               ("cudnn_off",) . showValue <$> (args !? #cudnn_off :: Maybe Bool),
               ("layout",) . showValue <$>
                 (args !? #layout ::
                    Maybe
                      (Maybe (EnumType '["NCDHW", "NCHW", "NCW", "NDHWC", "NHWC"])))]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("bias",) <$> (args !? #bias :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Deconvolution"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_Deconvolution_upd ::
                   forall args . Fullfilled "_Deconvolution(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_Deconvolution(ndarray)" args -> IO ()
_Deconvolution_upd outputs args
  = let scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> (args !? #kernel :: Maybe [Int]),
               ("stride",) . showValue <$> (args !? #stride :: Maybe [Int]),
               ("dilate",) . showValue <$> (args !? #dilate :: Maybe [Int]),
               ("pad",) . showValue <$> (args !? #pad :: Maybe [Int]),
               ("adj",) . showValue <$> (args !? #adj :: Maybe [Int]),
               ("target_shape",) . showValue <$>
                 (args !? #target_shape :: Maybe [Int]),
               ("num_filter",) . showValue <$> (args !? #num_filter :: Maybe Int),
               ("num_group",) . showValue <$> (args !? #num_group :: Maybe Int),
               ("workspace",) . showValue <$> (args !? #workspace :: Maybe Int),
               ("no_bias",) . showValue <$> (args !? #no_bias :: Maybe Bool),
               ("cudnn_tune",) . showValue <$>
                 (args !? #cudnn_tune ::
                    Maybe (Maybe (EnumType '["fastest", "limited_workspace", "off"]))),
               ("cudnn_off",) . showValue <$> (args !? #cudnn_off :: Maybe Bool),
               ("layout",) . showValue <$>
                 (args !? #layout ::
                    Maybe
                      (Maybe (EnumType '["NCDHW", "NCHW", "NCW", "NDHWC", "NHWC"])))]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("bias",) <$> (args !? #bias :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Deconvolution"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_Deconvolution(ndarray)" =
     '[]

_backward_Deconvolution ::
                        forall args . Fullfilled "_backward_Deconvolution(ndarray)" args =>
                          ArgsHMap "_backward_Deconvolution(ndarray)" args ->
                            IO [NDArrayHandle]
_backward_Deconvolution args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Deconvolution"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_Deconvolution_upd ::
                            forall args . Fullfilled "_backward_Deconvolution(ndarray)" args =>
                              [NDArrayHandle] ->
                                ArgsHMap "_backward_Deconvolution(ndarray)" args -> IO ()
_backward_Deconvolution_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Deconvolution"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_BilinearSampler(ndarray)" =
     '[ '("data", AttrOpt NDArrayHandle),
        '("grid", AttrOpt NDArrayHandle)]

_BilinearSampler ::
                 forall args . Fullfilled "_BilinearSampler(ndarray)" args =>
                   ArgsHMap "_BilinearSampler(ndarray)" args -> IO [NDArrayHandle]
_BilinearSampler args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("grid",) <$> (args !? #grid :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "BilinearSampler"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_BilinearSampler_upd ::
                     forall args . Fullfilled "_BilinearSampler(ndarray)" args =>
                       [NDArrayHandle] ->
                         ArgsHMap "_BilinearSampler(ndarray)" args -> IO ()
_BilinearSampler_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("grid",) <$> (args !? #grid :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "BilinearSampler"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_BilinearSampler(ndarray)" =
     '[]

_backward_BilinearSampler ::
                          forall args .
                            Fullfilled "_backward_BilinearSampler(ndarray)" args =>
                            ArgsHMap "_backward_BilinearSampler(ndarray)" args ->
                              IO [NDArrayHandle]
_backward_BilinearSampler args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_BilinearSampler"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_BilinearSampler_upd ::
                              forall args .
                                Fullfilled "_backward_BilinearSampler(ndarray)" args =>
                                [NDArrayHandle] ->
                                  ArgsHMap "_backward_BilinearSampler(ndarray)" args -> IO ()
_backward_BilinearSampler_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_BilinearSampler"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_ROIPooling(ndarray)" =
     '[ '("pooled_size", AttrReq [Int]),
        '("spatial_scale", AttrReq Float),
        '("data", AttrOpt NDArrayHandle), '("rois", AttrOpt NDArrayHandle)]

_ROIPooling ::
            forall args . Fullfilled "_ROIPooling(ndarray)" args =>
              ArgsHMap "_ROIPooling(ndarray)" args -> IO [NDArrayHandle]
_ROIPooling args
  = let scalarArgs
          = catMaybes
              [("pooled_size",) . showValue <$>
                 (args !? #pooled_size :: Maybe [Int]),
               ("spatial_scale",) . showValue <$>
                 (args !? #spatial_scale :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("rois",) <$> (args !? #rois :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "ROIPooling"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_ROIPooling_upd ::
                forall args . Fullfilled "_ROIPooling(ndarray)" args =>
                  [NDArrayHandle] -> ArgsHMap "_ROIPooling(ndarray)" args -> IO ()
_ROIPooling_upd outputs args
  = let scalarArgs
          = catMaybes
              [("pooled_size",) . showValue <$>
                 (args !? #pooled_size :: Maybe [Int]),
               ("spatial_scale",) . showValue <$>
                 (args !? #spatial_scale :: Maybe Float)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("rois",) <$> (args !? #rois :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "ROIPooling"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_ROIPooling(ndarray)" = '[]

_backward_ROIPooling ::
                     forall args . Fullfilled "_backward_ROIPooling(ndarray)" args =>
                       ArgsHMap "_backward_ROIPooling(ndarray)" args -> IO [NDArrayHandle]
_backward_ROIPooling args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_ROIPooling"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_ROIPooling_upd ::
                         forall args . Fullfilled "_backward_ROIPooling(ndarray)" args =>
                           [NDArrayHandle] ->
                             ArgsHMap "_backward_ROIPooling(ndarray)" args -> IO ()
_backward_ROIPooling_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_ROIPooling"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_SliceChannel(ndarray)" = '[]

_backward_SliceChannel ::
                       forall args . Fullfilled "_backward_SliceChannel(ndarray)" args =>
                         ArgsHMap "_backward_SliceChannel(ndarray)" args ->
                           IO [NDArrayHandle]
_backward_SliceChannel args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SliceChannel"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_SliceChannel_upd ::
                           forall args . Fullfilled "_backward_SliceChannel(ndarray)" args =>
                             [NDArrayHandle] ->
                               ArgsHMap "_backward_SliceChannel(ndarray)" args -> IO ()
_backward_SliceChannel_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SliceChannel"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_CuDNNBatchNorm(ndarray)" =
     '[]

_backward_CuDNNBatchNorm ::
                         forall args .
                           Fullfilled "_backward_CuDNNBatchNorm(ndarray)" args =>
                           ArgsHMap "_backward_CuDNNBatchNorm(ndarray)" args ->
                             IO [NDArrayHandle]
_backward_CuDNNBatchNorm args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_CuDNNBatchNorm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_CuDNNBatchNorm_upd ::
                             forall args .
                               Fullfilled "_backward_CuDNNBatchNorm(ndarray)" args =>
                               [NDArrayHandle] ->
                                 ArgsHMap "_backward_CuDNNBatchNorm(ndarray)" args -> IO ()
_backward_CuDNNBatchNorm_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_CuDNNBatchNorm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_FullyConnected(ndarray)" =
     '[ '("num_hidden", AttrReq Int), '("no_bias", AttrOpt Bool),
        '("flatten", AttrOpt Bool), '("data", AttrOpt NDArrayHandle),
        '("weight", AttrOpt NDArrayHandle),
        '("bias", AttrOpt NDArrayHandle)]

_FullyConnected ::
                forall args . Fullfilled "_FullyConnected(ndarray)" args =>
                  ArgsHMap "_FullyConnected(ndarray)" args -> IO [NDArrayHandle]
_FullyConnected args
  = let scalarArgs
          = catMaybes
              [("num_hidden",) . showValue <$>
                 (args !? #num_hidden :: Maybe Int),
               ("no_bias",) . showValue <$> (args !? #no_bias :: Maybe Bool),
               ("flatten",) . showValue <$> (args !? #flatten :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("bias",) <$> (args !? #bias :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "FullyConnected"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_FullyConnected_upd ::
                    forall args . Fullfilled "_FullyConnected(ndarray)" args =>
                      [NDArrayHandle] ->
                        ArgsHMap "_FullyConnected(ndarray)" args -> IO ()
_FullyConnected_upd outputs args
  = let scalarArgs
          = catMaybes
              [("num_hidden",) . showValue <$>
                 (args !? #num_hidden :: Maybe Int),
               ("no_bias",) . showValue <$> (args !? #no_bias :: Maybe Bool),
               ("flatten",) . showValue <$> (args !? #flatten :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("bias",) <$> (args !? #bias :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "FullyConnected"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_FullyConnected(ndarray)" =
     '[]

_backward_FullyConnected ::
                         forall args .
                           Fullfilled "_backward_FullyConnected(ndarray)" args =>
                           ArgsHMap "_backward_FullyConnected(ndarray)" args ->
                             IO [NDArrayHandle]
_backward_FullyConnected args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_FullyConnected"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_FullyConnected_upd ::
                             forall args .
                               Fullfilled "_backward_FullyConnected(ndarray)" args =>
                               [NDArrayHandle] ->
                                 ArgsHMap "_backward_FullyConnected(ndarray)" args -> IO ()
_backward_FullyConnected_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_FullyConnected"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_Convolution(ndarray)" =
     '[ '("kernel", AttrReq [Int]), '("stride", AttrOpt [Int]),
        '("dilate", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("num_filter", AttrReq Int), '("num_group", AttrOpt Int),
        '("workspace", AttrOpt Int), '("no_bias", AttrOpt Bool),
        '("cudnn_tune",
          AttrOpt
            (Maybe (EnumType '["fastest", "limited_workspace", "off"]))),
        '("cudnn_off", AttrOpt Bool),
        '("layout",
          AttrOpt
            (Maybe (EnumType '["NCDHW", "NCHW", "NCW", "NDHWC", "NHWC"]))),
        '("data", AttrOpt NDArrayHandle),
        '("weight", AttrOpt NDArrayHandle),
        '("bias", AttrOpt NDArrayHandle)]

_Convolution ::
             forall args . Fullfilled "_Convolution(ndarray)" args =>
               ArgsHMap "_Convolution(ndarray)" args -> IO [NDArrayHandle]
_Convolution args
  = let scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> (args !? #kernel :: Maybe [Int]),
               ("stride",) . showValue <$> (args !? #stride :: Maybe [Int]),
               ("dilate",) . showValue <$> (args !? #dilate :: Maybe [Int]),
               ("pad",) . showValue <$> (args !? #pad :: Maybe [Int]),
               ("num_filter",) . showValue <$> (args !? #num_filter :: Maybe Int),
               ("num_group",) . showValue <$> (args !? #num_group :: Maybe Int),
               ("workspace",) . showValue <$> (args !? #workspace :: Maybe Int),
               ("no_bias",) . showValue <$> (args !? #no_bias :: Maybe Bool),
               ("cudnn_tune",) . showValue <$>
                 (args !? #cudnn_tune ::
                    Maybe (Maybe (EnumType '["fastest", "limited_workspace", "off"]))),
               ("cudnn_off",) . showValue <$> (args !? #cudnn_off :: Maybe Bool),
               ("layout",) . showValue <$>
                 (args !? #layout ::
                    Maybe
                      (Maybe (EnumType '["NCDHW", "NCHW", "NCW", "NDHWC", "NHWC"])))]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("bias",) <$> (args !? #bias :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Convolution"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_Convolution_upd ::
                 forall args . Fullfilled "_Convolution(ndarray)" args =>
                   [NDArrayHandle] -> ArgsHMap "_Convolution(ndarray)" args -> IO ()
_Convolution_upd outputs args
  = let scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> (args !? #kernel :: Maybe [Int]),
               ("stride",) . showValue <$> (args !? #stride :: Maybe [Int]),
               ("dilate",) . showValue <$> (args !? #dilate :: Maybe [Int]),
               ("pad",) . showValue <$> (args !? #pad :: Maybe [Int]),
               ("num_filter",) . showValue <$> (args !? #num_filter :: Maybe Int),
               ("num_group",) . showValue <$> (args !? #num_group :: Maybe Int),
               ("workspace",) . showValue <$> (args !? #workspace :: Maybe Int),
               ("no_bias",) . showValue <$> (args !? #no_bias :: Maybe Bool),
               ("cudnn_tune",) . showValue <$>
                 (args !? #cudnn_tune ::
                    Maybe (Maybe (EnumType '["fastest", "limited_workspace", "off"]))),
               ("cudnn_off",) . showValue <$> (args !? #cudnn_off :: Maybe Bool),
               ("layout",) . showValue <$>
                 (args !? #layout ::
                    Maybe
                      (Maybe (EnumType '["NCDHW", "NCHW", "NCW", "NDHWC", "NHWC"])))]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("weight",) <$> (args !? #weight :: Maybe NDArrayHandle),
               ("bias",) <$> (args !? #bias :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Convolution"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_Convolution(ndarray)" = '[]

_backward_Convolution ::
                      forall args . Fullfilled "_backward_Convolution(ndarray)" args =>
                        ArgsHMap "_backward_Convolution(ndarray)" args ->
                          IO [NDArrayHandle]
_backward_Convolution args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Convolution"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_Convolution_upd ::
                          forall args . Fullfilled "_backward_Convolution(ndarray)" args =>
                            [NDArrayHandle] ->
                              ArgsHMap "_backward_Convolution(ndarray)" args -> IO ()
_backward_Convolution_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Convolution"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_L2Normalization(ndarray)" =
     '[ '("eps", AttrOpt Float),
        '("mode", AttrOpt (EnumType '["channel", "instance", "spatial"])),
        '("data", AttrOpt NDArrayHandle)]

_L2Normalization ::
                 forall args . Fullfilled "_L2Normalization(ndarray)" args =>
                   ArgsHMap "_L2Normalization(ndarray)" args -> IO [NDArrayHandle]
_L2Normalization args
  = let scalarArgs
          = catMaybes
              [("eps",) . showValue <$> (args !? #eps :: Maybe Float),
               ("mode",) . showValue <$>
                 (args !? #mode ::
                    Maybe (EnumType '["channel", "instance", "spatial"]))]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "L2Normalization"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_L2Normalization_upd ::
                     forall args . Fullfilled "_L2Normalization(ndarray)" args =>
                       [NDArrayHandle] ->
                         ArgsHMap "_L2Normalization(ndarray)" args -> IO ()
_L2Normalization_upd outputs args
  = let scalarArgs
          = catMaybes
              [("eps",) . showValue <$> (args !? #eps :: Maybe Float),
               ("mode",) . showValue <$>
                 (args !? #mode ::
                    Maybe (EnumType '["channel", "instance", "spatial"]))]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "L2Normalization"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_L2Normalization(ndarray)" =
     '[]

_backward_L2Normalization ::
                          forall args .
                            Fullfilled "_backward_L2Normalization(ndarray)" args =>
                            ArgsHMap "_backward_L2Normalization(ndarray)" args ->
                              IO [NDArrayHandle]
_backward_L2Normalization args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_L2Normalization"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_L2Normalization_upd ::
                              forall args .
                                Fullfilled "_backward_L2Normalization(ndarray)" args =>
                                [NDArrayHandle] ->
                                  ArgsHMap "_backward_L2Normalization(ndarray)" args -> IO ()
_backward_L2Normalization_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_L2Normalization"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_Concat(ndarray)" = '[]

_backward_Concat ::
                 forall args . Fullfilled "_backward_Concat(ndarray)" args =>
                   ArgsHMap "_backward_Concat(ndarray)" args -> IO [NDArrayHandle]
_backward_Concat args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Concat"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_Concat_upd ::
                     forall args . Fullfilled "_backward_Concat(ndarray)" args =>
                       [NDArrayHandle] ->
                         ArgsHMap "_backward_Concat(ndarray)" args -> IO ()
_backward_Concat_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Concat"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_RNN(ndarray)" =
     '[ '("state_size", AttrReq Int), '("num_layers", AttrReq Int),
        '("bidirectional", AttrOpt Bool),
        '("mode",
          AttrReq (EnumType '["gru", "lstm", "rnn_relu", "rnn_tanh"])),
        '("p", AttrOpt Float), '("state_outputs", AttrOpt Bool),
        '("data", AttrOpt NDArrayHandle),
        '("parameters", AttrOpt NDArrayHandle),
        '("state", AttrOpt NDArrayHandle),
        '("state_cell", AttrOpt NDArrayHandle)]

_RNN ::
     forall args . Fullfilled "_RNN(ndarray)" args =>
       ArgsHMap "_RNN(ndarray)" args -> IO [NDArrayHandle]
_RNN args
  = let scalarArgs
          = catMaybes
              [("state_size",) . showValue <$>
                 (args !? #state_size :: Maybe Int),
               ("num_layers",) . showValue <$> (args !? #num_layers :: Maybe Int),
               ("bidirectional",) . showValue <$>
                 (args !? #bidirectional :: Maybe Bool),
               ("mode",) . showValue <$>
                 (args !? #mode ::
                    Maybe (EnumType '["gru", "lstm", "rnn_relu", "rnn_tanh"])),
               ("p",) . showValue <$> (args !? #p :: Maybe Float),
               ("state_outputs",) . showValue <$>
                 (args !? #state_outputs :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("parameters",) <$> (args !? #parameters :: Maybe NDArrayHandle),
               ("state",) <$> (args !? #state :: Maybe NDArrayHandle),
               ("state_cell",) <$> (args !? #state_cell :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "RNN"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_RNN_upd ::
         forall args . Fullfilled "_RNN(ndarray)" args =>
           [NDArrayHandle] -> ArgsHMap "_RNN(ndarray)" args -> IO ()
_RNN_upd outputs args
  = let scalarArgs
          = catMaybes
              [("state_size",) . showValue <$>
                 (args !? #state_size :: Maybe Int),
               ("num_layers",) . showValue <$> (args !? #num_layers :: Maybe Int),
               ("bidirectional",) . showValue <$>
                 (args !? #bidirectional :: Maybe Bool),
               ("mode",) . showValue <$>
                 (args !? #mode ::
                    Maybe (EnumType '["gru", "lstm", "rnn_relu", "rnn_tanh"])),
               ("p",) . showValue <$> (args !? #p :: Maybe Float),
               ("state_outputs",) . showValue <$>
                 (args !? #state_outputs :: Maybe Bool)]
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe NDArrayHandle),
               ("parameters",) <$> (args !? #parameters :: Maybe NDArrayHandle),
               ("state",) <$> (args !? #state :: Maybe NDArrayHandle),
               ("state_cell",) <$> (args !? #state_cell :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "RNN"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_RNN(ndarray)" = '[]

_backward_RNN ::
              forall args . Fullfilled "_backward_RNN(ndarray)" args =>
                ArgsHMap "_backward_RNN(ndarray)" args -> IO [NDArrayHandle]
_backward_RNN args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_RNN"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_RNN_upd ::
                  forall args . Fullfilled "_backward_RNN(ndarray)" args =>
                    [NDArrayHandle] -> ArgsHMap "_backward_RNN(ndarray)" args -> IO ()
_backward_RNN_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_RNN"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_BatchNorm(ndarray)" = '[]

_backward_BatchNorm ::
                    forall args . Fullfilled "_backward_BatchNorm(ndarray)" args =>
                      ArgsHMap "_backward_BatchNorm(ndarray)" args -> IO [NDArrayHandle]
_backward_BatchNorm args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_BatchNorm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_BatchNorm_upd ::
                        forall args . Fullfilled "_backward_BatchNorm(ndarray)" args =>
                          [NDArrayHandle] ->
                            ArgsHMap "_backward_BatchNorm(ndarray)" args -> IO ()
_backward_BatchNorm_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_BatchNorm"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_Dropout(ndarray)" =
     '[ '("p", AttrOpt Float),
        '("mode", AttrOpt (EnumType '["always", "training"])),
        '("data", AttrOpt NDArrayHandle)]

_Dropout ::
         forall args . Fullfilled "_Dropout(ndarray)" args =>
           ArgsHMap "_Dropout(ndarray)" args -> IO [NDArrayHandle]
_Dropout args
  = let scalarArgs
          = catMaybes
              [("p",) . showValue <$> (args !? #p :: Maybe Float),
               ("mode",) . showValue <$>
                 (args !? #mode :: Maybe (EnumType '["always", "training"]))]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Dropout"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_Dropout_upd ::
             forall args . Fullfilled "_Dropout(ndarray)" args =>
               [NDArrayHandle] -> ArgsHMap "_Dropout(ndarray)" args -> IO ()
_Dropout_upd outputs args
  = let scalarArgs
          = catMaybes
              [("p",) . showValue <$> (args !? #p :: Maybe Float),
               ("mode",) . showValue <$>
                 (args !? #mode :: Maybe (EnumType '["always", "training"]))]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Dropout"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_Dropout(ndarray)" = '[]

_backward_Dropout ::
                  forall args . Fullfilled "_backward_Dropout(ndarray)" args =>
                    ArgsHMap "_backward_Dropout(ndarray)" args -> IO [NDArrayHandle]
_backward_Dropout args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Dropout"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_Dropout_upd ::
                      forall args . Fullfilled "_backward_Dropout(ndarray)" args =>
                        [NDArrayHandle] ->
                          ArgsHMap "_backward_Dropout(ndarray)" args -> IO ()
_backward_Dropout_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Dropout"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_CrossDeviceCopy(ndarray)" = '[]

_CrossDeviceCopy ::
                 forall args . Fullfilled "_CrossDeviceCopy(ndarray)" args =>
                   ArgsHMap "_CrossDeviceCopy(ndarray)" args -> IO [NDArrayHandle]
_CrossDeviceCopy args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_CrossDeviceCopy"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_CrossDeviceCopy_upd ::
                     forall args . Fullfilled "_CrossDeviceCopy(ndarray)" args =>
                       [NDArrayHandle] ->
                         ArgsHMap "_CrossDeviceCopy(ndarray)" args -> IO ()
_CrossDeviceCopy_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_CrossDeviceCopy"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward__CrossDeviceCopy(ndarray)" =
     '[]

_backward__CrossDeviceCopy ::
                           forall args .
                             Fullfilled "_backward__CrossDeviceCopy(ndarray)" args =>
                             ArgsHMap "_backward__CrossDeviceCopy(ndarray)" args ->
                               IO [NDArrayHandle]
_backward__CrossDeviceCopy args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__CrossDeviceCopy"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward__CrossDeviceCopy_upd ::
                               forall args .
                                 Fullfilled "_backward__CrossDeviceCopy(ndarray)" args =>
                                 [NDArrayHandle] ->
                                   ArgsHMap "_backward__CrossDeviceCopy(ndarray)" args -> IO ()
_backward__CrossDeviceCopy_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__CrossDeviceCopy"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_SoftmaxActivation(ndarray)" =
     '[ '("mode", AttrOpt (EnumType '["channel", "instance"])),
        '("data", AttrOpt NDArrayHandle)]

_SoftmaxActivation ::
                   forall args . Fullfilled "_SoftmaxActivation(ndarray)" args =>
                     ArgsHMap "_SoftmaxActivation(ndarray)" args -> IO [NDArrayHandle]
_SoftmaxActivation args
  = let scalarArgs
          = catMaybes
              [("mode",) . showValue <$>
                 (args !? #mode :: Maybe (EnumType '["channel", "instance"]))]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SoftmaxActivation"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_SoftmaxActivation_upd ::
                       forall args . Fullfilled "_SoftmaxActivation(ndarray)" args =>
                         [NDArrayHandle] ->
                           ArgsHMap "_SoftmaxActivation(ndarray)" args -> IO ()
_SoftmaxActivation_upd outputs args
  = let scalarArgs
          = catMaybes
              [("mode",) . showValue <$>
                 (args !? #mode :: Maybe (EnumType '["channel", "instance"]))]
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SoftmaxActivation"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_backward_SoftmaxActivation(ndarray)"
     = '[]

_backward_SoftmaxActivation ::
                            forall args .
                              Fullfilled "_backward_SoftmaxActivation(ndarray)" args =>
                              ArgsHMap "_backward_SoftmaxActivation(ndarray)" args ->
                                IO [NDArrayHandle]
_backward_SoftmaxActivation args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SoftmaxActivation"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_backward_SoftmaxActivation_upd ::
                                forall args .
                                  Fullfilled "_backward_SoftmaxActivation(ndarray)" args =>
                                  [NDArrayHandle] ->
                                    ArgsHMap "_backward_SoftmaxActivation(ndarray)" args -> IO ()
_backward_SoftmaxActivation_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SoftmaxActivation"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_set_value(ndarray)" =
     '[ '("src", AttrOpt Float)]

_set_value ::
           forall args . Fullfilled "_set_value(ndarray)" args =>
             ArgsHMap "_set_value(ndarray)" args -> IO [NDArrayHandle]
_set_value args
  = let scalarArgs
          = catMaybes
              [("src",) . showValue <$> (args !? #src :: Maybe Float)]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_set_value"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_set_value_upd ::
               forall args . Fullfilled "_set_value(ndarray)" args =>
                 [NDArrayHandle] -> ArgsHMap "_set_value(ndarray)" args -> IO ()
_set_value_upd outputs args
  = let scalarArgs
          = catMaybes
              [("src",) . showValue <$> (args !? #src :: Maybe Float)]
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_set_value"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_onehot_encode(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

_onehot_encode ::
               forall args . Fullfilled "_onehot_encode(ndarray)" args =>
                 ArgsHMap "_onehot_encode(ndarray)" args -> IO [NDArrayHandle]
_onehot_encode args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_onehot_encode"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_onehot_encode_upd ::
                   forall args . Fullfilled "_onehot_encode(ndarray)" args =>
                     [NDArrayHandle] -> ArgsHMap "_onehot_encode(ndarray)" args -> IO ()
_onehot_encode_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_onehot_encode"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "choose_element_0index(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("rhs", AttrOpt NDArrayHandle)]

choose_element_0index ::
                      forall args . Fullfilled "choose_element_0index(ndarray)" args =>
                        ArgsHMap "choose_element_0index(ndarray)" args ->
                          IO [NDArrayHandle]
choose_element_0index args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "choose_element_0index"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

choose_element_0index_upd ::
                          forall args . Fullfilled "choose_element_0index(ndarray)" args =>
                            [NDArrayHandle] ->
                              ArgsHMap "choose_element_0index(ndarray)" args -> IO ()
choose_element_0index_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "choose_element_0index"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "fill_element_0index(ndarray)" =
     '[ '("lhs", AttrOpt NDArrayHandle),
        '("mhs", AttrOpt NDArrayHandle), '("rhs", AttrOpt NDArrayHandle)]

fill_element_0index ::
                    forall args . Fullfilled "fill_element_0index(ndarray)" args =>
                      ArgsHMap "fill_element_0index(ndarray)" args -> IO [NDArrayHandle]
fill_element_0index args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("mhs",) <$> (args !? #mhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "fill_element_0index"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

fill_element_0index_upd ::
                        forall args . Fullfilled "fill_element_0index(ndarray)" args =>
                          [NDArrayHandle] ->
                            ArgsHMap "fill_element_0index(ndarray)" args -> IO ()
fill_element_0index_upd outputs args
  = let scalarArgs = catMaybes []
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe NDArrayHandle),
               ("mhs",) <$> (args !? #mhs :: Maybe NDArrayHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "fill_element_0index"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()

type instance ParameterList "_imdecode(ndarray)" =
     '[ '("index", AttrOpt Int), '("x0", AttrOpt Int),
        '("y0", AttrOpt Int), '("x1", AttrOpt Int), '("y1", AttrOpt Int),
        '("c", AttrOpt Int), '("size", AttrOpt Int),
        '("mean", AttrOpt NDArrayHandle)]

_imdecode ::
          forall args . Fullfilled "_imdecode(ndarray)" args =>
            ArgsHMap "_imdecode(ndarray)" args -> IO [NDArrayHandle]
_imdecode args
  = let scalarArgs
          = catMaybes
              [("index",) . showValue <$> (args !? #index :: Maybe Int),
               ("x0",) . showValue <$> (args !? #x0 :: Maybe Int),
               ("y0",) . showValue <$> (args !? #y0 :: Maybe Int),
               ("x1",) . showValue <$> (args !? #x1 :: Maybe Int),
               ("y1",) . showValue <$> (args !? #y1 :: Maybe Int),
               ("c",) . showValue <$> (args !? #c :: Maybe Int),
               ("size",) . showValue <$> (args !? #size :: Maybe Int)]
        tensorArgs
          = catMaybes [("mean",) <$> (args !? #mean :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_imdecode"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        Nothing
         return listndarr

_imdecode_upd ::
              forall args . Fullfilled "_imdecode(ndarray)" args =>
                [NDArrayHandle] -> ArgsHMap "_imdecode(ndarray)" args -> IO ()
_imdecode_upd outputs args
  = let scalarArgs
          = catMaybes
              [("index",) . showValue <$> (args !? #index :: Maybe Int),
               ("x0",) . showValue <$> (args !? #x0 :: Maybe Int),
               ("y0",) . showValue <$> (args !? #y0 :: Maybe Int),
               ("x1",) . showValue <$> (args !? #x1 :: Maybe Int),
               ("y1",) . showValue <$> (args !? #y1 :: Maybe Int),
               ("c",) . showValue <$> (args !? #c :: Maybe Int),
               ("size",) . showValue <$> (args !? #size :: Maybe Int)]
        tensorArgs
          = catMaybes [("mean",) <$> (args !? #mean :: Maybe NDArrayHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_imdecode"
         listndarr <- mxImperativeInvoke (fromOpHandle op) tensorvals
                        scalarArgs
                        (Just outputs)
         return ()