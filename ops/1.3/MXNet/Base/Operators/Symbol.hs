module MXNet.Base.Operators.Symbol where
import MXNet.Base.Raw
import MXNet.Base.Spec.Operator
import MXNet.Base.Spec.HMap
import Data.Maybe (catMaybes, fromMaybe)

type instance ParameterList "_Activation(symbol)" =
     '[ '("act_type",
          AttrReq
            (EnumType '["relu", "sigmoid", "softrelu", "softsign", "tanh"])),
        '("data", AttrOpt SymbolHandle)]

_Activation ::
            forall args . Fullfilled "_Activation(symbol)" args =>
              String -> ArgsHMap "_Activation(symbol)" args -> IO SymbolHandle
_Activation name args
  = let scalarArgs
          = catMaybes
              [("act_type",) . showValue <$>
                 (args !? #act_type ::
                    Maybe
                      (EnumType '["relu", "sigmoid", "softrelu", "softsign", "tanh"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Activation"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_BatchNorm(symbol)" =
     '[ '("eps", AttrOpt Double), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool), '("axis", AttrOpt Int),
        '("cudnn_off", AttrOpt Bool), '("data", AttrOpt SymbolHandle),
        '("gamma", AttrOpt SymbolHandle), '("beta", AttrOpt SymbolHandle),
        '("moving_mean", AttrOpt SymbolHandle),
        '("moving_var", AttrOpt SymbolHandle)]

_BatchNorm ::
           forall args . Fullfilled "_BatchNorm(symbol)" args =>
             String -> ArgsHMap "_BatchNorm(symbol)" args -> IO SymbolHandle
_BatchNorm name args
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
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("gamma",) <$> (args !? #gamma :: Maybe SymbolHandle),
               ("beta",) <$> (args !? #beta :: Maybe SymbolHandle),
               ("moving_mean",) <$> (args !? #moving_mean :: Maybe SymbolHandle),
               ("moving_var",) <$> (args !? #moving_var :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "BatchNorm"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_BatchNorm_v1(symbol)" =
     '[ '("eps", AttrOpt Float), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool),
        '("data", AttrOpt SymbolHandle), '("gamma", AttrOpt SymbolHandle),
        '("beta", AttrOpt SymbolHandle)]

_BatchNorm_v1 ::
              forall args . Fullfilled "_BatchNorm_v1(symbol)" args =>
                String -> ArgsHMap "_BatchNorm_v1(symbol)" args -> IO SymbolHandle
_BatchNorm_v1 name args
  = let scalarArgs
          = catMaybes
              [("eps",) . showValue <$> (args !? #eps :: Maybe Float),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float),
               ("fix_gamma",) . showValue <$> (args !? #fix_gamma :: Maybe Bool),
               ("use_global_stats",) . showValue <$>
                 (args !? #use_global_stats :: Maybe Bool),
               ("output_mean_var",) . showValue <$>
                 (args !? #output_mean_var :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("gamma",) <$> (args !? #gamma :: Maybe SymbolHandle),
               ("beta",) <$> (args !? #beta :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "BatchNorm_v1"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_BilinearSampler(symbol)" =
     '[ '("data", AttrOpt SymbolHandle),
        '("grid", AttrOpt SymbolHandle)]

_BilinearSampler ::
                 forall args . Fullfilled "_BilinearSampler(symbol)" args =>
                   String ->
                     ArgsHMap "_BilinearSampler(symbol)" args -> IO SymbolHandle
_BilinearSampler name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("grid",) <$> (args !? #grid :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "BilinearSampler"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_BlockGrad(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

_BlockGrad ::
           forall args . Fullfilled "_BlockGrad(symbol)" args =>
             String -> ArgsHMap "_BlockGrad(symbol)" args -> IO SymbolHandle
_BlockGrad name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "BlockGrad"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_Cast(symbol)" =
     '[ '("dtype",
          AttrReq
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"])),
        '("data", AttrOpt SymbolHandle)]

_Cast ::
      forall args . Fullfilled "_Cast(symbol)" args =>
        String -> ArgsHMap "_Cast(symbol)" args -> IO SymbolHandle
_Cast name args
  = let scalarArgs
          = catMaybes
              [("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["float16", "float32", "float64", "int32", "int64", "int8",
                           "uint8"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Cast"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_Concat(symbol)" =
     '[ '("num_args", AttrReq Int), '("dim", AttrOpt Int),
        '("data", AttrOpt [SymbolHandle])]

_Concat ::
        forall args . Fullfilled "_Concat(symbol)" args =>
          String -> ArgsHMap "_Concat(symbol)" args -> IO SymbolHandle
_Concat name args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("dim",) . showValue <$> (args !? #dim :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
        array = fromMaybe [] (args !? #data :: Maybe [SymbolHandle])
      in
      do op <- nnGetOpHandle "Concat"
         sym <- if hasKey args #num_args then
                  mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys scalarvals
                  else
                  mxSymbolCreateAtomicSymbol (fromOpHandle op)
                    ("num_args" : scalarkeys)
                    (showValue (length array) : scalarvals)
         mxSymbolCompose sym name Nothing array
         return sym

type instance ParameterList "_Convolution(symbol)" =
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
        '("data", AttrOpt SymbolHandle), '("weight", AttrOpt SymbolHandle),
        '("bias", AttrOpt SymbolHandle)]

_Convolution ::
             forall args . Fullfilled "_Convolution(symbol)" args =>
               String -> ArgsHMap "_Convolution(symbol)" args -> IO SymbolHandle
_Convolution name args
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
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("weight",) <$> (args !? #weight :: Maybe SymbolHandle),
               ("bias",) <$> (args !? #bias :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Convolution"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_Convolution_v1(symbol)" =
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
        '("data", AttrOpt SymbolHandle), '("weight", AttrOpt SymbolHandle),
        '("bias", AttrOpt SymbolHandle)]

_Convolution_v1 ::
                forall args . Fullfilled "_Convolution_v1(symbol)" args =>
                  String ->
                    ArgsHMap "_Convolution_v1(symbol)" args -> IO SymbolHandle
_Convolution_v1 name args
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
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("weight",) <$> (args !? #weight :: Maybe SymbolHandle),
               ("bias",) <$> (args !? #bias :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Convolution_v1"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_Correlation(symbol)" =
     '[ '("kernel_size", AttrOpt Int),
        '("max_displacement", AttrOpt Int), '("stride1", AttrOpt Int),
        '("stride2", AttrOpt Int), '("pad_size", AttrOpt Int),
        '("is_multiply", AttrOpt Bool), '("data1", AttrOpt SymbolHandle),
        '("data2", AttrOpt SymbolHandle)]

_Correlation ::
             forall args . Fullfilled "_Correlation(symbol)" args =>
               String -> ArgsHMap "_Correlation(symbol)" args -> IO SymbolHandle
_Correlation name args
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
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data1",) <$> (args !? #data1 :: Maybe SymbolHandle),
               ("data2",) <$> (args !? #data2 :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Correlation"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_Crop(symbol)" =
     '[ '("num_args", AttrReq Int), '("offset", AttrOpt [Int]),
        '("h_w", AttrOpt [Int]), '("center_crop", AttrOpt Bool),
        '("data", AttrOpt [SymbolHandle])]

_Crop ::
      forall args . Fullfilled "_Crop(symbol)" args =>
        String -> ArgsHMap "_Crop(symbol)" args -> IO SymbolHandle
_Crop name args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("offset",) . showValue <$> (args !? #offset :: Maybe [Int]),
               ("h_w",) . showValue <$> (args !? #h_w :: Maybe [Int]),
               ("center_crop",) . showValue <$>
                 (args !? #center_crop :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
        array = fromMaybe [] (args !? #data :: Maybe [SymbolHandle])
      in
      do op <- nnGetOpHandle "Crop"
         sym <- if hasKey args #num_args then
                  mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys scalarvals
                  else
                  mxSymbolCreateAtomicSymbol (fromOpHandle op)
                    ("num_args" : scalarkeys)
                    (showValue (length array) : scalarvals)
         mxSymbolCompose sym name Nothing array
         return sym

type instance ParameterList "_Custom(symbol)" =
     '[ '("op_type", AttrOpt String), '("data", AttrOpt [SymbolHandle])]

_Custom ::
        forall args .
          (Fullfilled "_Custom(symbol)" args,
           PopKey (ArgOf "_Custom(symbol)") args "data",
           Dump (PopResult (ArgOf "_Custom(symbol)") args "data")) =>
          String -> ArgsHMap "_Custom(symbol)" args -> IO SymbolHandle
_Custom name args
  = let scalarArgs = dump (pop args #data)
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
        array = fromMaybe [] (args !? #data :: Maybe [SymbolHandle])
      in
      do op <- nnGetOpHandle "Custom"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name Nothing array
         return sym

type instance ParameterList "_Deconvolution(symbol)" =
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
        '("data", AttrOpt SymbolHandle), '("weight", AttrOpt SymbolHandle),
        '("bias", AttrOpt SymbolHandle)]

_Deconvolution ::
               forall args . Fullfilled "_Deconvolution(symbol)" args =>
                 String -> ArgsHMap "_Deconvolution(symbol)" args -> IO SymbolHandle
_Deconvolution name args
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
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("weight",) <$> (args !? #weight :: Maybe SymbolHandle),
               ("bias",) <$> (args !? #bias :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Deconvolution"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_Dropout(symbol)" =
     '[ '("p", AttrOpt Float),
        '("mode", AttrOpt (EnumType '["always", "training"])),
        '("axes", AttrOpt [Int]), '("data", AttrOpt SymbolHandle)]

_Dropout ::
         forall args . Fullfilled "_Dropout(symbol)" args =>
           String -> ArgsHMap "_Dropout(symbol)" args -> IO SymbolHandle
_Dropout name args
  = let scalarArgs
          = catMaybes
              [("p",) . showValue <$> (args !? #p :: Maybe Float),
               ("mode",) . showValue <$>
                 (args !? #mode :: Maybe (EnumType '["always", "training"])),
               ("axes",) . showValue <$> (args !? #axes :: Maybe [Int])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Dropout"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_Embedding(symbol)" =
     '[ '("input_dim", AttrReq Int), '("output_dim", AttrReq Int),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"])),
        '("sparse_grad", AttrOpt Bool), '("data", AttrOpt SymbolHandle),
        '("weight", AttrOpt SymbolHandle)]

_Embedding ::
           forall args . Fullfilled "_Embedding(symbol)" args =>
             String -> ArgsHMap "_Embedding(symbol)" args -> IO SymbolHandle
_Embedding name args
  = let scalarArgs
          = catMaybes
              [("input_dim",) . showValue <$> (args !? #input_dim :: Maybe Int),
               ("output_dim",) . showValue <$> (args !? #output_dim :: Maybe Int),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["float16", "float32", "float64", "int32", "int64", "int8",
                           "uint8"])),
               ("sparse_grad",) . showValue <$>
                 (args !? #sparse_grad :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("weight",) <$> (args !? #weight :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Embedding"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_Flatten(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

_Flatten ::
         forall args . Fullfilled "_Flatten(symbol)" args =>
           String -> ArgsHMap "_Flatten(symbol)" args -> IO SymbolHandle
_Flatten name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Flatten"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_FullyConnected(symbol)" =
     '[ '("num_hidden", AttrReq Int), '("no_bias", AttrOpt Bool),
        '("flatten", AttrOpt Bool), '("data", AttrOpt SymbolHandle),
        '("weight", AttrOpt SymbolHandle), '("bias", AttrOpt SymbolHandle)]

_FullyConnected ::
                forall args . Fullfilled "_FullyConnected(symbol)" args =>
                  String ->
                    ArgsHMap "_FullyConnected(symbol)" args -> IO SymbolHandle
_FullyConnected name args
  = let scalarArgs
          = catMaybes
              [("num_hidden",) . showValue <$>
                 (args !? #num_hidden :: Maybe Int),
               ("no_bias",) . showValue <$> (args !? #no_bias :: Maybe Bool),
               ("flatten",) . showValue <$> (args !? #flatten :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("weight",) <$> (args !? #weight :: Maybe SymbolHandle),
               ("bias",) <$> (args !? #bias :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "FullyConnected"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_GridGenerator(symbol)" =
     '[ '("transform_type", AttrReq (EnumType '["affine", "warp"])),
        '("target_shape", AttrOpt [Int]), '("data", AttrOpt SymbolHandle)]

_GridGenerator ::
               forall args . Fullfilled "_GridGenerator(symbol)" args =>
                 String -> ArgsHMap "_GridGenerator(symbol)" args -> IO SymbolHandle
_GridGenerator name args
  = let scalarArgs
          = catMaybes
              [("transform_type",) . showValue <$>
                 (args !? #transform_type :: Maybe (EnumType '["affine", "warp"])),
               ("target_shape",) . showValue <$>
                 (args !? #target_shape :: Maybe [Int])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "GridGenerator"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_IdentityAttachKLSparseReg(symbol)" =
     '[ '("sparseness_target", AttrOpt Float),
        '("penalty", AttrOpt Float), '("momentum", AttrOpt Float),
        '("data", AttrOpt SymbolHandle)]

_IdentityAttachKLSparseReg ::
                           forall args .
                             Fullfilled "_IdentityAttachKLSparseReg(symbol)" args =>
                             String ->
                               ArgsHMap "_IdentityAttachKLSparseReg(symbol)" args ->
                                 IO SymbolHandle
_IdentityAttachKLSparseReg name args
  = let scalarArgs
          = catMaybes
              [("sparseness_target",) . showValue <$>
                 (args !? #sparseness_target :: Maybe Float),
               ("penalty",) . showValue <$> (args !? #penalty :: Maybe Float),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "IdentityAttachKLSparseReg"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_InstanceNorm(symbol)" =
     '[ '("eps", AttrOpt Float), '("data", AttrOpt SymbolHandle),
        '("gamma", AttrOpt SymbolHandle), '("beta", AttrOpt SymbolHandle)]

_InstanceNorm ::
              forall args . Fullfilled "_InstanceNorm(symbol)" args =>
                String -> ArgsHMap "_InstanceNorm(symbol)" args -> IO SymbolHandle
_InstanceNorm name args
  = let scalarArgs
          = catMaybes
              [("eps",) . showValue <$> (args !? #eps :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("gamma",) <$> (args !? #gamma :: Maybe SymbolHandle),
               ("beta",) <$> (args !? #beta :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "InstanceNorm"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_L2Normalization(symbol)" =
     '[ '("eps", AttrOpt Float),
        '("mode", AttrOpt (EnumType '["channel", "instance", "spatial"])),
        '("data", AttrOpt SymbolHandle)]

_L2Normalization ::
                 forall args . Fullfilled "_L2Normalization(symbol)" args =>
                   String ->
                     ArgsHMap "_L2Normalization(symbol)" args -> IO SymbolHandle
_L2Normalization name args
  = let scalarArgs
          = catMaybes
              [("eps",) . showValue <$> (args !? #eps :: Maybe Float),
               ("mode",) . showValue <$>
                 (args !? #mode ::
                    Maybe (EnumType '["channel", "instance", "spatial"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "L2Normalization"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_LRN(symbol)" =
     '[ '("alpha", AttrOpt Float), '("beta", AttrOpt Float),
        '("knorm", AttrOpt Float), '("nsize", AttrReq Int),
        '("data", AttrOpt SymbolHandle)]

_LRN ::
     forall args . Fullfilled "_LRN(symbol)" args =>
       String -> ArgsHMap "_LRN(symbol)" args -> IO SymbolHandle
_LRN name args
  = let scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> (args !? #alpha :: Maybe Float),
               ("beta",) . showValue <$> (args !? #beta :: Maybe Float),
               ("knorm",) . showValue <$> (args !? #knorm :: Maybe Float),
               ("nsize",) . showValue <$> (args !? #nsize :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "LRN"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_LayerNorm(symbol)" =
     '[ '("axis", AttrOpt Int), '("eps", AttrOpt Float),
        '("output_mean_var", AttrOpt Bool),
        '("data", AttrOpt SymbolHandle), '("gamma", AttrOpt SymbolHandle),
        '("beta", AttrOpt SymbolHandle)]

_LayerNorm ::
           forall args . Fullfilled "_LayerNorm(symbol)" args =>
             String -> ArgsHMap "_LayerNorm(symbol)" args -> IO SymbolHandle
_LayerNorm name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("eps",) . showValue <$> (args !? #eps :: Maybe Float),
               ("output_mean_var",) . showValue <$>
                 (args !? #output_mean_var :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("gamma",) <$> (args !? #gamma :: Maybe SymbolHandle),
               ("beta",) <$> (args !? #beta :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "LayerNorm"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_LeakyReLU(symbol)" =
     '[ '("act_type",
          AttrOpt (EnumType '["elu", "leaky", "prelu", "rrelu", "selu"])),
        '("slope", AttrOpt Float), '("lower_bound", AttrOpt Float),
        '("upper_bound", AttrOpt Float), '("data", AttrOpt SymbolHandle),
        '("gamma", AttrOpt SymbolHandle)]

_LeakyReLU ::
           forall args . Fullfilled "_LeakyReLU(symbol)" args =>
             String -> ArgsHMap "_LeakyReLU(symbol)" args -> IO SymbolHandle
_LeakyReLU name args
  = let scalarArgs
          = catMaybes
              [("act_type",) . showValue <$>
                 (args !? #act_type ::
                    Maybe (EnumType '["elu", "leaky", "prelu", "rrelu", "selu"])),
               ("slope",) . showValue <$> (args !? #slope :: Maybe Float),
               ("lower_bound",) . showValue <$>
                 (args !? #lower_bound :: Maybe Float),
               ("upper_bound",) . showValue <$>
                 (args !? #upper_bound :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("gamma",) <$> (args !? #gamma :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "LeakyReLU"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_LinearRegressionOutput(symbol)" =
     '[ '("grad_scale", AttrOpt Float), '("data", AttrOpt SymbolHandle),
        '("label", AttrOpt SymbolHandle)]

_LinearRegressionOutput ::
                        forall args . Fullfilled "_LinearRegressionOutput(symbol)" args =>
                          String ->
                            ArgsHMap "_LinearRegressionOutput(symbol)" args -> IO SymbolHandle
_LinearRegressionOutput name args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("label",) <$> (args !? #label :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "LinearRegressionOutput"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_LogisticRegressionOutput(symbol)" =
     '[ '("grad_scale", AttrOpt Float), '("data", AttrOpt SymbolHandle),
        '("label", AttrOpt SymbolHandle)]

_LogisticRegressionOutput ::
                          forall args .
                            Fullfilled "_LogisticRegressionOutput(symbol)" args =>
                            String ->
                              ArgsHMap "_LogisticRegressionOutput(symbol)" args ->
                                IO SymbolHandle
_LogisticRegressionOutput name args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("label",) <$> (args !? #label :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "LogisticRegressionOutput"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_MAERegressionOutput(symbol)" =
     '[ '("grad_scale", AttrOpt Float), '("data", AttrOpt SymbolHandle),
        '("label", AttrOpt SymbolHandle)]

_MAERegressionOutput ::
                     forall args . Fullfilled "_MAERegressionOutput(symbol)" args =>
                       String ->
                         ArgsHMap "_MAERegressionOutput(symbol)" args -> IO SymbolHandle
_MAERegressionOutput name args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("label",) <$> (args !? #label :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "MAERegressionOutput"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_MakeLoss(symbol)" =
     '[ '("grad_scale", AttrOpt Float),
        '("valid_thresh", AttrOpt Float),
        '("normalization", AttrOpt (EnumType '["batch", "null", "valid"])),
        '("data", AttrOpt SymbolHandle)]

_MakeLoss ::
          forall args . Fullfilled "_MakeLoss(symbol)" args =>
            String -> ArgsHMap "_MakeLoss(symbol)" args -> IO SymbolHandle
_MakeLoss name args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float),
               ("valid_thresh",) . showValue <$>
                 (args !? #valid_thresh :: Maybe Float),
               ("normalization",) . showValue <$>
                 (args !? #normalization ::
                    Maybe (EnumType '["batch", "null", "valid"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "MakeLoss"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_Pad(symbol)" =
     '[ '("mode", AttrReq (EnumType '["constant", "edge", "reflect"])),
        '("pad_width", AttrReq [Int]), '("constant_value", AttrOpt Double),
        '("data", AttrOpt SymbolHandle)]

_Pad ::
     forall args . Fullfilled "_Pad(symbol)" args =>
       String -> ArgsHMap "_Pad(symbol)" args -> IO SymbolHandle
_Pad name args
  = let scalarArgs
          = catMaybes
              [("mode",) . showValue <$>
                 (args !? #mode ::
                    Maybe (EnumType '["constant", "edge", "reflect"])),
               ("pad_width",) . showValue <$> (args !? #pad_width :: Maybe [Int]),
               ("constant_value",) . showValue <$>
                 (args !? #constant_value :: Maybe Double)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Pad"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_Pooling(symbol)" =
     '[ '("kernel", AttrOpt [Int]),
        '("pool_type", AttrOpt (EnumType '["avg", "lp", "max", "sum"])),
        '("global_pool", AttrOpt Bool), '("cudnn_off", AttrOpt Bool),
        '("pooling_convention", AttrOpt (EnumType '["full", "valid"])),
        '("stride", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("p_value", AttrOpt (Maybe Int)),
        '("count_include_pad", AttrOpt (Maybe Bool)),
        '("data", AttrOpt SymbolHandle)]

_Pooling ::
         forall args . Fullfilled "_Pooling(symbol)" args =>
           String -> ArgsHMap "_Pooling(symbol)" args -> IO SymbolHandle
_Pooling name args
  = let scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> (args !? #kernel :: Maybe [Int]),
               ("pool_type",) . showValue <$>
                 (args !? #pool_type ::
                    Maybe (EnumType '["avg", "lp", "max", "sum"])),
               ("global_pool",) . showValue <$>
                 (args !? #global_pool :: Maybe Bool),
               ("cudnn_off",) . showValue <$> (args !? #cudnn_off :: Maybe Bool),
               ("pooling_convention",) . showValue <$>
                 (args !? #pooling_convention ::
                    Maybe (EnumType '["full", "valid"])),
               ("stride",) . showValue <$> (args !? #stride :: Maybe [Int]),
               ("pad",) . showValue <$> (args !? #pad :: Maybe [Int]),
               ("p_value",) . showValue <$>
                 (args !? #p_value :: Maybe (Maybe Int)),
               ("count_include_pad",) . showValue <$>
                 (args !? #count_include_pad :: Maybe (Maybe Bool))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Pooling"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_Pooling_v1(symbol)" =
     '[ '("kernel", AttrOpt [Int]),
        '("pool_type", AttrOpt (EnumType '["avg", "max", "sum"])),
        '("global_pool", AttrOpt Bool),
        '("pooling_convention", AttrOpt (EnumType '["full", "valid"])),
        '("stride", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("data", AttrOpt SymbolHandle)]

_Pooling_v1 ::
            forall args . Fullfilled "_Pooling_v1(symbol)" args =>
              String -> ArgsHMap "_Pooling_v1(symbol)" args -> IO SymbolHandle
_Pooling_v1 name args
  = let scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> (args !? #kernel :: Maybe [Int]),
               ("pool_type",) . showValue <$>
                 (args !? #pool_type :: Maybe (EnumType '["avg", "max", "sum"])),
               ("global_pool",) . showValue <$>
                 (args !? #global_pool :: Maybe Bool),
               ("pooling_convention",) . showValue <$>
                 (args !? #pooling_convention ::
                    Maybe (EnumType '["full", "valid"])),
               ("stride",) . showValue <$> (args !? #stride :: Maybe [Int]),
               ("pad",) . showValue <$> (args !? #pad :: Maybe [Int])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Pooling_v1"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_RNN(symbol)" =
     '[ '("state_size", AttrReq Int), '("num_layers", AttrReq Int),
        '("bidirectional", AttrOpt Bool),
        '("mode",
          AttrReq (EnumType '["gru", "lstm", "rnn_relu", "rnn_tanh"])),
        '("p", AttrOpt Float), '("state_outputs", AttrOpt Bool),
        '("data", AttrOpt SymbolHandle),
        '("parameters", AttrOpt SymbolHandle),
        '("state", AttrOpt SymbolHandle),
        '("state_cell", AttrOpt SymbolHandle)]

_RNN ::
     forall args . Fullfilled "_RNN(symbol)" args =>
       String -> ArgsHMap "_RNN(symbol)" args -> IO SymbolHandle
_RNN name args
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
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("parameters",) <$> (args !? #parameters :: Maybe SymbolHandle),
               ("state",) <$> (args !? #state :: Maybe SymbolHandle),
               ("state_cell",) <$> (args !? #state_cell :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "RNN"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_ROIPooling(symbol)" =
     '[ '("pooled_size", AttrReq [Int]),
        '("spatial_scale", AttrReq Float), '("data", AttrOpt SymbolHandle),
        '("rois", AttrOpt SymbolHandle)]

_ROIPooling ::
            forall args . Fullfilled "_ROIPooling(symbol)" args =>
              String -> ArgsHMap "_ROIPooling(symbol)" args -> IO SymbolHandle
_ROIPooling name args
  = let scalarArgs
          = catMaybes
              [("pooled_size",) . showValue <$>
                 (args !? #pooled_size :: Maybe [Int]),
               ("spatial_scale",) . showValue <$>
                 (args !? #spatial_scale :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("rois",) <$> (args !? #rois :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "ROIPooling"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_Reshape(symbol)" =
     '[ '("shape", AttrOpt [Int]), '("reverse", AttrOpt Bool),
        '("target_shape", AttrOpt [Int]), '("keep_highest", AttrOpt Bool),
        '("data", AttrOpt SymbolHandle)]

_Reshape ::
         forall args . Fullfilled "_Reshape(symbol)" args =>
           String -> ArgsHMap "_Reshape(symbol)" args -> IO SymbolHandle
_Reshape name args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("reverse",) . showValue <$> (args !? #reverse :: Maybe Bool),
               ("target_shape",) . showValue <$>
                 (args !? #target_shape :: Maybe [Int]),
               ("keep_highest",) . showValue <$>
                 (args !? #keep_highest :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Reshape"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_SVMOutput(symbol)" =
     '[ '("margin", AttrOpt Float),
        '("regularization_coefficient", AttrOpt Float),
        '("use_linear", AttrOpt Bool), '("data", AttrOpt SymbolHandle),
        '("label", AttrOpt SymbolHandle)]

_SVMOutput ::
           forall args . Fullfilled "_SVMOutput(symbol)" args =>
             String -> ArgsHMap "_SVMOutput(symbol)" args -> IO SymbolHandle
_SVMOutput name args
  = let scalarArgs
          = catMaybes
              [("margin",) . showValue <$> (args !? #margin :: Maybe Float),
               ("regularization_coefficient",) . showValue <$>
                 (args !? #regularization_coefficient :: Maybe Float),
               ("use_linear",) . showValue <$>
                 (args !? #use_linear :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("label",) <$> (args !? #label :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SVMOutput"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_SequenceLast(symbol)" =
     '[ '("use_sequence_length", AttrOpt Bool), '("axis", AttrOpt Int),
        '("data", AttrOpt SymbolHandle),
        '("sequence_length", AttrOpt SymbolHandle)]

_SequenceLast ::
              forall args . Fullfilled "_SequenceLast(symbol)" args =>
                String -> ArgsHMap "_SequenceLast(symbol)" args -> IO SymbolHandle
_SequenceLast name args
  = let scalarArgs
          = catMaybes
              [("use_sequence_length",) . showValue <$>
                 (args !? #use_sequence_length :: Maybe Bool),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("sequence_length",) <$>
                 (args !? #sequence_length :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SequenceLast"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_SequenceMask(symbol)" =
     '[ '("use_sequence_length", AttrOpt Bool),
        '("value", AttrOpt Float), '("axis", AttrOpt Int),
        '("data", AttrOpt SymbolHandle),
        '("sequence_length", AttrOpt SymbolHandle)]

_SequenceMask ::
              forall args . Fullfilled "_SequenceMask(symbol)" args =>
                String -> ArgsHMap "_SequenceMask(symbol)" args -> IO SymbolHandle
_SequenceMask name args
  = let scalarArgs
          = catMaybes
              [("use_sequence_length",) . showValue <$>
                 (args !? #use_sequence_length :: Maybe Bool),
               ("value",) . showValue <$> (args !? #value :: Maybe Float),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("sequence_length",) <$>
                 (args !? #sequence_length :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SequenceMask"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_SequenceReverse(symbol)" =
     '[ '("use_sequence_length", AttrOpt Bool), '("axis", AttrOpt Int),
        '("data", AttrOpt SymbolHandle),
        '("sequence_length", AttrOpt SymbolHandle)]

_SequenceReverse ::
                 forall args . Fullfilled "_SequenceReverse(symbol)" args =>
                   String ->
                     ArgsHMap "_SequenceReverse(symbol)" args -> IO SymbolHandle
_SequenceReverse name args
  = let scalarArgs
          = catMaybes
              [("use_sequence_length",) . showValue <$>
                 (args !? #use_sequence_length :: Maybe Bool),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("sequence_length",) <$>
                 (args !? #sequence_length :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SequenceReverse"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_SliceChannel(symbol)" =
     '[ '("num_outputs", AttrReq Int), '("axis", AttrOpt Int),
        '("squeeze_axis", AttrOpt Bool), '("data", AttrOpt SymbolHandle)]

_SliceChannel ::
              forall args . Fullfilled "_SliceChannel(symbol)" args =>
                String -> ArgsHMap "_SliceChannel(symbol)" args -> IO SymbolHandle
_SliceChannel name args
  = let scalarArgs
          = catMaybes
              [("num_outputs",) . showValue <$>
                 (args !? #num_outputs :: Maybe Int),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("squeeze_axis",) . showValue <$>
                 (args !? #squeeze_axis :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SliceChannel"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_Softmax(symbol)" =
     '[ '("grad_scale", AttrOpt Float),
        '("ignore_label", AttrOpt Float), '("multi_output", AttrOpt Bool),
        '("use_ignore", AttrOpt Bool), '("preserve_shape", AttrOpt Bool),
        '("normalization", AttrOpt (EnumType '["batch", "null", "valid"])),
        '("out_grad", AttrOpt Bool), '("smooth_alpha", AttrOpt Float),
        '("data", AttrOpt SymbolHandle)]

_Softmax ::
         forall args . Fullfilled "_Softmax(symbol)" args =>
           String -> ArgsHMap "_Softmax(symbol)" args -> IO SymbolHandle
_Softmax name args
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
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "Softmax"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_SoftmaxActivation(symbol)" =
     '[ '("mode", AttrOpt (EnumType '["channel", "instance"])),
        '("data", AttrOpt SymbolHandle)]

_SoftmaxActivation ::
                   forall args . Fullfilled "_SoftmaxActivation(symbol)" args =>
                     String ->
                       ArgsHMap "_SoftmaxActivation(symbol)" args -> IO SymbolHandle
_SoftmaxActivation name args
  = let scalarArgs
          = catMaybes
              [("mode",) . showValue <$>
                 (args !? #mode :: Maybe (EnumType '["channel", "instance"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SoftmaxActivation"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_SoftmaxOutput(symbol)" =
     '[ '("grad_scale", AttrOpt Float),
        '("ignore_label", AttrOpt Float), '("multi_output", AttrOpt Bool),
        '("use_ignore", AttrOpt Bool), '("preserve_shape", AttrOpt Bool),
        '("normalization", AttrOpt (EnumType '["batch", "null", "valid"])),
        '("out_grad", AttrOpt Bool), '("smooth_alpha", AttrOpt Float),
        '("data", AttrOpt SymbolHandle), '("label", AttrOpt SymbolHandle)]

_SoftmaxOutput ::
               forall args . Fullfilled "_SoftmaxOutput(symbol)" args =>
                 String -> ArgsHMap "_SoftmaxOutput(symbol)" args -> IO SymbolHandle
_SoftmaxOutput name args
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
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("label",) <$> (args !? #label :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SoftmaxOutput"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_SpatialTransformer(symbol)" =
     '[ '("target_shape", AttrOpt [Int]),
        '("transform_type", AttrReq (EnumType '["affine"])),
        '("sampler_type", AttrReq (EnumType '["bilinear"])),
        '("data", AttrOpt SymbolHandle), '("loc", AttrOpt SymbolHandle)]

_SpatialTransformer ::
                    forall args . Fullfilled "_SpatialTransformer(symbol)" args =>
                      String ->
                        ArgsHMap "_SpatialTransformer(symbol)" args -> IO SymbolHandle
_SpatialTransformer name args
  = let scalarArgs
          = catMaybes
              [("target_shape",) . showValue <$>
                 (args !? #target_shape :: Maybe [Int]),
               ("transform_type",) . showValue <$>
                 (args !? #transform_type :: Maybe (EnumType '["affine"])),
               ("sampler_type",) . showValue <$>
                 (args !? #sampler_type :: Maybe (EnumType '["bilinear"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("loc",) <$> (args !? #loc :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SpatialTransformer"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_SwapAxis(symbol)" =
     '[ '("dim1", AttrOpt Int), '("dim2", AttrOpt Int),
        '("data", AttrOpt SymbolHandle)]

_SwapAxis ::
          forall args . Fullfilled "_SwapAxis(symbol)" args =>
            String -> ArgsHMap "_SwapAxis(symbol)" args -> IO SymbolHandle
_SwapAxis name args
  = let scalarArgs
          = catMaybes
              [("dim1",) . showValue <$> (args !? #dim1 :: Maybe Int),
               ("dim2",) . showValue <$> (args !? #dim2 :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "SwapAxis"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_UpSampling(symbol)" =
     '[ '("scale", AttrReq Int), '("num_filter", AttrOpt Int),
        '("sample_type", AttrReq (EnumType '["bilinear", "nearest"])),
        '("multi_input_mode", AttrOpt (EnumType '["concat", "sum"])),
        '("num_args", AttrReq Int), '("workspace", AttrOpt Int),
        '("data", AttrOpt [SymbolHandle])]

_UpSampling ::
            forall args . Fullfilled "_UpSampling(symbol)" args =>
              String -> ArgsHMap "_UpSampling(symbol)" args -> IO SymbolHandle
_UpSampling name args
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
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
        array = fromMaybe [] (args !? #data :: Maybe [SymbolHandle])
      in
      do op <- nnGetOpHandle "UpSampling"
         sym <- if hasKey args #num_args then
                  mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys scalarvals
                  else
                  mxSymbolCreateAtomicSymbol (fromOpHandle op)
                    ("num_args" : scalarkeys)
                    (showValue (length array) : scalarvals)
         mxSymbolCompose sym name Nothing array
         return sym

type instance ParameterList "_CachedOp(symbol)" = '[]

_CachedOp ::
          forall args . Fullfilled "_CachedOp(symbol)" args =>
            String -> ArgsHMap "_CachedOp(symbol)" args -> IO SymbolHandle
_CachedOp name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_CachedOp"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_CrossDeviceCopy(symbol)" = '[]

_CrossDeviceCopy ::
                 forall args . Fullfilled "_CrossDeviceCopy(symbol)" args =>
                   String ->
                     ArgsHMap "_CrossDeviceCopy(symbol)" args -> IO SymbolHandle
_CrossDeviceCopy name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_CrossDeviceCopy"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_CustomFunction(symbol)" = '[]

_CustomFunction ::
                forall args . Fullfilled "_CustomFunction(symbol)" args =>
                  String ->
                    ArgsHMap "_CustomFunction(symbol)" args -> IO SymbolHandle
_CustomFunction name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_CustomFunction"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_NoGradient(symbol)" = '[]

_NoGradient ::
            forall args . Fullfilled "_NoGradient(symbol)" args =>
              String -> ArgsHMap "_NoGradient(symbol)" args -> IO SymbolHandle
_NoGradient name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_NoGradient"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_arange(symbol)" =
     '[ '("start", AttrReq Double), '("stop", AttrOpt (Maybe Double)),
        '("step", AttrOpt Double), '("repeat", AttrOpt Int),
        '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"]))]

_arange ::
        forall args . Fullfilled "_arange(symbol)" args =>
          String -> ArgsHMap "_arange(symbol)" args -> IO SymbolHandle
_arange name args
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
                         '["float16", "float32", "float64", "int32", "int64", "int8",
                           "uint8"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_arange"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_Activation(symbol)" = '[]

_backward_Activation ::
                     forall args . Fullfilled "_backward_Activation(symbol)" args =>
                       String ->
                         ArgsHMap "_backward_Activation(symbol)" args -> IO SymbolHandle
_backward_Activation name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Activation"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_BatchNorm(symbol)" = '[]

_backward_BatchNorm ::
                    forall args . Fullfilled "_backward_BatchNorm(symbol)" args =>
                      String ->
                        ArgsHMap "_backward_BatchNorm(symbol)" args -> IO SymbolHandle
_backward_BatchNorm name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_BatchNorm"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_BatchNorm_v1(symbol)" = '[]

_backward_BatchNorm_v1 ::
                       forall args . Fullfilled "_backward_BatchNorm_v1(symbol)" args =>
                         String ->
                           ArgsHMap "_backward_BatchNorm_v1(symbol)" args -> IO SymbolHandle
_backward_BatchNorm_v1 name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_BatchNorm_v1"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_BilinearSampler(symbol)" =
     '[]

_backward_BilinearSampler ::
                          forall args .
                            Fullfilled "_backward_BilinearSampler(symbol)" args =>
                            String ->
                              ArgsHMap "_backward_BilinearSampler(symbol)" args ->
                                IO SymbolHandle
_backward_BilinearSampler name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_BilinearSampler"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_CachedOp(symbol)" = '[]

_backward_CachedOp ::
                   forall args . Fullfilled "_backward_CachedOp(symbol)" args =>
                     String ->
                       ArgsHMap "_backward_CachedOp(symbol)" args -> IO SymbolHandle
_backward_CachedOp name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_CachedOp"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_Concat(symbol)" = '[]

_backward_Concat ::
                 forall args . Fullfilled "_backward_Concat(symbol)" args =>
                   String ->
                     ArgsHMap "_backward_Concat(symbol)" args -> IO SymbolHandle
_backward_Concat name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Concat"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_Convolution(symbol)" = '[]

_backward_Convolution ::
                      forall args . Fullfilled "_backward_Convolution(symbol)" args =>
                        String ->
                          ArgsHMap "_backward_Convolution(symbol)" args -> IO SymbolHandle
_backward_Convolution name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Convolution"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_Convolution_v1(symbol)" =
     '[]

_backward_Convolution_v1 ::
                         forall args . Fullfilled "_backward_Convolution_v1(symbol)" args =>
                           String ->
                             ArgsHMap "_backward_Convolution_v1(symbol)" args -> IO SymbolHandle
_backward_Convolution_v1 name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Convolution_v1"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_Correlation(symbol)" = '[]

_backward_Correlation ::
                      forall args . Fullfilled "_backward_Correlation(symbol)" args =>
                        String ->
                          ArgsHMap "_backward_Correlation(symbol)" args -> IO SymbolHandle
_backward_Correlation name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Correlation"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_Crop(symbol)" = '[]

_backward_Crop ::
               forall args . Fullfilled "_backward_Crop(symbol)" args =>
                 String -> ArgsHMap "_backward_Crop(symbol)" args -> IO SymbolHandle
_backward_Crop name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Crop"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_Custom(symbol)" = '[]

_backward_Custom ::
                 forall args . Fullfilled "_backward_Custom(symbol)" args =>
                   String ->
                     ArgsHMap "_backward_Custom(symbol)" args -> IO SymbolHandle
_backward_Custom name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Custom"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_CustomFunction(symbol)" =
     '[]

_backward_CustomFunction ::
                         forall args . Fullfilled "_backward_CustomFunction(symbol)" args =>
                           String ->
                             ArgsHMap "_backward_CustomFunction(symbol)" args -> IO SymbolHandle
_backward_CustomFunction name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_CustomFunction"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_Deconvolution(symbol)" = '[]

_backward_Deconvolution ::
                        forall args . Fullfilled "_backward_Deconvolution(symbol)" args =>
                          String ->
                            ArgsHMap "_backward_Deconvolution(symbol)" args -> IO SymbolHandle
_backward_Deconvolution name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Deconvolution"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_Dropout(symbol)" = '[]

_backward_Dropout ::
                  forall args . Fullfilled "_backward_Dropout(symbol)" args =>
                    String ->
                      ArgsHMap "_backward_Dropout(symbol)" args -> IO SymbolHandle
_backward_Dropout name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Dropout"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_Embedding(symbol)" = '[]

_backward_Embedding ::
                    forall args . Fullfilled "_backward_Embedding(symbol)" args =>
                      String ->
                        ArgsHMap "_backward_Embedding(symbol)" args -> IO SymbolHandle
_backward_Embedding name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Embedding"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_FullyConnected(symbol)" =
     '[]

_backward_FullyConnected ::
                         forall args . Fullfilled "_backward_FullyConnected(symbol)" args =>
                           String ->
                             ArgsHMap "_backward_FullyConnected(symbol)" args -> IO SymbolHandle
_backward_FullyConnected name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_FullyConnected"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_GridGenerator(symbol)" = '[]

_backward_GridGenerator ::
                        forall args . Fullfilled "_backward_GridGenerator(symbol)" args =>
                          String ->
                            ArgsHMap "_backward_GridGenerator(symbol)" args -> IO SymbolHandle
_backward_GridGenerator name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_GridGenerator"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance
     ParameterList "_backward_IdentityAttachKLSparseReg(symbol)" = '[]

_backward_IdentityAttachKLSparseReg ::
                                    forall args .
                                      Fullfilled "_backward_IdentityAttachKLSparseReg(symbol)"
                                        args =>
                                      String ->
                                        ArgsHMap "_backward_IdentityAttachKLSparseReg(symbol)" args
                                          -> IO SymbolHandle
_backward_IdentityAttachKLSparseReg name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_IdentityAttachKLSparseReg"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_InstanceNorm(symbol)" = '[]

_backward_InstanceNorm ::
                       forall args . Fullfilled "_backward_InstanceNorm(symbol)" args =>
                         String ->
                           ArgsHMap "_backward_InstanceNorm(symbol)" args -> IO SymbolHandle
_backward_InstanceNorm name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_InstanceNorm"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_L2Normalization(symbol)" =
     '[]

_backward_L2Normalization ::
                          forall args .
                            Fullfilled "_backward_L2Normalization(symbol)" args =>
                            String ->
                              ArgsHMap "_backward_L2Normalization(symbol)" args ->
                                IO SymbolHandle
_backward_L2Normalization name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_L2Normalization"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_LRN(symbol)" = '[]

_backward_LRN ::
              forall args . Fullfilled "_backward_LRN(symbol)" args =>
                String -> ArgsHMap "_backward_LRN(symbol)" args -> IO SymbolHandle
_backward_LRN name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_LRN"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_LayerNorm(symbol)" = '[]

_backward_LayerNorm ::
                    forall args . Fullfilled "_backward_LayerNorm(symbol)" args =>
                      String ->
                        ArgsHMap "_backward_LayerNorm(symbol)" args -> IO SymbolHandle
_backward_LayerNorm name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_LayerNorm"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_LeakyReLU(symbol)" = '[]

_backward_LeakyReLU ::
                    forall args . Fullfilled "_backward_LeakyReLU(symbol)" args =>
                      String ->
                        ArgsHMap "_backward_LeakyReLU(symbol)" args -> IO SymbolHandle
_backward_LeakyReLU name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_LeakyReLU"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_MakeLoss(symbol)" = '[]

_backward_MakeLoss ::
                   forall args . Fullfilled "_backward_MakeLoss(symbol)" args =>
                     String ->
                       ArgsHMap "_backward_MakeLoss(symbol)" args -> IO SymbolHandle
_backward_MakeLoss name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_MakeLoss"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_Pad(symbol)" = '[]

_backward_Pad ::
              forall args . Fullfilled "_backward_Pad(symbol)" args =>
                String -> ArgsHMap "_backward_Pad(symbol)" args -> IO SymbolHandle
_backward_Pad name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Pad"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_Pooling(symbol)" = '[]

_backward_Pooling ::
                  forall args . Fullfilled "_backward_Pooling(symbol)" args =>
                    String ->
                      ArgsHMap "_backward_Pooling(symbol)" args -> IO SymbolHandle
_backward_Pooling name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Pooling"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_Pooling_v1(symbol)" = '[]

_backward_Pooling_v1 ::
                     forall args . Fullfilled "_backward_Pooling_v1(symbol)" args =>
                       String ->
                         ArgsHMap "_backward_Pooling_v1(symbol)" args -> IO SymbolHandle
_backward_Pooling_v1 name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Pooling_v1"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_RNN(symbol)" = '[]

_backward_RNN ::
              forall args . Fullfilled "_backward_RNN(symbol)" args =>
                String -> ArgsHMap "_backward_RNN(symbol)" args -> IO SymbolHandle
_backward_RNN name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_RNN"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_ROIAlign(symbol)" = '[]

_backward_ROIAlign ::
                   forall args . Fullfilled "_backward_ROIAlign(symbol)" args =>
                     String ->
                       ArgsHMap "_backward_ROIAlign(symbol)" args -> IO SymbolHandle
_backward_ROIAlign name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_ROIAlign"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_ROIPooling(symbol)" = '[]

_backward_ROIPooling ::
                     forall args . Fullfilled "_backward_ROIPooling(symbol)" args =>
                       String ->
                         ArgsHMap "_backward_ROIPooling(symbol)" args -> IO SymbolHandle
_backward_ROIPooling name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_ROIPooling"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_SVMOutput(symbol)" = '[]

_backward_SVMOutput ::
                    forall args . Fullfilled "_backward_SVMOutput(symbol)" args =>
                      String ->
                        ArgsHMap "_backward_SVMOutput(symbol)" args -> IO SymbolHandle
_backward_SVMOutput name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SVMOutput"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_SequenceLast(symbol)" = '[]

_backward_SequenceLast ::
                       forall args . Fullfilled "_backward_SequenceLast(symbol)" args =>
                         String ->
                           ArgsHMap "_backward_SequenceLast(symbol)" args -> IO SymbolHandle
_backward_SequenceLast name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SequenceLast"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_SequenceMask(symbol)" = '[]

_backward_SequenceMask ::
                       forall args . Fullfilled "_backward_SequenceMask(symbol)" args =>
                         String ->
                           ArgsHMap "_backward_SequenceMask(symbol)" args -> IO SymbolHandle
_backward_SequenceMask name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SequenceMask"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_SequenceReverse(symbol)" =
     '[]

_backward_SequenceReverse ::
                          forall args .
                            Fullfilled "_backward_SequenceReverse(symbol)" args =>
                            String ->
                              ArgsHMap "_backward_SequenceReverse(symbol)" args ->
                                IO SymbolHandle
_backward_SequenceReverse name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SequenceReverse"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_SliceChannel(symbol)" = '[]

_backward_SliceChannel ::
                       forall args . Fullfilled "_backward_SliceChannel(symbol)" args =>
                         String ->
                           ArgsHMap "_backward_SliceChannel(symbol)" args -> IO SymbolHandle
_backward_SliceChannel name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SliceChannel"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_Softmax(symbol)" = '[]

_backward_Softmax ::
                  forall args . Fullfilled "_backward_Softmax(symbol)" args =>
                    String ->
                      ArgsHMap "_backward_Softmax(symbol)" args -> IO SymbolHandle
_backward_Softmax name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_Softmax"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_SoftmaxActivation(symbol)" =
     '[]

_backward_SoftmaxActivation ::
                            forall args .
                              Fullfilled "_backward_SoftmaxActivation(symbol)" args =>
                              String ->
                                ArgsHMap "_backward_SoftmaxActivation(symbol)" args ->
                                  IO SymbolHandle
_backward_SoftmaxActivation name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SoftmaxActivation"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_SoftmaxOutput(symbol)" = '[]

_backward_SoftmaxOutput ::
                        forall args . Fullfilled "_backward_SoftmaxOutput(symbol)" args =>
                          String ->
                            ArgsHMap "_backward_SoftmaxOutput(symbol)" args -> IO SymbolHandle
_backward_SoftmaxOutput name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SoftmaxOutput"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_SparseEmbedding(symbol)" =
     '[]

_backward_SparseEmbedding ::
                          forall args .
                            Fullfilled "_backward_SparseEmbedding(symbol)" args =>
                            String ->
                              ArgsHMap "_backward_SparseEmbedding(symbol)" args ->
                                IO SymbolHandle
_backward_SparseEmbedding name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SparseEmbedding"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_SpatialTransformer(symbol)"
     = '[]

_backward_SpatialTransformer ::
                             forall args .
                               Fullfilled "_backward_SpatialTransformer(symbol)" args =>
                               String ->
                                 ArgsHMap "_backward_SpatialTransformer(symbol)" args ->
                                   IO SymbolHandle
_backward_SpatialTransformer name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SpatialTransformer"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_SwapAxis(symbol)" = '[]

_backward_SwapAxis ::
                   forall args . Fullfilled "_backward_SwapAxis(symbol)" args =>
                     String ->
                       ArgsHMap "_backward_SwapAxis(symbol)" args -> IO SymbolHandle
_backward_SwapAxis name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_SwapAxis"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_UpSampling(symbol)" = '[]

_backward_UpSampling ::
                     forall args . Fullfilled "_backward_UpSampling(symbol)" args =>
                       String ->
                         ArgsHMap "_backward_UpSampling(symbol)" args -> IO SymbolHandle
_backward_UpSampling name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_UpSampling"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward__CrossDeviceCopy(symbol)" =
     '[]

_backward__CrossDeviceCopy ::
                           forall args .
                             Fullfilled "_backward__CrossDeviceCopy(symbol)" args =>
                             String ->
                               ArgsHMap "_backward__CrossDeviceCopy(symbol)" args ->
                                 IO SymbolHandle
_backward__CrossDeviceCopy name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__CrossDeviceCopy"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward__NDArray(symbol)" = '[]

_backward__NDArray ::
                   forall args . Fullfilled "_backward__NDArray(symbol)" args =>
                     String ->
                       ArgsHMap "_backward__NDArray(symbol)" args -> IO SymbolHandle
_backward__NDArray name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__NDArray"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward__Native(symbol)" = '[]

_backward__Native ::
                  forall args . Fullfilled "_backward__Native(symbol)" args =>
                    String ->
                      ArgsHMap "_backward__Native(symbol)" args -> IO SymbolHandle
_backward__Native name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__Native"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward__contrib_CTCLoss(symbol)" =
     '[]

_backward__contrib_CTCLoss ::
                           forall args .
                             Fullfilled "_backward__contrib_CTCLoss(symbol)" args =>
                             String ->
                               ArgsHMap "_backward__contrib_CTCLoss(symbol)" args ->
                                 IO SymbolHandle
_backward__contrib_CTCLoss name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_CTCLoss"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance
     ParameterList "_backward__contrib_DeformableConvolution(symbol)" =
     '[]

_backward__contrib_DeformableConvolution ::
                                         forall args .
                                           Fullfilled
                                             "_backward__contrib_DeformableConvolution(symbol)"
                                             args =>
                                           String ->
                                             ArgsHMap
                                               "_backward__contrib_DeformableConvolution(symbol)"
                                               args
                                               -> IO SymbolHandle
_backward__contrib_DeformableConvolution name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_DeformableConvolution"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance
     ParameterList "_backward__contrib_DeformablePSROIPooling(symbol)" =
     '[]

_backward__contrib_DeformablePSROIPooling ::
                                          forall args .
                                            Fullfilled
                                              "_backward__contrib_DeformablePSROIPooling(symbol)"
                                              args =>
                                            String ->
                                              ArgsHMap
                                                "_backward__contrib_DeformablePSROIPooling(symbol)"
                                                args
                                                -> IO SymbolHandle
_backward__contrib_DeformablePSROIPooling name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_DeformablePSROIPooling"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance
     ParameterList "_backward__contrib_MultiBoxDetection(symbol)" = '[]

_backward__contrib_MultiBoxDetection ::
                                     forall args .
                                       Fullfilled "_backward__contrib_MultiBoxDetection(symbol)"
                                         args =>
                                       String ->
                                         ArgsHMap "_backward__contrib_MultiBoxDetection(symbol)"
                                           args
                                           -> IO SymbolHandle
_backward__contrib_MultiBoxDetection name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_MultiBoxDetection"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance
     ParameterList "_backward__contrib_MultiBoxPrior(symbol)" = '[]

_backward__contrib_MultiBoxPrior ::
                                 forall args .
                                   Fullfilled "_backward__contrib_MultiBoxPrior(symbol)" args =>
                                   String ->
                                     ArgsHMap "_backward__contrib_MultiBoxPrior(symbol)" args ->
                                       IO SymbolHandle
_backward__contrib_MultiBoxPrior name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_MultiBoxPrior"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance
     ParameterList "_backward__contrib_MultiBoxTarget(symbol)" = '[]

_backward__contrib_MultiBoxTarget ::
                                  forall args .
                                    Fullfilled "_backward__contrib_MultiBoxTarget(symbol)" args =>
                                    String ->
                                      ArgsHMap "_backward__contrib_MultiBoxTarget(symbol)" args ->
                                        IO SymbolHandle
_backward__contrib_MultiBoxTarget name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_MultiBoxTarget"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance
     ParameterList "_backward__contrib_MultiProposal(symbol)" = '[]

_backward__contrib_MultiProposal ::
                                 forall args .
                                   Fullfilled "_backward__contrib_MultiProposal(symbol)" args =>
                                   String ->
                                     ArgsHMap "_backward__contrib_MultiProposal(symbol)" args ->
                                       IO SymbolHandle
_backward__contrib_MultiProposal name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_MultiProposal"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance
     ParameterList "_backward__contrib_PSROIPooling(symbol)" = '[]

_backward__contrib_PSROIPooling ::
                                forall args .
                                  Fullfilled "_backward__contrib_PSROIPooling(symbol)" args =>
                                  String ->
                                    ArgsHMap "_backward__contrib_PSROIPooling(symbol)" args ->
                                      IO SymbolHandle
_backward__contrib_PSROIPooling name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_PSROIPooling"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward__contrib_Proposal(symbol)" =
     '[]

_backward__contrib_Proposal ::
                            forall args .
                              Fullfilled "_backward__contrib_Proposal(symbol)" args =>
                              String ->
                                ArgsHMap "_backward__contrib_Proposal(symbol)" args ->
                                  IO SymbolHandle
_backward__contrib_Proposal name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_Proposal"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance
     ParameterList "_backward__contrib_SyncBatchNorm(symbol)" = '[]

_backward__contrib_SyncBatchNorm ::
                                 forall args .
                                   Fullfilled "_backward__contrib_SyncBatchNorm(symbol)" args =>
                                   String ->
                                     ArgsHMap "_backward__contrib_SyncBatchNorm(symbol)" args ->
                                       IO SymbolHandle
_backward__contrib_SyncBatchNorm name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_SyncBatchNorm"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance
     ParameterList "_backward__contrib_count_sketch(symbol)" = '[]

_backward__contrib_count_sketch ::
                                forall args .
                                  Fullfilled "_backward__contrib_count_sketch(symbol)" args =>
                                  String ->
                                    ArgsHMap "_backward__contrib_count_sketch(symbol)" args ->
                                      IO SymbolHandle
_backward__contrib_count_sketch name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_count_sketch"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward__contrib_fft(symbol)" = '[]

_backward__contrib_fft ::
                       forall args . Fullfilled "_backward__contrib_fft(symbol)" args =>
                         String ->
                           ArgsHMap "_backward__contrib_fft(symbol)" args -> IO SymbolHandle
_backward__contrib_fft name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_fft"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward__contrib_ifft(symbol)" = '[]

_backward__contrib_ifft ::
                        forall args . Fullfilled "_backward__contrib_ifft(symbol)" args =>
                          String ->
                            ArgsHMap "_backward__contrib_ifft(symbol)" args -> IO SymbolHandle
_backward__contrib_ifft name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward__contrib_ifft"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_abs(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_abs ::
              forall args . Fullfilled "_backward_abs(symbol)" args =>
                String -> ArgsHMap "_backward_abs(symbol)" args -> IO SymbolHandle
_backward_abs name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_abs"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_add(symbol)" = '[]

_backward_add ::
              forall args . Fullfilled "_backward_add(symbol)" args =>
                String -> ArgsHMap "_backward_add(symbol)" args -> IO SymbolHandle
_backward_add name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_add"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_arccos(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_arccos ::
                 forall args . Fullfilled "_backward_arccos(symbol)" args =>
                   String ->
                     ArgsHMap "_backward_arccos(symbol)" args -> IO SymbolHandle
_backward_arccos name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_arccos"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_arccosh(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_arccosh ::
                  forall args . Fullfilled "_backward_arccosh(symbol)" args =>
                    String ->
                      ArgsHMap "_backward_arccosh(symbol)" args -> IO SymbolHandle
_backward_arccosh name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_arccosh"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_arcsin(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_arcsin ::
                 forall args . Fullfilled "_backward_arcsin(symbol)" args =>
                   String ->
                     ArgsHMap "_backward_arcsin(symbol)" args -> IO SymbolHandle
_backward_arcsin name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_arcsin"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_arcsinh(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_arcsinh ::
                  forall args . Fullfilled "_backward_arcsinh(symbol)" args =>
                    String ->
                      ArgsHMap "_backward_arcsinh(symbol)" args -> IO SymbolHandle
_backward_arcsinh name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_arcsinh"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_arctan(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_arctan ::
                 forall args . Fullfilled "_backward_arctan(symbol)" args =>
                   String ->
                     ArgsHMap "_backward_arctan(symbol)" args -> IO SymbolHandle
_backward_arctan name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_arctan"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_arctanh(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_arctanh ::
                  forall args . Fullfilled "_backward_arctanh(symbol)" args =>
                    String ->
                      ArgsHMap "_backward_arctanh(symbol)" args -> IO SymbolHandle
_backward_arctanh name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_arctanh"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_batch_dot(symbol)" = '[]

_backward_batch_dot ::
                    forall args . Fullfilled "_backward_batch_dot(symbol)" args =>
                      String ->
                        ArgsHMap "_backward_batch_dot(symbol)" args -> IO SymbolHandle
_backward_batch_dot name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_batch_dot"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_broadcast_add(symbol)" = '[]

_backward_broadcast_add ::
                        forall args . Fullfilled "_backward_broadcast_add(symbol)" args =>
                          String ->
                            ArgsHMap "_backward_broadcast_add(symbol)" args -> IO SymbolHandle
_backward_broadcast_add name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_add"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_broadcast_div(symbol)" = '[]

_backward_broadcast_div ::
                        forall args . Fullfilled "_backward_broadcast_div(symbol)" args =>
                          String ->
                            ArgsHMap "_backward_broadcast_div(symbol)" args -> IO SymbolHandle
_backward_broadcast_div name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_div"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_broadcast_hypot(symbol)" =
     '[]

_backward_broadcast_hypot ::
                          forall args .
                            Fullfilled "_backward_broadcast_hypot(symbol)" args =>
                            String ->
                              ArgsHMap "_backward_broadcast_hypot(symbol)" args ->
                                IO SymbolHandle
_backward_broadcast_hypot name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_hypot"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_broadcast_maximum(symbol)" =
     '[]

_backward_broadcast_maximum ::
                            forall args .
                              Fullfilled "_backward_broadcast_maximum(symbol)" args =>
                              String ->
                                ArgsHMap "_backward_broadcast_maximum(symbol)" args ->
                                  IO SymbolHandle
_backward_broadcast_maximum name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_maximum"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_broadcast_minimum(symbol)" =
     '[]

_backward_broadcast_minimum ::
                            forall args .
                              Fullfilled "_backward_broadcast_minimum(symbol)" args =>
                              String ->
                                ArgsHMap "_backward_broadcast_minimum(symbol)" args ->
                                  IO SymbolHandle
_backward_broadcast_minimum name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_minimum"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_broadcast_mod(symbol)" = '[]

_backward_broadcast_mod ::
                        forall args . Fullfilled "_backward_broadcast_mod(symbol)" args =>
                          String ->
                            ArgsHMap "_backward_broadcast_mod(symbol)" args -> IO SymbolHandle
_backward_broadcast_mod name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_mod"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_broadcast_mul(symbol)" = '[]

_backward_broadcast_mul ::
                        forall args . Fullfilled "_backward_broadcast_mul(symbol)" args =>
                          String ->
                            ArgsHMap "_backward_broadcast_mul(symbol)" args -> IO SymbolHandle
_backward_broadcast_mul name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_mul"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_broadcast_power(symbol)" =
     '[]

_backward_broadcast_power ::
                          forall args .
                            Fullfilled "_backward_broadcast_power(symbol)" args =>
                            String ->
                              ArgsHMap "_backward_broadcast_power(symbol)" args ->
                                IO SymbolHandle
_backward_broadcast_power name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_power"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_broadcast_sub(symbol)" = '[]

_backward_broadcast_sub ::
                        forall args . Fullfilled "_backward_broadcast_sub(symbol)" args =>
                          String ->
                            ArgsHMap "_backward_broadcast_sub(symbol)" args -> IO SymbolHandle
_backward_broadcast_sub name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_broadcast_sub"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_cast(symbol)" = '[]

_backward_cast ::
               forall args . Fullfilled "_backward_cast(symbol)" args =>
                 String -> ArgsHMap "_backward_cast(symbol)" args -> IO SymbolHandle
_backward_cast name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_cast"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_cbrt(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_cbrt ::
               forall args . Fullfilled "_backward_cbrt(symbol)" args =>
                 String -> ArgsHMap "_backward_cbrt(symbol)" args -> IO SymbolHandle
_backward_cbrt name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_cbrt"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_clip(symbol)" = '[]

_backward_clip ::
               forall args . Fullfilled "_backward_clip(symbol)" args =>
                 String -> ArgsHMap "_backward_clip(symbol)" args -> IO SymbolHandle
_backward_clip name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_clip"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_cond(symbol)" = '[]

_backward_cond ::
               forall args . Fullfilled "_backward_cond(symbol)" args =>
                 String -> ArgsHMap "_backward_cond(symbol)" args -> IO SymbolHandle
_backward_cond name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_cond"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance
     ParameterList "_backward_contrib_AdaptiveAvgPooling2D(symbol)" =
     '[]

_backward_contrib_AdaptiveAvgPooling2D ::
                                       forall args .
                                         Fullfilled "_backward_contrib_AdaptiveAvgPooling2D(symbol)"
                                           args =>
                                         String ->
                                           ArgsHMap "_backward_contrib_AdaptiveAvgPooling2D(symbol)"
                                             args
                                             -> IO SymbolHandle
_backward_contrib_AdaptiveAvgPooling2D name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_contrib_AdaptiveAvgPooling2D"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance
     ParameterList "_backward_contrib_BilinearResize2D(symbol)" = '[]

_backward_contrib_BilinearResize2D ::
                                   forall args .
                                     Fullfilled "_backward_contrib_BilinearResize2D(symbol)" args =>
                                     String ->
                                       ArgsHMap "_backward_contrib_BilinearResize2D(symbol)" args ->
                                         IO SymbolHandle
_backward_contrib_BilinearResize2D name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_contrib_BilinearResize2D"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance
     ParameterList "_backward_contrib_bipartite_matching(symbol)" =
     '[ '("is_ascend", AttrOpt Bool), '("threshold", AttrReq Float),
        '("topk", AttrOpt Int)]

_backward_contrib_bipartite_matching ::
                                     forall args .
                                       Fullfilled "_backward_contrib_bipartite_matching(symbol)"
                                         args =>
                                       String ->
                                         ArgsHMap "_backward_contrib_bipartite_matching(symbol)"
                                           args
                                           -> IO SymbolHandle
_backward_contrib_bipartite_matching name args
  = let scalarArgs
          = catMaybes
              [("is_ascend",) . showValue <$> (args !? #is_ascend :: Maybe Bool),
               ("threshold",) . showValue <$> (args !? #threshold :: Maybe Float),
               ("topk",) . showValue <$> (args !? #topk :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_contrib_bipartite_matching"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_contrib_box_iou(symbol)" =
     '[ '("format", AttrOpt (EnumType '["center", "corner"]))]

_backward_contrib_box_iou ::
                          forall args .
                            Fullfilled "_backward_contrib_box_iou(symbol)" args =>
                            String ->
                              ArgsHMap "_backward_contrib_box_iou(symbol)" args ->
                                IO SymbolHandle
_backward_contrib_box_iou name args
  = let scalarArgs
          = catMaybes
              [("format",) . showValue <$>
                 (args !? #format :: Maybe (EnumType '["center", "corner"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_contrib_box_iou"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_contrib_box_nms(symbol)" =
     '[ '("overlap_thresh", AttrOpt Float),
        '("valid_thresh", AttrOpt Float), '("topk", AttrOpt Int),
        '("coord_start", AttrOpt Int), '("score_index", AttrOpt Int),
        '("id_index", AttrOpt Int), '("force_suppress", AttrOpt Bool),
        '("in_format", AttrOpt (EnumType '["center", "corner"])),
        '("out_format", AttrOpt (EnumType '["center", "corner"]))]

_backward_contrib_box_nms ::
                          forall args .
                            Fullfilled "_backward_contrib_box_nms(symbol)" args =>
                            String ->
                              ArgsHMap "_backward_contrib_box_nms(symbol)" args ->
                                IO SymbolHandle
_backward_contrib_box_nms name args
  = let scalarArgs
          = catMaybes
              [("overlap_thresh",) . showValue <$>
                 (args !? #overlap_thresh :: Maybe Float),
               ("valid_thresh",) . showValue <$>
                 (args !? #valid_thresh :: Maybe Float),
               ("topk",) . showValue <$> (args !? #topk :: Maybe Int),
               ("coord_start",) . showValue <$>
                 (args !? #coord_start :: Maybe Int),
               ("score_index",) . showValue <$>
                 (args !? #score_index :: Maybe Int),
               ("id_index",) . showValue <$> (args !? #id_index :: Maybe Int),
               ("force_suppress",) . showValue <$>
                 (args !? #force_suppress :: Maybe Bool),
               ("in_format",) . showValue <$>
                 (args !? #in_format :: Maybe (EnumType '["center", "corner"])),
               ("out_format",) . showValue <$>
                 (args !? #out_format :: Maybe (EnumType '["center", "corner"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_contrib_box_nms"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_copy(symbol)" = '[]

_backward_copy ::
               forall args . Fullfilled "_backward_copy(symbol)" args =>
                 String -> ArgsHMap "_backward_copy(symbol)" args -> IO SymbolHandle
_backward_copy name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_copy"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_cos(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_cos ::
              forall args . Fullfilled "_backward_cos(symbol)" args =>
                String -> ArgsHMap "_backward_cos(symbol)" args -> IO SymbolHandle
_backward_cos name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_cos"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_cosh(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_cosh ::
               forall args . Fullfilled "_backward_cosh(symbol)" args =>
                 String -> ArgsHMap "_backward_cosh(symbol)" args -> IO SymbolHandle
_backward_cosh name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_cosh"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_degrees(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_degrees ::
                  forall args . Fullfilled "_backward_degrees(symbol)" args =>
                    String ->
                      ArgsHMap "_backward_degrees(symbol)" args -> IO SymbolHandle
_backward_degrees name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_degrees"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_diag(symbol)" = '[]

_backward_diag ::
               forall args . Fullfilled "_backward_diag(symbol)" args =>
                 String -> ArgsHMap "_backward_diag(symbol)" args -> IO SymbolHandle
_backward_diag name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_diag"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_div(symbol)" = '[]

_backward_div ::
              forall args . Fullfilled "_backward_div(symbol)" args =>
                String -> ArgsHMap "_backward_div(symbol)" args -> IO SymbolHandle
_backward_div name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_div"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_div_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_backward_div_scalar ::
                     forall args . Fullfilled "_backward_div_scalar(symbol)" args =>
                       String ->
                         ArgsHMap "_backward_div_scalar(symbol)" args -> IO SymbolHandle
_backward_div_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_div_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_dot(symbol)" =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("forward_stype",
          AttrOpt (Maybe (EnumType '["csr", "default", "row_sparse"])))]

_backward_dot ::
              forall args . Fullfilled "_backward_dot(symbol)" args =>
                String -> ArgsHMap "_backward_dot(symbol)" args -> IO SymbolHandle
_backward_dot name args
  = let scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$>
                 (args !? #transpose_a :: Maybe Bool),
               ("transpose_b",) . showValue <$>
                 (args !? #transpose_b :: Maybe Bool),
               ("forward_stype",) . showValue <$>
                 (args !? #forward_stype ::
                    Maybe (Maybe (EnumType '["csr", "default", "row_sparse"])))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_dot"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_expm1(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_expm1 ::
                forall args . Fullfilled "_backward_expm1(symbol)" args =>
                  String ->
                    ArgsHMap "_backward_expm1(symbol)" args -> IO SymbolHandle
_backward_expm1 name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_expm1"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_foreach(symbol)" = '[]

_backward_foreach ::
                  forall args . Fullfilled "_backward_foreach(symbol)" args =>
                    String ->
                      ArgsHMap "_backward_foreach(symbol)" args -> IO SymbolHandle
_backward_foreach name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_foreach"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_gamma(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_gamma ::
                forall args . Fullfilled "_backward_gamma(symbol)" args =>
                  String ->
                    ArgsHMap "_backward_gamma(symbol)" args -> IO SymbolHandle
_backward_gamma name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_gamma"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_gammaln(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_gammaln ::
                  forall args . Fullfilled "_backward_gammaln(symbol)" args =>
                    String ->
                      ArgsHMap "_backward_gammaln(symbol)" args -> IO SymbolHandle
_backward_gammaln name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_gammaln"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_gather_nd(symbol)" =
     '[ '("shape", AttrReq [Int]), '("data", AttrOpt SymbolHandle),
        '("indices", AttrOpt SymbolHandle)]

_backward_gather_nd ::
                    forall args . Fullfilled "_backward_gather_nd(symbol)" args =>
                      String ->
                        ArgsHMap "_backward_gather_nd(symbol)" args -> IO SymbolHandle
_backward_gather_nd name args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("indices",) <$> (args !? #indices :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_gather_nd"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_hard_sigmoid(symbol)" = '[]

_backward_hard_sigmoid ::
                       forall args . Fullfilled "_backward_hard_sigmoid(symbol)" args =>
                         String ->
                           ArgsHMap "_backward_hard_sigmoid(symbol)" args -> IO SymbolHandle
_backward_hard_sigmoid name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_hard_sigmoid"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_hypot(symbol)" = '[]

_backward_hypot ::
                forall args . Fullfilled "_backward_hypot(symbol)" args =>
                  String ->
                    ArgsHMap "_backward_hypot(symbol)" args -> IO SymbolHandle
_backward_hypot name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_hypot"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_hypot_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt SymbolHandle),
        '("rhs", AttrOpt SymbolHandle)]

_backward_hypot_scalar ::
                       forall args . Fullfilled "_backward_hypot_scalar(symbol)" args =>
                         String ->
                           ArgsHMap "_backward_hypot_scalar(symbol)" args -> IO SymbolHandle
_backward_hypot_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_hypot_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_linalg_gelqf(symbol)" = '[]

_backward_linalg_gelqf ::
                       forall args . Fullfilled "_backward_linalg_gelqf(symbol)" args =>
                         String ->
                           ArgsHMap "_backward_linalg_gelqf(symbol)" args -> IO SymbolHandle
_backward_linalg_gelqf name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_gelqf"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_linalg_gemm(symbol)" = '[]

_backward_linalg_gemm ::
                      forall args . Fullfilled "_backward_linalg_gemm(symbol)" args =>
                        String ->
                          ArgsHMap "_backward_linalg_gemm(symbol)" args -> IO SymbolHandle
_backward_linalg_gemm name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_gemm"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_linalg_gemm2(symbol)" = '[]

_backward_linalg_gemm2 ::
                       forall args . Fullfilled "_backward_linalg_gemm2(symbol)" args =>
                         String ->
                           ArgsHMap "_backward_linalg_gemm2(symbol)" args -> IO SymbolHandle
_backward_linalg_gemm2 name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_gemm2"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_linalg_potrf(symbol)" = '[]

_backward_linalg_potrf ::
                       forall args . Fullfilled "_backward_linalg_potrf(symbol)" args =>
                         String ->
                           ArgsHMap "_backward_linalg_potrf(symbol)" args -> IO SymbolHandle
_backward_linalg_potrf name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_potrf"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_linalg_potri(symbol)" = '[]

_backward_linalg_potri ::
                       forall args . Fullfilled "_backward_linalg_potri(symbol)" args =>
                         String ->
                           ArgsHMap "_backward_linalg_potri(symbol)" args -> IO SymbolHandle
_backward_linalg_potri name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_potri"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_linalg_sumlogdiag(symbol)" =
     '[]

_backward_linalg_sumlogdiag ::
                            forall args .
                              Fullfilled "_backward_linalg_sumlogdiag(symbol)" args =>
                              String ->
                                ArgsHMap "_backward_linalg_sumlogdiag(symbol)" args ->
                                  IO SymbolHandle
_backward_linalg_sumlogdiag name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_sumlogdiag"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_linalg_syevd(symbol)" = '[]

_backward_linalg_syevd ::
                       forall args . Fullfilled "_backward_linalg_syevd(symbol)" args =>
                         String ->
                           ArgsHMap "_backward_linalg_syevd(symbol)" args -> IO SymbolHandle
_backward_linalg_syevd name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_syevd"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_linalg_syrk(symbol)" = '[]

_backward_linalg_syrk ::
                      forall args . Fullfilled "_backward_linalg_syrk(symbol)" args =>
                        String ->
                          ArgsHMap "_backward_linalg_syrk(symbol)" args -> IO SymbolHandle
_backward_linalg_syrk name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_syrk"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_linalg_trmm(symbol)" = '[]

_backward_linalg_trmm ::
                      forall args . Fullfilled "_backward_linalg_trmm(symbol)" args =>
                        String ->
                          ArgsHMap "_backward_linalg_trmm(symbol)" args -> IO SymbolHandle
_backward_linalg_trmm name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_trmm"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_linalg_trsm(symbol)" = '[]

_backward_linalg_trsm ::
                      forall args . Fullfilled "_backward_linalg_trsm(symbol)" args =>
                        String ->
                          ArgsHMap "_backward_linalg_trsm(symbol)" args -> IO SymbolHandle
_backward_linalg_trsm name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linalg_trsm"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_linear_reg_out(symbol)" =
     '[]

_backward_linear_reg_out ::
                         forall args . Fullfilled "_backward_linear_reg_out(symbol)" args =>
                           String ->
                             ArgsHMap "_backward_linear_reg_out(symbol)" args -> IO SymbolHandle
_backward_linear_reg_out name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_linear_reg_out"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_log(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_log ::
              forall args . Fullfilled "_backward_log(symbol)" args =>
                String -> ArgsHMap "_backward_log(symbol)" args -> IO SymbolHandle
_backward_log name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_log"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_log10(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_log10 ::
                forall args . Fullfilled "_backward_log10(symbol)" args =>
                  String ->
                    ArgsHMap "_backward_log10(symbol)" args -> IO SymbolHandle
_backward_log10 name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_log10"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_log1p(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_log1p ::
                forall args . Fullfilled "_backward_log1p(symbol)" args =>
                  String ->
                    ArgsHMap "_backward_log1p(symbol)" args -> IO SymbolHandle
_backward_log1p name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_log1p"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_log2(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_log2 ::
               forall args . Fullfilled "_backward_log2(symbol)" args =>
                 String -> ArgsHMap "_backward_log2(symbol)" args -> IO SymbolHandle
_backward_log2 name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_log2"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_log_softmax(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_log_softmax ::
                      forall args . Fullfilled "_backward_log_softmax(symbol)" args =>
                        String ->
                          ArgsHMap "_backward_log_softmax(symbol)" args -> IO SymbolHandle
_backward_log_softmax name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_log_softmax"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_logistic_reg_out(symbol)" =
     '[]

_backward_logistic_reg_out ::
                           forall args .
                             Fullfilled "_backward_logistic_reg_out(symbol)" args =>
                             String ->
                               ArgsHMap "_backward_logistic_reg_out(symbol)" args ->
                                 IO SymbolHandle
_backward_logistic_reg_out name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_logistic_reg_out"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_mae_reg_out(symbol)" = '[]

_backward_mae_reg_out ::
                      forall args . Fullfilled "_backward_mae_reg_out(symbol)" args =>
                        String ->
                          ArgsHMap "_backward_mae_reg_out(symbol)" args -> IO SymbolHandle
_backward_mae_reg_out name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_mae_reg_out"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_max(symbol)" = '[]

_backward_max ::
              forall args . Fullfilled "_backward_max(symbol)" args =>
                String -> ArgsHMap "_backward_max(symbol)" args -> IO SymbolHandle
_backward_max name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_max"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_maximum(symbol)" = '[]

_backward_maximum ::
                  forall args . Fullfilled "_backward_maximum(symbol)" args =>
                    String ->
                      ArgsHMap "_backward_maximum(symbol)" args -> IO SymbolHandle
_backward_maximum name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_maximum"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_maximum_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt SymbolHandle),
        '("rhs", AttrOpt SymbolHandle)]

_backward_maximum_scalar ::
                         forall args . Fullfilled "_backward_maximum_scalar(symbol)" args =>
                           String ->
                             ArgsHMap "_backward_maximum_scalar(symbol)" args -> IO SymbolHandle
_backward_maximum_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_maximum_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_mean(symbol)" = '[]

_backward_mean ::
               forall args . Fullfilled "_backward_mean(symbol)" args =>
                 String -> ArgsHMap "_backward_mean(symbol)" args -> IO SymbolHandle
_backward_mean name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_mean"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_min(symbol)" = '[]

_backward_min ::
              forall args . Fullfilled "_backward_min(symbol)" args =>
                String -> ArgsHMap "_backward_min(symbol)" args -> IO SymbolHandle
_backward_min name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_min"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_minimum(symbol)" = '[]

_backward_minimum ::
                  forall args . Fullfilled "_backward_minimum(symbol)" args =>
                    String ->
                      ArgsHMap "_backward_minimum(symbol)" args -> IO SymbolHandle
_backward_minimum name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_minimum"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_minimum_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt SymbolHandle),
        '("rhs", AttrOpt SymbolHandle)]

_backward_minimum_scalar ::
                         forall args . Fullfilled "_backward_minimum_scalar(symbol)" args =>
                           String ->
                             ArgsHMap "_backward_minimum_scalar(symbol)" args -> IO SymbolHandle
_backward_minimum_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_minimum_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_mod(symbol)" = '[]

_backward_mod ::
              forall args . Fullfilled "_backward_mod(symbol)" args =>
                String -> ArgsHMap "_backward_mod(symbol)" args -> IO SymbolHandle
_backward_mod name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_mod"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_mod_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt SymbolHandle),
        '("rhs", AttrOpt SymbolHandle)]

_backward_mod_scalar ::
                     forall args . Fullfilled "_backward_mod_scalar(symbol)" args =>
                       String ->
                         ArgsHMap "_backward_mod_scalar(symbol)" args -> IO SymbolHandle
_backward_mod_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_mod_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_mul(symbol)" = '[]

_backward_mul ::
              forall args . Fullfilled "_backward_mul(symbol)" args =>
                String -> ArgsHMap "_backward_mul(symbol)" args -> IO SymbolHandle
_backward_mul name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_mul"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_mul_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_backward_mul_scalar ::
                     forall args . Fullfilled "_backward_mul_scalar(symbol)" args =>
                       String ->
                         ArgsHMap "_backward_mul_scalar(symbol)" args -> IO SymbolHandle
_backward_mul_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_mul_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_nanprod(symbol)" = '[]

_backward_nanprod ::
                  forall args . Fullfilled "_backward_nanprod(symbol)" args =>
                    String ->
                      ArgsHMap "_backward_nanprod(symbol)" args -> IO SymbolHandle
_backward_nanprod name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_nanprod"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_nansum(symbol)" = '[]

_backward_nansum ::
                 forall args . Fullfilled "_backward_nansum(symbol)" args =>
                   String ->
                     ArgsHMap "_backward_nansum(symbol)" args -> IO SymbolHandle
_backward_nansum name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_nansum"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_norm(symbol)" = '[]

_backward_norm ::
               forall args . Fullfilled "_backward_norm(symbol)" args =>
                 String -> ArgsHMap "_backward_norm(symbol)" args -> IO SymbolHandle
_backward_norm name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_norm"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_pick(symbol)" = '[]

_backward_pick ::
               forall args . Fullfilled "_backward_pick(symbol)" args =>
                 String -> ArgsHMap "_backward_pick(symbol)" args -> IO SymbolHandle
_backward_pick name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_pick"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_power(symbol)" = '[]

_backward_power ::
                forall args . Fullfilled "_backward_power(symbol)" args =>
                  String ->
                    ArgsHMap "_backward_power(symbol)" args -> IO SymbolHandle
_backward_power name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_power"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_power_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt SymbolHandle),
        '("rhs", AttrOpt SymbolHandle)]

_backward_power_scalar ::
                       forall args . Fullfilled "_backward_power_scalar(symbol)" args =>
                         String ->
                           ArgsHMap "_backward_power_scalar(symbol)" args -> IO SymbolHandle
_backward_power_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_power_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_prod(symbol)" = '[]

_backward_prod ::
               forall args . Fullfilled "_backward_prod(symbol)" args =>
                 String -> ArgsHMap "_backward_prod(symbol)" args -> IO SymbolHandle
_backward_prod name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_prod"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_radians(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_radians ::
                  forall args . Fullfilled "_backward_radians(symbol)" args =>
                    String ->
                      ArgsHMap "_backward_radians(symbol)" args -> IO SymbolHandle
_backward_radians name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_radians"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_rcbrt(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_rcbrt ::
                forall args . Fullfilled "_backward_rcbrt(symbol)" args =>
                  String ->
                    ArgsHMap "_backward_rcbrt(symbol)" args -> IO SymbolHandle
_backward_rcbrt name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_rcbrt"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_rdiv_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt SymbolHandle),
        '("rhs", AttrOpt SymbolHandle)]

_backward_rdiv_scalar ::
                      forall args . Fullfilled "_backward_rdiv_scalar(symbol)" args =>
                        String ->
                          ArgsHMap "_backward_rdiv_scalar(symbol)" args -> IO SymbolHandle
_backward_rdiv_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_rdiv_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_reciprocal(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_reciprocal ::
                     forall args . Fullfilled "_backward_reciprocal(symbol)" args =>
                       String ->
                         ArgsHMap "_backward_reciprocal(symbol)" args -> IO SymbolHandle
_backward_reciprocal name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_reciprocal"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_relu(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_relu ::
               forall args . Fullfilled "_backward_relu(symbol)" args =>
                 String -> ArgsHMap "_backward_relu(symbol)" args -> IO SymbolHandle
_backward_relu name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_relu"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_repeat(symbol)" = '[]

_backward_repeat ::
                 forall args . Fullfilled "_backward_repeat(symbol)" args =>
                   String ->
                     ArgsHMap "_backward_repeat(symbol)" args -> IO SymbolHandle
_backward_repeat name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_repeat"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_reverse(symbol)" = '[]

_backward_reverse ::
                  forall args . Fullfilled "_backward_reverse(symbol)" args =>
                    String ->
                      ArgsHMap "_backward_reverse(symbol)" args -> IO SymbolHandle
_backward_reverse name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_reverse"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_rmod_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt SymbolHandle),
        '("rhs", AttrOpt SymbolHandle)]

_backward_rmod_scalar ::
                      forall args . Fullfilled "_backward_rmod_scalar(symbol)" args =>
                        String ->
                          ArgsHMap "_backward_rmod_scalar(symbol)" args -> IO SymbolHandle
_backward_rmod_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_rmod_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_rpower_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt SymbolHandle),
        '("rhs", AttrOpt SymbolHandle)]

_backward_rpower_scalar ::
                        forall args . Fullfilled "_backward_rpower_scalar(symbol)" args =>
                          String ->
                            ArgsHMap "_backward_rpower_scalar(symbol)" args -> IO SymbolHandle
_backward_rpower_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_rpower_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_rsqrt(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_rsqrt ::
                forall args . Fullfilled "_backward_rsqrt(symbol)" args =>
                  String ->
                    ArgsHMap "_backward_rsqrt(symbol)" args -> IO SymbolHandle
_backward_rsqrt name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_rsqrt"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_sample_multinomial(symbol)"
     = '[]

_backward_sample_multinomial ::
                             forall args .
                               Fullfilled "_backward_sample_multinomial(symbol)" args =>
                               String ->
                                 ArgsHMap "_backward_sample_multinomial(symbol)" args ->
                                   IO SymbolHandle
_backward_sample_multinomial name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sample_multinomial"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_sigmoid(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_sigmoid ::
                  forall args . Fullfilled "_backward_sigmoid(symbol)" args =>
                    String ->
                      ArgsHMap "_backward_sigmoid(symbol)" args -> IO SymbolHandle
_backward_sigmoid name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sigmoid"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_sign(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_sign ::
               forall args . Fullfilled "_backward_sign(symbol)" args =>
                 String -> ArgsHMap "_backward_sign(symbol)" args -> IO SymbolHandle
_backward_sign name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sign"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_sin(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_sin ::
              forall args . Fullfilled "_backward_sin(symbol)" args =>
                String -> ArgsHMap "_backward_sin(symbol)" args -> IO SymbolHandle
_backward_sin name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sin"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_sinh(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_sinh ::
               forall args . Fullfilled "_backward_sinh(symbol)" args =>
                 String -> ArgsHMap "_backward_sinh(symbol)" args -> IO SymbolHandle
_backward_sinh name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sinh"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_slice(symbol)" = '[]

_backward_slice ::
                forall args . Fullfilled "_backward_slice(symbol)" args =>
                  String ->
                    ArgsHMap "_backward_slice(symbol)" args -> IO SymbolHandle
_backward_slice name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_slice"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_slice_axis(symbol)" = '[]

_backward_slice_axis ::
                     forall args . Fullfilled "_backward_slice_axis(symbol)" args =>
                       String ->
                         ArgsHMap "_backward_slice_axis(symbol)" args -> IO SymbolHandle
_backward_slice_axis name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_slice_axis"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_slice_like(symbol)" = '[]

_backward_slice_like ::
                     forall args . Fullfilled "_backward_slice_like(symbol)" args =>
                       String ->
                         ArgsHMap "_backward_slice_like(symbol)" args -> IO SymbolHandle
_backward_slice_like name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_slice_like"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_smooth_l1(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_smooth_l1 ::
                    forall args . Fullfilled "_backward_smooth_l1(symbol)" args =>
                      String ->
                        ArgsHMap "_backward_smooth_l1(symbol)" args -> IO SymbolHandle
_backward_smooth_l1 name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_smooth_l1"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_softmax(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_softmax ::
                  forall args . Fullfilled "_backward_softmax(symbol)" args =>
                    String ->
                      ArgsHMap "_backward_softmax(symbol)" args -> IO SymbolHandle
_backward_softmax name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_softmax"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance
     ParameterList "_backward_softmax_cross_entropy(symbol)" = '[]

_backward_softmax_cross_entropy ::
                                forall args .
                                  Fullfilled "_backward_softmax_cross_entropy(symbol)" args =>
                                  String ->
                                    ArgsHMap "_backward_softmax_cross_entropy(symbol)" args ->
                                      IO SymbolHandle
_backward_softmax_cross_entropy name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_softmax_cross_entropy"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_softsign(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_softsign ::
                   forall args . Fullfilled "_backward_softsign(symbol)" args =>
                     String ->
                       ArgsHMap "_backward_softsign(symbol)" args -> IO SymbolHandle
_backward_softsign name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_softsign"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_sparse_retain(symbol)" = '[]

_backward_sparse_retain ::
                        forall args . Fullfilled "_backward_sparse_retain(symbol)" args =>
                          String ->
                            ArgsHMap "_backward_sparse_retain(symbol)" args -> IO SymbolHandle
_backward_sparse_retain name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sparse_retain"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_sqrt(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_sqrt ::
               forall args . Fullfilled "_backward_sqrt(symbol)" args =>
                 String -> ArgsHMap "_backward_sqrt(symbol)" args -> IO SymbolHandle
_backward_sqrt name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sqrt"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_square(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_square ::
                 forall args . Fullfilled "_backward_square(symbol)" args =>
                   String ->
                     ArgsHMap "_backward_square(symbol)" args -> IO SymbolHandle
_backward_square name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_square"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_square_sum(symbol)" = '[]

_backward_square_sum ::
                     forall args . Fullfilled "_backward_square_sum(symbol)" args =>
                       String ->
                         ArgsHMap "_backward_square_sum(symbol)" args -> IO SymbolHandle
_backward_square_sum name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_square_sum"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_squeeze(symbol)" = '[]

_backward_squeeze ::
                  forall args . Fullfilled "_backward_squeeze(symbol)" args =>
                    String ->
                      ArgsHMap "_backward_squeeze(symbol)" args -> IO SymbolHandle
_backward_squeeze name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_squeeze"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_stack(symbol)" = '[]

_backward_stack ::
                forall args . Fullfilled "_backward_stack(symbol)" args =>
                  String ->
                    ArgsHMap "_backward_stack(symbol)" args -> IO SymbolHandle
_backward_stack name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_stack"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_sub(symbol)" = '[]

_backward_sub ::
              forall args . Fullfilled "_backward_sub(symbol)" args =>
                String -> ArgsHMap "_backward_sub(symbol)" args -> IO SymbolHandle
_backward_sub name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sub"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_sum(symbol)" = '[]

_backward_sum ::
              forall args . Fullfilled "_backward_sum(symbol)" args =>
                String -> ArgsHMap "_backward_sum(symbol)" args -> IO SymbolHandle
_backward_sum name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_sum"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_take(symbol)" = '[]

_backward_take ::
               forall args . Fullfilled "_backward_take(symbol)" args =>
                 String -> ArgsHMap "_backward_take(symbol)" args -> IO SymbolHandle
_backward_take name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_take"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_tan(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_tan ::
              forall args . Fullfilled "_backward_tan(symbol)" args =>
                String -> ArgsHMap "_backward_tan(symbol)" args -> IO SymbolHandle
_backward_tan name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_tan"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_tanh(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_backward_tanh ::
               forall args . Fullfilled "_backward_tanh(symbol)" args =>
                 String -> ArgsHMap "_backward_tanh(symbol)" args -> IO SymbolHandle
_backward_tanh name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_tanh"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_tile(symbol)" = '[]

_backward_tile ::
               forall args . Fullfilled "_backward_tile(symbol)" args =>
                 String -> ArgsHMap "_backward_tile(symbol)" args -> IO SymbolHandle
_backward_tile name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_tile"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_topk(symbol)" = '[]

_backward_topk ::
               forall args . Fullfilled "_backward_topk(symbol)" args =>
                 String -> ArgsHMap "_backward_topk(symbol)" args -> IO SymbolHandle
_backward_topk name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_topk"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_where(symbol)" = '[]

_backward_where ::
                forall args . Fullfilled "_backward_where(symbol)" args =>
                  String ->
                    ArgsHMap "_backward_where(symbol)" args -> IO SymbolHandle
_backward_where name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_where"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_backward_while_loop(symbol)" = '[]

_backward_while_loop ::
                     forall args . Fullfilled "_backward_while_loop(symbol)" args =>
                       String ->
                         ArgsHMap "_backward_while_loop(symbol)" args -> IO SymbolHandle
_backward_while_loop name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_backward_while_loop"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_broadcast_backward(symbol)" = '[]

_broadcast_backward ::
                    forall args . Fullfilled "_broadcast_backward(symbol)" args =>
                      String ->
                        ArgsHMap "_broadcast_backward(symbol)" args -> IO SymbolHandle
_broadcast_backward name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_broadcast_backward"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_AdaptiveAvgPooling2D(symbol)"
     =
     '[ '("output_size", AttrOpt [Int]),
        '("data", AttrOpt SymbolHandle)]

_contrib_AdaptiveAvgPooling2D ::
                              forall args .
                                Fullfilled "_contrib_AdaptiveAvgPooling2D(symbol)" args =>
                                String ->
                                  ArgsHMap "_contrib_AdaptiveAvgPooling2D(symbol)" args ->
                                    IO SymbolHandle
_contrib_AdaptiveAvgPooling2D name args
  = let scalarArgs
          = catMaybes
              [("output_size",) . showValue <$>
                 (args !? #output_size :: Maybe [Int])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_AdaptiveAvgPooling2D"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_BilinearResize2D(symbol)" =
     '[ '("height", AttrReq Int), '("width", AttrReq Int),
        '("data", AttrOpt SymbolHandle)]

_contrib_BilinearResize2D ::
                          forall args .
                            Fullfilled "_contrib_BilinearResize2D(symbol)" args =>
                            String ->
                              ArgsHMap "_contrib_BilinearResize2D(symbol)" args ->
                                IO SymbolHandle
_contrib_BilinearResize2D name args
  = let scalarArgs
          = catMaybes
              [("height",) . showValue <$> (args !? #height :: Maybe Int),
               ("width",) . showValue <$> (args !? #width :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_BilinearResize2D"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_CTCLoss(symbol)" =
     '[ '("use_data_lengths", AttrOpt Bool),
        '("use_label_lengths", AttrOpt Bool),
        '("blank_label", AttrOpt (EnumType '["first", "last"])),
        '("data", AttrOpt SymbolHandle), '("label", AttrOpt SymbolHandle),
        '("data_lengths", AttrOpt SymbolHandle),
        '("label_lengths", AttrOpt SymbolHandle)]

_contrib_CTCLoss ::
                 forall args . Fullfilled "_contrib_CTCLoss(symbol)" args =>
                   String ->
                     ArgsHMap "_contrib_CTCLoss(symbol)" args -> IO SymbolHandle
_contrib_CTCLoss name args
  = let scalarArgs
          = catMaybes
              [("use_data_lengths",) . showValue <$>
                 (args !? #use_data_lengths :: Maybe Bool),
               ("use_label_lengths",) . showValue <$>
                 (args !? #use_label_lengths :: Maybe Bool),
               ("blank_label",) . showValue <$>
                 (args !? #blank_label :: Maybe (EnumType '["first", "last"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("label",) <$> (args !? #label :: Maybe SymbolHandle),
               ("data_lengths",) <$>
                 (args !? #data_lengths :: Maybe SymbolHandle),
               ("label_lengths",) <$>
                 (args !? #label_lengths :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_CTCLoss"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance
     ParameterList "_contrib_DeformableConvolution(symbol)" =
     '[ '("kernel", AttrReq [Int]), '("stride", AttrOpt [Int]),
        '("dilate", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("num_filter", AttrReq Int), '("num_group", AttrOpt Int),
        '("num_deformable_group", AttrOpt Int),
        '("workspace", AttrOpt Int), '("no_bias", AttrOpt Bool),
        '("layout", AttrOpt (Maybe (EnumType '["NCDHW", "NCHW", "NCW"]))),
        '("data", AttrOpt SymbolHandle), '("offset", AttrOpt SymbolHandle),
        '("weight", AttrOpt SymbolHandle), '("bias", AttrOpt SymbolHandle)]

_contrib_DeformableConvolution ::
                               forall args .
                                 Fullfilled "_contrib_DeformableConvolution(symbol)" args =>
                                 String ->
                                   ArgsHMap "_contrib_DeformableConvolution(symbol)" args ->
                                     IO SymbolHandle
_contrib_DeformableConvolution name args
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
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("offset",) <$> (args !? #offset :: Maybe SymbolHandle),
               ("weight",) <$> (args !? #weight :: Maybe SymbolHandle),
               ("bias",) <$> (args !? #bias :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_DeformableConvolution"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance
     ParameterList "_contrib_DeformablePSROIPooling(symbol)" =
     '[ '("spatial_scale", AttrReq Float), '("output_dim", AttrReq Int),
        '("group_size", AttrReq Int), '("pooled_size", AttrReq Int),
        '("part_size", AttrOpt Int), '("sample_per_part", AttrOpt Int),
        '("trans_std", AttrOpt Float), '("no_trans", AttrOpt Bool),
        '("data", AttrOpt SymbolHandle), '("rois", AttrOpt SymbolHandle),
        '("trans", AttrOpt SymbolHandle)]

_contrib_DeformablePSROIPooling ::
                                forall args .
                                  Fullfilled "_contrib_DeformablePSROIPooling(symbol)" args =>
                                  String ->
                                    ArgsHMap "_contrib_DeformablePSROIPooling(symbol)" args ->
                                      IO SymbolHandle
_contrib_DeformablePSROIPooling name args
  = let scalarArgs
          = catMaybes
              [("spatial_scale",) . showValue <$>
                 (args !? #spatial_scale :: Maybe Float),
               ("output_dim",) . showValue <$> (args !? #output_dim :: Maybe Int),
               ("group_size",) . showValue <$> (args !? #group_size :: Maybe Int),
               ("pooled_size",) . showValue <$>
                 (args !? #pooled_size :: Maybe Int),
               ("part_size",) . showValue <$> (args !? #part_size :: Maybe Int),
               ("sample_per_part",) . showValue <$>
                 (args !? #sample_per_part :: Maybe Int),
               ("trans_std",) . showValue <$> (args !? #trans_std :: Maybe Float),
               ("no_trans",) . showValue <$> (args !? #no_trans :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("rois",) <$> (args !? #rois :: Maybe SymbolHandle),
               ("trans",) <$> (args !? #trans :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_DeformablePSROIPooling"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_MultiBoxDetection(symbol)" =
     '[ '("clip", AttrOpt Bool), '("threshold", AttrOpt Float),
        '("background_id", AttrOpt Int), '("nms_threshold", AttrOpt Float),
        '("force_suppress", AttrOpt Bool), '("variances", AttrOpt [Float]),
        '("nms_topk", AttrOpt Int), '("cls_prob", AttrOpt SymbolHandle),
        '("loc_pred", AttrOpt SymbolHandle),
        '("anchor", AttrOpt SymbolHandle)]

_contrib_MultiBoxDetection ::
                           forall args .
                             Fullfilled "_contrib_MultiBoxDetection(symbol)" args =>
                             String ->
                               ArgsHMap "_contrib_MultiBoxDetection(symbol)" args ->
                                 IO SymbolHandle
_contrib_MultiBoxDetection name args
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
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("cls_prob",) <$> (args !? #cls_prob :: Maybe SymbolHandle),
               ("loc_pred",) <$> (args !? #loc_pred :: Maybe SymbolHandle),
               ("anchor",) <$> (args !? #anchor :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_MultiBoxDetection"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_MultiBoxPrior(symbol)" =
     '[ '("sizes", AttrOpt [Float]), '("ratios", AttrOpt [Float]),
        '("clip", AttrOpt Bool), '("steps", AttrOpt [Float]),
        '("offsets", AttrOpt [Float]), '("data", AttrOpt SymbolHandle)]

_contrib_MultiBoxPrior ::
                       forall args . Fullfilled "_contrib_MultiBoxPrior(symbol)" args =>
                         String ->
                           ArgsHMap "_contrib_MultiBoxPrior(symbol)" args -> IO SymbolHandle
_contrib_MultiBoxPrior name args
  = let scalarArgs
          = catMaybes
              [("sizes",) . showValue <$> (args !? #sizes :: Maybe [Float]),
               ("ratios",) . showValue <$> (args !? #ratios :: Maybe [Float]),
               ("clip",) . showValue <$> (args !? #clip :: Maybe Bool),
               ("steps",) . showValue <$> (args !? #steps :: Maybe [Float]),
               ("offsets",) . showValue <$> (args !? #offsets :: Maybe [Float])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_MultiBoxPrior"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_MultiBoxTarget(symbol)" =
     '[ '("overlap_threshold", AttrOpt Float),
        '("ignore_label", AttrOpt Float),
        '("negative_mining_ratio", AttrOpt Float),
        '("negative_mining_thresh", AttrOpt Float),
        '("minimum_negative_samples", AttrOpt Int),
        '("variances", AttrOpt [Float]), '("anchor", AttrOpt SymbolHandle),
        '("label", AttrOpt SymbolHandle),
        '("cls_pred", AttrOpt SymbolHandle)]

_contrib_MultiBoxTarget ::
                        forall args . Fullfilled "_contrib_MultiBoxTarget(symbol)" args =>
                          String ->
                            ArgsHMap "_contrib_MultiBoxTarget(symbol)" args -> IO SymbolHandle
_contrib_MultiBoxTarget name args
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
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("anchor",) <$> (args !? #anchor :: Maybe SymbolHandle),
               ("label",) <$> (args !? #label :: Maybe SymbolHandle),
               ("cls_pred",) <$> (args !? #cls_pred :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_MultiBoxTarget"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_MultiProposal(symbol)" =
     '[ '("rpn_pre_nms_top_n", AttrOpt Int),
        '("rpn_post_nms_top_n", AttrOpt Int),
        '("threshold", AttrOpt Float), '("rpn_min_size", AttrOpt Int),
        '("scales", AttrOpt [Float]), '("ratios", AttrOpt [Float]),
        '("feature_stride", AttrOpt Int), '("output_score", AttrOpt Bool),
        '("iou_loss", AttrOpt Bool), '("cls_prob", AttrOpt SymbolHandle),
        '("bbox_pred", AttrOpt SymbolHandle),
        '("im_info", AttrOpt SymbolHandle)]

_contrib_MultiProposal ::
                       forall args . Fullfilled "_contrib_MultiProposal(symbol)" args =>
                         String ->
                           ArgsHMap "_contrib_MultiProposal(symbol)" args -> IO SymbolHandle
_contrib_MultiProposal name args
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
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("cls_prob",) <$> (args !? #cls_prob :: Maybe SymbolHandle),
               ("bbox_pred",) <$> (args !? #bbox_pred :: Maybe SymbolHandle),
               ("im_info",) <$> (args !? #im_info :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_MultiProposal"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_PSROIPooling(symbol)" =
     '[ '("spatial_scale", AttrReq Float), '("output_dim", AttrReq Int),
        '("pooled_size", AttrReq Int), '("group_size", AttrOpt Int),
        '("data", AttrOpt SymbolHandle), '("rois", AttrOpt SymbolHandle)]

_contrib_PSROIPooling ::
                      forall args . Fullfilled "_contrib_PSROIPooling(symbol)" args =>
                        String ->
                          ArgsHMap "_contrib_PSROIPooling(symbol)" args -> IO SymbolHandle
_contrib_PSROIPooling name args
  = let scalarArgs
          = catMaybes
              [("spatial_scale",) . showValue <$>
                 (args !? #spatial_scale :: Maybe Float),
               ("output_dim",) . showValue <$> (args !? #output_dim :: Maybe Int),
               ("pooled_size",) . showValue <$>
                 (args !? #pooled_size :: Maybe Int),
               ("group_size",) . showValue <$> (args !? #group_size :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("rois",) <$> (args !? #rois :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_PSROIPooling"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_Proposal(symbol)" =
     '[ '("rpn_pre_nms_top_n", AttrOpt Int),
        '("rpn_post_nms_top_n", AttrOpt Int),
        '("threshold", AttrOpt Float), '("rpn_min_size", AttrOpt Int),
        '("scales", AttrOpt [Float]), '("ratios", AttrOpt [Float]),
        '("feature_stride", AttrOpt Int), '("output_score", AttrOpt Bool),
        '("iou_loss", AttrOpt Bool), '("cls_prob", AttrOpt SymbolHandle),
        '("bbox_pred", AttrOpt SymbolHandle),
        '("im_info", AttrOpt SymbolHandle)]

_contrib_Proposal ::
                  forall args . Fullfilled "_contrib_Proposal(symbol)" args =>
                    String ->
                      ArgsHMap "_contrib_Proposal(symbol)" args -> IO SymbolHandle
_contrib_Proposal name args
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
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("cls_prob",) <$> (args !? #cls_prob :: Maybe SymbolHandle),
               ("bbox_pred",) <$> (args !? #bbox_pred :: Maybe SymbolHandle),
               ("im_info",) <$> (args !? #im_info :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_Proposal"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_ROIAlign(symbol)" =
     '[ '("pooled_size", AttrReq [Int]),
        '("spatial_scale", AttrReq Float), '("sample_ratio", AttrOpt Int),
        '("data", AttrOpt SymbolHandle), '("rois", AttrOpt SymbolHandle)]

_contrib_ROIAlign ::
                  forall args . Fullfilled "_contrib_ROIAlign(symbol)" args =>
                    String ->
                      ArgsHMap "_contrib_ROIAlign(symbol)" args -> IO SymbolHandle
_contrib_ROIAlign name args
  = let scalarArgs
          = catMaybes
              [("pooled_size",) . showValue <$>
                 (args !? #pooled_size :: Maybe [Int]),
               ("spatial_scale",) . showValue <$>
                 (args !? #spatial_scale :: Maybe Float),
               ("sample_ratio",) . showValue <$>
                 (args !? #sample_ratio :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("rois",) <$> (args !? #rois :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_ROIAlign"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_SparseEmbedding(symbol)" =
     '[ '("input_dim", AttrReq Int), '("output_dim", AttrReq Int),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"])),
        '("sparse_grad", AttrOpt Bool), '("data", AttrOpt SymbolHandle),
        '("weight", AttrOpt SymbolHandle)]

_contrib_SparseEmbedding ::
                         forall args . Fullfilled "_contrib_SparseEmbedding(symbol)" args =>
                           String ->
                             ArgsHMap "_contrib_SparseEmbedding(symbol)" args -> IO SymbolHandle
_contrib_SparseEmbedding name args
  = let scalarArgs
          = catMaybes
              [("input_dim",) . showValue <$> (args !? #input_dim :: Maybe Int),
               ("output_dim",) . showValue <$> (args !? #output_dim :: Maybe Int),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["float16", "float32", "float64", "int32", "int64", "int8",
                           "uint8"])),
               ("sparse_grad",) . showValue <$>
                 (args !? #sparse_grad :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("weight",) <$> (args !? #weight :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_SparseEmbedding"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_SyncBatchNorm(symbol)" =
     '[ '("eps", AttrOpt Float), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool), '("ndev", AttrOpt Int),
        '("key", AttrOpt String), '("data", AttrOpt SymbolHandle),
        '("gamma", AttrOpt SymbolHandle), '("beta", AttrOpt SymbolHandle),
        '("moving_mean", AttrOpt SymbolHandle),
        '("moving_var", AttrOpt SymbolHandle)]

_contrib_SyncBatchNorm ::
                       forall args . Fullfilled "_contrib_SyncBatchNorm(symbol)" args =>
                         String ->
                           ArgsHMap "_contrib_SyncBatchNorm(symbol)" args -> IO SymbolHandle
_contrib_SyncBatchNorm name args
  = let scalarArgs
          = catMaybes
              [("eps",) . showValue <$> (args !? #eps :: Maybe Float),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float),
               ("fix_gamma",) . showValue <$> (args !? #fix_gamma :: Maybe Bool),
               ("use_global_stats",) . showValue <$>
                 (args !? #use_global_stats :: Maybe Bool),
               ("output_mean_var",) . showValue <$>
                 (args !? #output_mean_var :: Maybe Bool),
               ("ndev",) . showValue <$> (args !? #ndev :: Maybe Int),
               ("key",) . showValue <$> (args !? #key :: Maybe String)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("gamma",) <$> (args !? #gamma :: Maybe SymbolHandle),
               ("beta",) <$> (args !? #beta :: Maybe SymbolHandle),
               ("moving_mean",) <$> (args !? #moving_mean :: Maybe SymbolHandle),
               ("moving_var",) <$> (args !? #moving_var :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_SyncBatchNorm"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_backward_quadratic(symbol)" =
     '[]

_contrib_backward_quadratic ::
                            forall args .
                              Fullfilled "_contrib_backward_quadratic(symbol)" args =>
                              String ->
                                ArgsHMap "_contrib_backward_quadratic(symbol)" args ->
                                  IO SymbolHandle
_contrib_backward_quadratic name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_backward_quadratic"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_bipartite_matching(symbol)" =
     '[ '("is_ascend", AttrOpt Bool), '("threshold", AttrReq Float),
        '("topk", AttrOpt Int), '("data", AttrOpt SymbolHandle)]

_contrib_bipartite_matching ::
                            forall args .
                              Fullfilled "_contrib_bipartite_matching(symbol)" args =>
                              String ->
                                ArgsHMap "_contrib_bipartite_matching(symbol)" args ->
                                  IO SymbolHandle
_contrib_bipartite_matching name args
  = let scalarArgs
          = catMaybes
              [("is_ascend",) . showValue <$> (args !? #is_ascend :: Maybe Bool),
               ("threshold",) . showValue <$> (args !? #threshold :: Maybe Float),
               ("topk",) . showValue <$> (args !? #topk :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_bipartite_matching"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_box_iou(symbol)" =
     '[ '("format", AttrOpt (EnumType '["center", "corner"])),
        '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_contrib_box_iou ::
                 forall args . Fullfilled "_contrib_box_iou(symbol)" args =>
                   String ->
                     ArgsHMap "_contrib_box_iou(symbol)" args -> IO SymbolHandle
_contrib_box_iou name args
  = let scalarArgs
          = catMaybes
              [("format",) . showValue <$>
                 (args !? #format :: Maybe (EnumType '["center", "corner"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_box_iou"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_box_nms(symbol)" =
     '[ '("overlap_thresh", AttrOpt Float),
        '("valid_thresh", AttrOpt Float), '("topk", AttrOpt Int),
        '("coord_start", AttrOpt Int), '("score_index", AttrOpt Int),
        '("id_index", AttrOpt Int), '("force_suppress", AttrOpt Bool),
        '("in_format", AttrOpt (EnumType '["center", "corner"])),
        '("out_format", AttrOpt (EnumType '["center", "corner"])),
        '("data", AttrOpt SymbolHandle)]

_contrib_box_nms ::
                 forall args . Fullfilled "_contrib_box_nms(symbol)" args =>
                   String ->
                     ArgsHMap "_contrib_box_nms(symbol)" args -> IO SymbolHandle
_contrib_box_nms name args
  = let scalarArgs
          = catMaybes
              [("overlap_thresh",) . showValue <$>
                 (args !? #overlap_thresh :: Maybe Float),
               ("valid_thresh",) . showValue <$>
                 (args !? #valid_thresh :: Maybe Float),
               ("topk",) . showValue <$> (args !? #topk :: Maybe Int),
               ("coord_start",) . showValue <$>
                 (args !? #coord_start :: Maybe Int),
               ("score_index",) . showValue <$>
                 (args !? #score_index :: Maybe Int),
               ("id_index",) . showValue <$> (args !? #id_index :: Maybe Int),
               ("force_suppress",) . showValue <$>
                 (args !? #force_suppress :: Maybe Bool),
               ("in_format",) . showValue <$>
                 (args !? #in_format :: Maybe (EnumType '["center", "corner"])),
               ("out_format",) . showValue <$>
                 (args !? #out_format :: Maybe (EnumType '["center", "corner"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_box_nms"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_count_sketch(symbol)" =
     '[ '("out_dim", AttrReq Int),
        '("processing_batch_size", AttrOpt Int),
        '("data", AttrOpt SymbolHandle), '("h", AttrOpt SymbolHandle),
        '("s", AttrOpt SymbolHandle)]

_contrib_count_sketch ::
                      forall args . Fullfilled "_contrib_count_sketch(symbol)" args =>
                        String ->
                          ArgsHMap "_contrib_count_sketch(symbol)" args -> IO SymbolHandle
_contrib_count_sketch name args
  = let scalarArgs
          = catMaybes
              [("out_dim",) . showValue <$> (args !? #out_dim :: Maybe Int),
               ("processing_batch_size",) . showValue <$>
                 (args !? #processing_batch_size :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("h",) <$> (args !? #h :: Maybe SymbolHandle),
               ("s",) <$> (args !? #s :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_count_sketch"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_dequantize(symbol)" =
     '[ '("out_type", AttrOpt (EnumType '["float32"])),
        '("data", AttrOpt SymbolHandle),
        '("min_range", AttrOpt SymbolHandle),
        '("max_range", AttrOpt SymbolHandle)]

_contrib_dequantize ::
                    forall args . Fullfilled "_contrib_dequantize(symbol)" args =>
                      String ->
                        ArgsHMap "_contrib_dequantize(symbol)" args -> IO SymbolHandle
_contrib_dequantize name args
  = let scalarArgs
          = catMaybes
              [("out_type",) . showValue <$>
                 (args !? #out_type :: Maybe (EnumType '["float32"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("min_range",) <$> (args !? #min_range :: Maybe SymbolHandle),
               ("max_range",) <$> (args !? #max_range :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_dequantize"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_div_sqrt_dim(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

_contrib_div_sqrt_dim ::
                      forall args . Fullfilled "_contrib_div_sqrt_dim(symbol)" args =>
                        String ->
                          ArgsHMap "_contrib_div_sqrt_dim(symbol)" args -> IO SymbolHandle
_contrib_div_sqrt_dim name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_div_sqrt_dim"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_fft(symbol)" =
     '[ '("compute_size", AttrOpt Int), '("data", AttrOpt SymbolHandle)]

_contrib_fft ::
             forall args . Fullfilled "_contrib_fft(symbol)" args =>
               String -> ArgsHMap "_contrib_fft(symbol)" args -> IO SymbolHandle
_contrib_fft name args
  = let scalarArgs
          = catMaybes
              [("compute_size",) . showValue <$>
                 (args !? #compute_size :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_fft"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_ifft(symbol)" =
     '[ '("compute_size", AttrOpt Int), '("data", AttrOpt SymbolHandle)]

_contrib_ifft ::
              forall args . Fullfilled "_contrib_ifft(symbol)" args =>
                String -> ArgsHMap "_contrib_ifft(symbol)" args -> IO SymbolHandle
_contrib_ifft name args
  = let scalarArgs
          = catMaybes
              [("compute_size",) . showValue <$>
                 (args !? #compute_size :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_ifft"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_quadratic(symbol)" =
     '[ '("a", AttrOpt Float), '("b", AttrOpt Float),
        '("c", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_contrib_quadratic ::
                   forall args . Fullfilled "_contrib_quadratic(symbol)" args =>
                     String ->
                       ArgsHMap "_contrib_quadratic(symbol)" args -> IO SymbolHandle
_contrib_quadratic name args
  = let scalarArgs
          = catMaybes
              [("a",) . showValue <$> (args !? #a :: Maybe Float),
               ("b",) . showValue <$> (args !? #b :: Maybe Float),
               ("c",) . showValue <$> (args !? #c :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_quadratic"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_quantize(symbol)" =
     '[ '("out_type", AttrOpt (EnumType '["int8", "uint8"])),
        '("data", AttrOpt SymbolHandle),
        '("min_range", AttrOpt SymbolHandle),
        '("max_range", AttrOpt SymbolHandle)]

_contrib_quantize ::
                  forall args . Fullfilled "_contrib_quantize(symbol)" args =>
                    String ->
                      ArgsHMap "_contrib_quantize(symbol)" args -> IO SymbolHandle
_contrib_quantize name args
  = let scalarArgs
          = catMaybes
              [("out_type",) . showValue <$>
                 (args !? #out_type :: Maybe (EnumType '["int8", "uint8"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("min_range",) <$> (args !? #min_range :: Maybe SymbolHandle),
               ("max_range",) <$> (args !? #max_range :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_quantize"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_quantized_conv(symbol)" =
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
        '("data", AttrOpt SymbolHandle), '("weight", AttrOpt SymbolHandle),
        '("bias", AttrOpt SymbolHandle),
        '("min_data", AttrOpt SymbolHandle),
        '("max_data", AttrOpt SymbolHandle),
        '("min_weight", AttrOpt SymbolHandle),
        '("max_weight", AttrOpt SymbolHandle),
        '("min_bias", AttrOpt SymbolHandle),
        '("max_bias", AttrOpt SymbolHandle)]

_contrib_quantized_conv ::
                        forall args . Fullfilled "_contrib_quantized_conv(symbol)" args =>
                          String ->
                            ArgsHMap "_contrib_quantized_conv(symbol)" args -> IO SymbolHandle
_contrib_quantized_conv name args
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
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("weight",) <$> (args !? #weight :: Maybe SymbolHandle),
               ("bias",) <$> (args !? #bias :: Maybe SymbolHandle),
               ("min_data",) <$> (args !? #min_data :: Maybe SymbolHandle),
               ("max_data",) <$> (args !? #max_data :: Maybe SymbolHandle),
               ("min_weight",) <$> (args !? #min_weight :: Maybe SymbolHandle),
               ("max_weight",) <$> (args !? #max_weight :: Maybe SymbolHandle),
               ("min_bias",) <$> (args !? #min_bias :: Maybe SymbolHandle),
               ("max_bias",) <$> (args !? #max_bias :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_quantized_conv"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_quantized_flatten(symbol)" =
     '[ '("data", AttrOpt SymbolHandle),
        '("min_data", AttrOpt SymbolHandle),
        '("max_data", AttrOpt SymbolHandle)]

_contrib_quantized_flatten ::
                           forall args .
                             Fullfilled "_contrib_quantized_flatten(symbol)" args =>
                             String ->
                               ArgsHMap "_contrib_quantized_flatten(symbol)" args ->
                                 IO SymbolHandle
_contrib_quantized_flatten name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("min_data",) <$> (args !? #min_data :: Maybe SymbolHandle),
               ("max_data",) <$> (args !? #max_data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_quantized_flatten"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance
     ParameterList "_contrib_quantized_fully_connected(symbol)" =
     '[ '("num_hidden", AttrReq Int), '("no_bias", AttrOpt Bool),
        '("flatten", AttrOpt Bool), '("data", AttrOpt SymbolHandle),
        '("weight", AttrOpt SymbolHandle), '("bias", AttrOpt SymbolHandle),
        '("min_data", AttrOpt SymbolHandle),
        '("max_data", AttrOpt SymbolHandle),
        '("min_weight", AttrOpt SymbolHandle),
        '("max_weight", AttrOpt SymbolHandle),
        '("min_bias", AttrOpt SymbolHandle),
        '("max_bias", AttrOpt SymbolHandle)]

_contrib_quantized_fully_connected ::
                                   forall args .
                                     Fullfilled "_contrib_quantized_fully_connected(symbol)" args =>
                                     String ->
                                       ArgsHMap "_contrib_quantized_fully_connected(symbol)" args ->
                                         IO SymbolHandle
_contrib_quantized_fully_connected name args
  = let scalarArgs
          = catMaybes
              [("num_hidden",) . showValue <$>
                 (args !? #num_hidden :: Maybe Int),
               ("no_bias",) . showValue <$> (args !? #no_bias :: Maybe Bool),
               ("flatten",) . showValue <$> (args !? #flatten :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("weight",) <$> (args !? #weight :: Maybe SymbolHandle),
               ("bias",) <$> (args !? #bias :: Maybe SymbolHandle),
               ("min_data",) <$> (args !? #min_data :: Maybe SymbolHandle),
               ("max_data",) <$> (args !? #max_data :: Maybe SymbolHandle),
               ("min_weight",) <$> (args !? #min_weight :: Maybe SymbolHandle),
               ("max_weight",) <$> (args !? #max_weight :: Maybe SymbolHandle),
               ("min_bias",) <$> (args !? #min_bias :: Maybe SymbolHandle),
               ("max_bias",) <$> (args !? #max_bias :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_quantized_fully_connected"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_quantized_pooling(symbol)" =
     '[ '("kernel", AttrOpt [Int]),
        '("pool_type", AttrOpt (EnumType '["avg", "lp", "max", "sum"])),
        '("global_pool", AttrOpt Bool), '("cudnn_off", AttrOpt Bool),
        '("pooling_convention", AttrOpt (EnumType '["full", "valid"])),
        '("stride", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("p_value", AttrOpt (Maybe Int)),
        '("count_include_pad", AttrOpt (Maybe Bool)),
        '("data", AttrOpt SymbolHandle),
        '("min_data", AttrOpt SymbolHandle),
        '("max_data", AttrOpt SymbolHandle)]

_contrib_quantized_pooling ::
                           forall args .
                             Fullfilled "_contrib_quantized_pooling(symbol)" args =>
                             String ->
                               ArgsHMap "_contrib_quantized_pooling(symbol)" args ->
                                 IO SymbolHandle
_contrib_quantized_pooling name args
  = let scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> (args !? #kernel :: Maybe [Int]),
               ("pool_type",) . showValue <$>
                 (args !? #pool_type ::
                    Maybe (EnumType '["avg", "lp", "max", "sum"])),
               ("global_pool",) . showValue <$>
                 (args !? #global_pool :: Maybe Bool),
               ("cudnn_off",) . showValue <$> (args !? #cudnn_off :: Maybe Bool),
               ("pooling_convention",) . showValue <$>
                 (args !? #pooling_convention ::
                    Maybe (EnumType '["full", "valid"])),
               ("stride",) . showValue <$> (args !? #stride :: Maybe [Int]),
               ("pad",) . showValue <$> (args !? #pad :: Maybe [Int]),
               ("p_value",) . showValue <$>
                 (args !? #p_value :: Maybe (Maybe Int)),
               ("count_include_pad",) . showValue <$>
                 (args !? #count_include_pad :: Maybe (Maybe Bool))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("min_data",) <$> (args !? #min_data :: Maybe SymbolHandle),
               ("max_data",) <$> (args !? #max_data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_quantized_pooling"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_contrib_requantize(symbol)" =
     '[ '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)),
        '("data", AttrOpt SymbolHandle),
        '("min_range", AttrOpt SymbolHandle),
        '("max_range", AttrOpt SymbolHandle)]

_contrib_requantize ::
                    forall args . Fullfilled "_contrib_requantize(symbol)" args =>
                      String ->
                        ArgsHMap "_contrib_requantize(symbol)" args -> IO SymbolHandle
_contrib_requantize name args
  = let scalarArgs
          = catMaybes
              [("min_calib_range",) . showValue <$>
                 (args !? #min_calib_range :: Maybe (Maybe Float)),
               ("max_calib_range",) . showValue <$>
                 (args !? #max_calib_range :: Maybe (Maybe Float))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("min_range",) <$> (args !? #min_range :: Maybe SymbolHandle),
               ("max_range",) <$> (args !? #max_range :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_contrib_requantize"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_copy(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

_copy ::
      forall args . Fullfilled "_copy(symbol)" args =>
        String -> ArgsHMap "_copy(symbol)" args -> IO SymbolHandle
_copy name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_copy"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_cvimread(symbol)" =
     '[ '("filename", AttrReq String), '("flag", AttrOpt Int),
        '("to_rgb", AttrOpt Bool)]

_cvimread ::
          forall args . Fullfilled "_cvimread(symbol)" args =>
            String -> ArgsHMap "_cvimread(symbol)" args -> IO SymbolHandle
_cvimread name args
  = let scalarArgs
          = catMaybes
              [("filename",) . showValue <$> (args !? #filename :: Maybe String),
               ("flag",) . showValue <$> (args !? #flag :: Maybe Int),
               ("to_rgb",) . showValue <$> (args !? #to_rgb :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_cvimread"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_div_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_div_scalar ::
            forall args . Fullfilled "_div_scalar(symbol)" args =>
              String -> ArgsHMap "_div_scalar(symbol)" args -> IO SymbolHandle
_div_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_div_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_equal(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_equal ::
       forall args . Fullfilled "_equal(symbol)" args =>
         String -> ArgsHMap "_equal(symbol)" args -> IO SymbolHandle
_equal name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_equal"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_equal_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_equal_scalar ::
              forall args . Fullfilled "_equal_scalar(symbol)" args =>
                String -> ArgsHMap "_equal_scalar(symbol)" args -> IO SymbolHandle
_equal_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_equal_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_eye(symbol)" =
     '[ '("_N", AttrReq Int), '("_M", AttrOpt Int), '("k", AttrOpt Int),
        '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"]))]

_eye ::
     forall args . Fullfilled "_eye(symbol)" args =>
       String -> ArgsHMap "_eye(symbol)" args -> IO SymbolHandle
_eye name args
  = let scalarArgs
          = catMaybes
              [("_N",) . showValue <$> (args !? #_N :: Maybe Int),
               ("_M",) . showValue <$> (args !? #_M :: Maybe Int),
               ("k",) . showValue <$> (args !? #k :: Maybe Int),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["float16", "float32", "float64", "int32", "int64", "int8",
                           "uint8"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_eye"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_full(symbol)" =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"])),
        '("value", AttrReq Double)]

_full ::
      forall args . Fullfilled "_full(symbol)" args =>
        String -> ArgsHMap "_full(symbol)" args -> IO SymbolHandle
_full name args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["float16", "float32", "float64", "int32", "int64", "int8",
                           "uint8"])),
               ("value",) . showValue <$> (args !? #value :: Maybe Double)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_full"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_grad_add(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_grad_add ::
          forall args . Fullfilled "_grad_add(symbol)" args =>
            String -> ArgsHMap "_grad_add(symbol)" args -> IO SymbolHandle
_grad_add name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_grad_add"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_greater(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_greater ::
         forall args . Fullfilled "_greater(symbol)" args =>
           String -> ArgsHMap "_greater(symbol)" args -> IO SymbolHandle
_greater name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_greater"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_greater_equal(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_greater_equal ::
               forall args . Fullfilled "_greater_equal(symbol)" args =>
                 String -> ArgsHMap "_greater_equal(symbol)" args -> IO SymbolHandle
_greater_equal name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_greater_equal"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_greater_equal_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_greater_equal_scalar ::
                      forall args . Fullfilled "_greater_equal_scalar(symbol)" args =>
                        String ->
                          ArgsHMap "_greater_equal_scalar(symbol)" args -> IO SymbolHandle
_greater_equal_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_greater_equal_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_greater_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_greater_scalar ::
                forall args . Fullfilled "_greater_scalar(symbol)" args =>
                  String ->
                    ArgsHMap "_greater_scalar(symbol)" args -> IO SymbolHandle
_greater_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_greater_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_histogram(symbol)" =
     '[ '("bin_cnt", AttrOpt (Maybe Int)), '("range", AttrOpt Int),
        '("data", AttrOpt SymbolHandle), '("bins", AttrOpt SymbolHandle)]

_histogram ::
           forall args . Fullfilled "_histogram(symbol)" args =>
             String -> ArgsHMap "_histogram(symbol)" args -> IO SymbolHandle
_histogram name args
  = let scalarArgs
          = catMaybes
              [("bin_cnt",) . showValue <$>
                 (args !? #bin_cnt :: Maybe (Maybe Int)),
               ("range",) . showValue <$> (args !? #range :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("bins",) <$> (args !? #bins :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_histogram"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_hypot(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_hypot ::
       forall args . Fullfilled "_hypot(symbol)" args =>
         String -> ArgsHMap "_hypot(symbol)" args -> IO SymbolHandle
_hypot name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_hypot"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_hypot_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_hypot_scalar ::
              forall args . Fullfilled "_hypot_scalar(symbol)" args =>
                String -> ArgsHMap "_hypot_scalar(symbol)" args -> IO SymbolHandle
_hypot_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_hypot_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_identity_with_attr_like_rhs(symbol)"
     =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_identity_with_attr_like_rhs ::
                             forall args .
                               Fullfilled "_identity_with_attr_like_rhs(symbol)" args =>
                               String ->
                                 ArgsHMap "_identity_with_attr_like_rhs(symbol)" args ->
                                   IO SymbolHandle
_identity_with_attr_like_rhs name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_identity_with_attr_like_rhs"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_image_adjust_lighting(symbol)" =
     '[ '("alpha", AttrReq [Float]), '("data", AttrOpt SymbolHandle)]

_image_adjust_lighting ::
                       forall args . Fullfilled "_image_adjust_lighting(symbol)" args =>
                         String ->
                           ArgsHMap "_image_adjust_lighting(symbol)" args -> IO SymbolHandle
_image_adjust_lighting name args
  = let scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> (args !? #alpha :: Maybe [Float])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_image_adjust_lighting"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_image_flip_left_right(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

_image_flip_left_right ::
                       forall args . Fullfilled "_image_flip_left_right(symbol)" args =>
                         String ->
                           ArgsHMap "_image_flip_left_right(symbol)" args -> IO SymbolHandle
_image_flip_left_right name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_image_flip_left_right"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_image_flip_top_bottom(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

_image_flip_top_bottom ::
                       forall args . Fullfilled "_image_flip_top_bottom(symbol)" args =>
                         String ->
                           ArgsHMap "_image_flip_top_bottom(symbol)" args -> IO SymbolHandle
_image_flip_top_bottom name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_image_flip_top_bottom"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_image_normalize(symbol)" =
     '[ '("mean", AttrReq [Float]), '("std", AttrReq [Float]),
        '("data", AttrOpt SymbolHandle)]

_image_normalize ::
                 forall args . Fullfilled "_image_normalize(symbol)" args =>
                   String ->
                     ArgsHMap "_image_normalize(symbol)" args -> IO SymbolHandle
_image_normalize name args
  = let scalarArgs
          = catMaybes
              [("mean",) . showValue <$> (args !? #mean :: Maybe [Float]),
               ("std",) . showValue <$> (args !? #std :: Maybe [Float])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_image_normalize"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_image_random_brightness(symbol)" =
     '[ '("min_factor", AttrReq Float), '("max_factor", AttrReq Float),
        '("data", AttrOpt SymbolHandle)]

_image_random_brightness ::
                         forall args . Fullfilled "_image_random_brightness(symbol)" args =>
                           String ->
                             ArgsHMap "_image_random_brightness(symbol)" args -> IO SymbolHandle
_image_random_brightness name args
  = let scalarArgs
          = catMaybes
              [("min_factor",) . showValue <$>
                 (args !? #min_factor :: Maybe Float),
               ("max_factor",) . showValue <$>
                 (args !? #max_factor :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_image_random_brightness"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_image_random_color_jitter(symbol)" =
     '[ '("brightness", AttrReq Float), '("contrast", AttrReq Float),
        '("saturation", AttrReq Float), '("hue", AttrReq Float),
        '("data", AttrOpt SymbolHandle)]

_image_random_color_jitter ::
                           forall args .
                             Fullfilled "_image_random_color_jitter(symbol)" args =>
                             String ->
                               ArgsHMap "_image_random_color_jitter(symbol)" args ->
                                 IO SymbolHandle
_image_random_color_jitter name args
  = let scalarArgs
          = catMaybes
              [("brightness",) . showValue <$>
                 (args !? #brightness :: Maybe Float),
               ("contrast",) . showValue <$> (args !? #contrast :: Maybe Float),
               ("saturation",) . showValue <$>
                 (args !? #saturation :: Maybe Float),
               ("hue",) . showValue <$> (args !? #hue :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_image_random_color_jitter"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_image_random_contrast(symbol)" =
     '[ '("min_factor", AttrReq Float), '("max_factor", AttrReq Float),
        '("data", AttrOpt SymbolHandle)]

_image_random_contrast ::
                       forall args . Fullfilled "_image_random_contrast(symbol)" args =>
                         String ->
                           ArgsHMap "_image_random_contrast(symbol)" args -> IO SymbolHandle
_image_random_contrast name args
  = let scalarArgs
          = catMaybes
              [("min_factor",) . showValue <$>
                 (args !? #min_factor :: Maybe Float),
               ("max_factor",) . showValue <$>
                 (args !? #max_factor :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_image_random_contrast"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_image_random_flip_left_right(symbol)"
     = '[ '("data", AttrOpt SymbolHandle)]

_image_random_flip_left_right ::
                              forall args .
                                Fullfilled "_image_random_flip_left_right(symbol)" args =>
                                String ->
                                  ArgsHMap "_image_random_flip_left_right(symbol)" args ->
                                    IO SymbolHandle
_image_random_flip_left_right name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_image_random_flip_left_right"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_image_random_flip_top_bottom(symbol)"
     = '[ '("data", AttrOpt SymbolHandle)]

_image_random_flip_top_bottom ::
                              forall args .
                                Fullfilled "_image_random_flip_top_bottom(symbol)" args =>
                                String ->
                                  ArgsHMap "_image_random_flip_top_bottom(symbol)" args ->
                                    IO SymbolHandle
_image_random_flip_top_bottom name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_image_random_flip_top_bottom"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_image_random_hue(symbol)" =
     '[ '("min_factor", AttrReq Float), '("max_factor", AttrReq Float),
        '("data", AttrOpt SymbolHandle)]

_image_random_hue ::
                  forall args . Fullfilled "_image_random_hue(symbol)" args =>
                    String ->
                      ArgsHMap "_image_random_hue(symbol)" args -> IO SymbolHandle
_image_random_hue name args
  = let scalarArgs
          = catMaybes
              [("min_factor",) . showValue <$>
                 (args !? #min_factor :: Maybe Float),
               ("max_factor",) . showValue <$>
                 (args !? #max_factor :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_image_random_hue"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_image_random_lighting(symbol)" =
     '[ '("alpha_std", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_image_random_lighting ::
                       forall args . Fullfilled "_image_random_lighting(symbol)" args =>
                         String ->
                           ArgsHMap "_image_random_lighting(symbol)" args -> IO SymbolHandle
_image_random_lighting name args
  = let scalarArgs
          = catMaybes
              [("alpha_std",) . showValue <$>
                 (args !? #alpha_std :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_image_random_lighting"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_image_random_saturation(symbol)" =
     '[ '("min_factor", AttrReq Float), '("max_factor", AttrReq Float),
        '("data", AttrOpt SymbolHandle)]

_image_random_saturation ::
                         forall args . Fullfilled "_image_random_saturation(symbol)" args =>
                           String ->
                             ArgsHMap "_image_random_saturation(symbol)" args -> IO SymbolHandle
_image_random_saturation name args
  = let scalarArgs
          = catMaybes
              [("min_factor",) . showValue <$>
                 (args !? #min_factor :: Maybe Float),
               ("max_factor",) . showValue <$>
                 (args !? #max_factor :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_image_random_saturation"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_image_to_tensor(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

_image_to_tensor ::
                 forall args . Fullfilled "_image_to_tensor(symbol)" args =>
                   String ->
                     ArgsHMap "_image_to_tensor(symbol)" args -> IO SymbolHandle
_image_to_tensor name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_image_to_tensor"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_imdecode(symbol)" =
     '[ '("index", AttrOpt Int), '("x0", AttrOpt Int),
        '("y0", AttrOpt Int), '("x1", AttrOpt Int), '("y1", AttrOpt Int),
        '("c", AttrOpt Int), '("size", AttrOpt Int),
        '("mean", AttrOpt SymbolHandle)]

_imdecode ::
          forall args . Fullfilled "_imdecode(symbol)" args =>
            String -> ArgsHMap "_imdecode(symbol)" args -> IO SymbolHandle
_imdecode name args
  = let scalarArgs
          = catMaybes
              [("index",) . showValue <$> (args !? #index :: Maybe Int),
               ("x0",) . showValue <$> (args !? #x0 :: Maybe Int),
               ("y0",) . showValue <$> (args !? #y0 :: Maybe Int),
               ("x1",) . showValue <$> (args !? #x1 :: Maybe Int),
               ("y1",) . showValue <$> (args !? #y1 :: Maybe Int),
               ("c",) . showValue <$> (args !? #c :: Maybe Int),
               ("size",) . showValue <$> (args !? #size :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("mean",) <$> (args !? #mean :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_imdecode"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_lesser(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_lesser ::
        forall args . Fullfilled "_lesser(symbol)" args =>
          String -> ArgsHMap "_lesser(symbol)" args -> IO SymbolHandle
_lesser name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_lesser"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_lesser_equal(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_lesser_equal ::
              forall args . Fullfilled "_lesser_equal(symbol)" args =>
                String -> ArgsHMap "_lesser_equal(symbol)" args -> IO SymbolHandle
_lesser_equal name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_lesser_equal"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_lesser_equal_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_lesser_equal_scalar ::
                     forall args . Fullfilled "_lesser_equal_scalar(symbol)" args =>
                       String ->
                         ArgsHMap "_lesser_equal_scalar(symbol)" args -> IO SymbolHandle
_lesser_equal_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_lesser_equal_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_lesser_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_lesser_scalar ::
               forall args . Fullfilled "_lesser_scalar(symbol)" args =>
                 String -> ArgsHMap "_lesser_scalar(symbol)" args -> IO SymbolHandle
_lesser_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_lesser_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_linalg_gelqf(symbol)" =
     '[ '("_A", AttrOpt SymbolHandle)]

_linalg_gelqf ::
              forall args . Fullfilled "_linalg_gelqf(symbol)" args =>
                String -> ArgsHMap "_linalg_gelqf(symbol)" args -> IO SymbolHandle
_linalg_gelqf name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("_A",) <$> (args !? #_A :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_gelqf"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_linalg_gemm(symbol)" =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("alpha", AttrOpt Double), '("beta", AttrOpt Double),
        '("axis", AttrOpt Int), '("_A", AttrOpt SymbolHandle),
        '("_B", AttrOpt SymbolHandle), '("_C", AttrOpt SymbolHandle)]

_linalg_gemm ::
             forall args . Fullfilled "_linalg_gemm(symbol)" args =>
               String -> ArgsHMap "_linalg_gemm(symbol)" args -> IO SymbolHandle
_linalg_gemm name args
  = let scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$>
                 (args !? #transpose_a :: Maybe Bool),
               ("transpose_b",) . showValue <$>
                 (args !? #transpose_b :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double),
               ("beta",) . showValue <$> (args !? #beta :: Maybe Double),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("_A",) <$> (args !? #_A :: Maybe SymbolHandle),
               ("_B",) <$> (args !? #_B :: Maybe SymbolHandle),
               ("_C",) <$> (args !? #_C :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_gemm"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_linalg_gemm2(symbol)" =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("alpha", AttrOpt Double), '("axis", AttrOpt Int),
        '("_A", AttrOpt SymbolHandle), '("_B", AttrOpt SymbolHandle)]

_linalg_gemm2 ::
              forall args . Fullfilled "_linalg_gemm2(symbol)" args =>
                String -> ArgsHMap "_linalg_gemm2(symbol)" args -> IO SymbolHandle
_linalg_gemm2 name args
  = let scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$>
                 (args !? #transpose_a :: Maybe Bool),
               ("transpose_b",) . showValue <$>
                 (args !? #transpose_b :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("_A",) <$> (args !? #_A :: Maybe SymbolHandle),
               ("_B",) <$> (args !? #_B :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_gemm2"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_linalg_potrf(symbol)" =
     '[ '("_A", AttrOpt SymbolHandle)]

_linalg_potrf ::
              forall args . Fullfilled "_linalg_potrf(symbol)" args =>
                String -> ArgsHMap "_linalg_potrf(symbol)" args -> IO SymbolHandle
_linalg_potrf name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("_A",) <$> (args !? #_A :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_potrf"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_linalg_potri(symbol)" =
     '[ '("_A", AttrOpt SymbolHandle)]

_linalg_potri ::
              forall args . Fullfilled "_linalg_potri(symbol)" args =>
                String -> ArgsHMap "_linalg_potri(symbol)" args -> IO SymbolHandle
_linalg_potri name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("_A",) <$> (args !? #_A :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_potri"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_linalg_sumlogdiag(symbol)" =
     '[ '("_A", AttrOpt SymbolHandle)]

_linalg_sumlogdiag ::
                   forall args . Fullfilled "_linalg_sumlogdiag(symbol)" args =>
                     String ->
                       ArgsHMap "_linalg_sumlogdiag(symbol)" args -> IO SymbolHandle
_linalg_sumlogdiag name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("_A",) <$> (args !? #_A :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_sumlogdiag"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_linalg_syevd(symbol)" =
     '[ '("_A", AttrOpt SymbolHandle)]

_linalg_syevd ::
              forall args . Fullfilled "_linalg_syevd(symbol)" args =>
                String -> ArgsHMap "_linalg_syevd(symbol)" args -> IO SymbolHandle
_linalg_syevd name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("_A",) <$> (args !? #_A :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_syevd"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_linalg_syrk(symbol)" =
     '[ '("transpose", AttrOpt Bool), '("alpha", AttrOpt Double),
        '("_A", AttrOpt SymbolHandle)]

_linalg_syrk ::
             forall args . Fullfilled "_linalg_syrk(symbol)" args =>
               String -> ArgsHMap "_linalg_syrk(symbol)" args -> IO SymbolHandle
_linalg_syrk name args
  = let scalarArgs
          = catMaybes
              [("transpose",) . showValue <$> (args !? #transpose :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("_A",) <$> (args !? #_A :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_syrk"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_linalg_trmm(symbol)" =
     '[ '("transpose", AttrOpt Bool), '("rightside", AttrOpt Bool),
        '("alpha", AttrOpt Double), '("_A", AttrOpt SymbolHandle),
        '("_B", AttrOpt SymbolHandle)]

_linalg_trmm ::
             forall args . Fullfilled "_linalg_trmm(symbol)" args =>
               String -> ArgsHMap "_linalg_trmm(symbol)" args -> IO SymbolHandle
_linalg_trmm name args
  = let scalarArgs
          = catMaybes
              [("transpose",) . showValue <$> (args !? #transpose :: Maybe Bool),
               ("rightside",) . showValue <$> (args !? #rightside :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("_A",) <$> (args !? #_A :: Maybe SymbolHandle),
               ("_B",) <$> (args !? #_B :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_trmm"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_linalg_trsm(symbol)" =
     '[ '("transpose", AttrOpt Bool), '("rightside", AttrOpt Bool),
        '("alpha", AttrOpt Double), '("_A", AttrOpt SymbolHandle),
        '("_B", AttrOpt SymbolHandle)]

_linalg_trsm ::
             forall args . Fullfilled "_linalg_trsm(symbol)" args =>
               String -> ArgsHMap "_linalg_trsm(symbol)" args -> IO SymbolHandle
_linalg_trsm name args
  = let scalarArgs
          = catMaybes
              [("transpose",) . showValue <$> (args !? #transpose :: Maybe Bool),
               ("rightside",) . showValue <$> (args !? #rightside :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("_A",) <$> (args !? #_A :: Maybe SymbolHandle),
               ("_B",) <$> (args !? #_B :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_linalg_trsm"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_logical_and(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_logical_and ::
             forall args . Fullfilled "_logical_and(symbol)" args =>
               String -> ArgsHMap "_logical_and(symbol)" args -> IO SymbolHandle
_logical_and name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_logical_and"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_logical_and_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_logical_and_scalar ::
                    forall args . Fullfilled "_logical_and_scalar(symbol)" args =>
                      String ->
                        ArgsHMap "_logical_and_scalar(symbol)" args -> IO SymbolHandle
_logical_and_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_logical_and_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_logical_or(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_logical_or ::
            forall args . Fullfilled "_logical_or(symbol)" args =>
              String -> ArgsHMap "_logical_or(symbol)" args -> IO SymbolHandle
_logical_or name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_logical_or"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_logical_or_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_logical_or_scalar ::
                   forall args . Fullfilled "_logical_or_scalar(symbol)" args =>
                     String ->
                       ArgsHMap "_logical_or_scalar(symbol)" args -> IO SymbolHandle
_logical_or_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_logical_or_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_logical_xor(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_logical_xor ::
             forall args . Fullfilled "_logical_xor(symbol)" args =>
               String -> ArgsHMap "_logical_xor(symbol)" args -> IO SymbolHandle
_logical_xor name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_logical_xor"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_logical_xor_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_logical_xor_scalar ::
                    forall args . Fullfilled "_logical_xor_scalar(symbol)" args =>
                      String ->
                        ArgsHMap "_logical_xor_scalar(symbol)" args -> IO SymbolHandle
_logical_xor_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_logical_xor_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_maximum(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_maximum ::
         forall args . Fullfilled "_maximum(symbol)" args =>
           String -> ArgsHMap "_maximum(symbol)" args -> IO SymbolHandle
_maximum name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_maximum"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_maximum_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_maximum_scalar ::
                forall args . Fullfilled "_maximum_scalar(symbol)" args =>
                  String ->
                    ArgsHMap "_maximum_scalar(symbol)" args -> IO SymbolHandle
_maximum_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_maximum_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_minimum(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_minimum ::
         forall args . Fullfilled "_minimum(symbol)" args =>
           String -> ArgsHMap "_minimum(symbol)" args -> IO SymbolHandle
_minimum name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_minimum"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_minimum_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_minimum_scalar ::
                forall args . Fullfilled "_minimum_scalar(symbol)" args =>
                  String ->
                    ArgsHMap "_minimum_scalar(symbol)" args -> IO SymbolHandle
_minimum_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_minimum_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_minus_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_minus_scalar ::
              forall args . Fullfilled "_minus_scalar(symbol)" args =>
                String -> ArgsHMap "_minus_scalar(symbol)" args -> IO SymbolHandle
_minus_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_minus_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_mod(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_mod ::
     forall args . Fullfilled "_mod(symbol)" args =>
       String -> ArgsHMap "_mod(symbol)" args -> IO SymbolHandle
_mod name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_mod"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_mod_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_mod_scalar ::
            forall args . Fullfilled "_mod_scalar(symbol)" args =>
              String -> ArgsHMap "_mod_scalar(symbol)" args -> IO SymbolHandle
_mod_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_mod_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_mul_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_mul_scalar ::
            forall args . Fullfilled "_mul_scalar(symbol)" args =>
              String -> ArgsHMap "_mul_scalar(symbol)" args -> IO SymbolHandle
_mul_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_mul_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_not_equal(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_not_equal ::
           forall args . Fullfilled "_not_equal(symbol)" args =>
             String -> ArgsHMap "_not_equal(symbol)" args -> IO SymbolHandle
_not_equal name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_not_equal"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_not_equal_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_not_equal_scalar ::
                  forall args . Fullfilled "_not_equal_scalar(symbol)" args =>
                    String ->
                      ArgsHMap "_not_equal_scalar(symbol)" args -> IO SymbolHandle
_not_equal_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_not_equal_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_ones(symbol)" =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"]))]

_ones ::
      forall args . Fullfilled "_ones(symbol)" args =>
        String -> ArgsHMap "_ones(symbol)" args -> IO SymbolHandle
_ones name args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["float16", "float32", "float64", "int32", "int64", "int8",
                           "uint8"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_ones"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_plus_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_plus_scalar ::
             forall args . Fullfilled "_plus_scalar(symbol)" args =>
               String -> ArgsHMap "_plus_scalar(symbol)" args -> IO SymbolHandle
_plus_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_plus_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_power(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_power ::
       forall args . Fullfilled "_power(symbol)" args =>
         String -> ArgsHMap "_power(symbol)" args -> IO SymbolHandle
_power name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_power"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_power_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_power_scalar ::
              forall args . Fullfilled "_power_scalar(symbol)" args =>
                String -> ArgsHMap "_power_scalar(symbol)" args -> IO SymbolHandle
_power_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_power_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_random_exponential(symbol)" =
     '[ '("lam", AttrOpt Float), '("shape", AttrOpt [Int]),
        '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

_random_exponential ::
                    forall args . Fullfilled "_random_exponential(symbol)" args =>
                      String ->
                        ArgsHMap "_random_exponential(symbol)" args -> IO SymbolHandle
_random_exponential name args
  = let scalarArgs
          = catMaybes
              [("lam",) . showValue <$> (args !? #lam :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_random_exponential"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_random_gamma(symbol)" =
     '[ '("alpha", AttrOpt Float), '("beta", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

_random_gamma ::
              forall args . Fullfilled "_random_gamma(symbol)" args =>
                String -> ArgsHMap "_random_gamma(symbol)" args -> IO SymbolHandle
_random_gamma name args
  = let scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> (args !? #alpha :: Maybe Float),
               ("beta",) . showValue <$> (args !? #beta :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_random_gamma"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance
     ParameterList "_random_generalized_negative_binomial(symbol)" =
     '[ '("mu", AttrOpt Float), '("alpha", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

_random_generalized_negative_binomial ::
                                      forall args .
                                        Fullfilled "_random_generalized_negative_binomial(symbol)"
                                          args =>
                                        String ->
                                          ArgsHMap "_random_generalized_negative_binomial(symbol)"
                                            args
                                            -> IO SymbolHandle
_random_generalized_negative_binomial name args
  = let scalarArgs
          = catMaybes
              [("mu",) . showValue <$> (args !? #mu :: Maybe Float),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_random_generalized_negative_binomial"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_random_negative_binomial(symbol)" =
     '[ '("k", AttrOpt Int), '("p", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

_random_negative_binomial ::
                          forall args .
                            Fullfilled "_random_negative_binomial(symbol)" args =>
                            String ->
                              ArgsHMap "_random_negative_binomial(symbol)" args ->
                                IO SymbolHandle
_random_negative_binomial name args
  = let scalarArgs
          = catMaybes
              [("k",) . showValue <$> (args !? #k :: Maybe Int),
               ("p",) . showValue <$> (args !? #p :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_random_negative_binomial"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_random_normal(symbol)" =
     '[ '("loc", AttrOpt Float), '("scale", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

_random_normal ::
               forall args . Fullfilled "_random_normal(symbol)" args =>
                 String -> ArgsHMap "_random_normal(symbol)" args -> IO SymbolHandle
_random_normal name args
  = let scalarArgs
          = catMaybes
              [("loc",) . showValue <$> (args !? #loc :: Maybe Float),
               ("scale",) . showValue <$> (args !? #scale :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_random_normal"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_random_poisson(symbol)" =
     '[ '("lam", AttrOpt Float), '("shape", AttrOpt [Int]),
        '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

_random_poisson ::
                forall args . Fullfilled "_random_poisson(symbol)" args =>
                  String ->
                    ArgsHMap "_random_poisson(symbol)" args -> IO SymbolHandle
_random_poisson name args
  = let scalarArgs
          = catMaybes
              [("lam",) . showValue <$> (args !? #lam :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_random_poisson"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_random_uniform(symbol)" =
     '[ '("low", AttrOpt Float), '("high", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

_random_uniform ::
                forall args . Fullfilled "_random_uniform(symbol)" args =>
                  String ->
                    ArgsHMap "_random_uniform(symbol)" args -> IO SymbolHandle
_random_uniform name args
  = let scalarArgs
          = catMaybes
              [("low",) . showValue <$> (args !? #low :: Maybe Float),
               ("high",) . showValue <$> (args !? #high :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_random_uniform"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_ravel_multi_index(symbol)" =
     '[ '("shape", AttrOpt [Int]), '("data", AttrOpt SymbolHandle)]

_ravel_multi_index ::
                   forall args . Fullfilled "_ravel_multi_index(symbol)" args =>
                     String ->
                       ArgsHMap "_ravel_multi_index(symbol)" args -> IO SymbolHandle
_ravel_multi_index name args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_ravel_multi_index"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_rdiv_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_rdiv_scalar ::
             forall args . Fullfilled "_rdiv_scalar(symbol)" args =>
               String -> ArgsHMap "_rdiv_scalar(symbol)" args -> IO SymbolHandle
_rdiv_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_rdiv_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_rminus_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_rminus_scalar ::
               forall args . Fullfilled "_rminus_scalar(symbol)" args =>
                 String -> ArgsHMap "_rminus_scalar(symbol)" args -> IO SymbolHandle
_rminus_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_rminus_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_rmod_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_rmod_scalar ::
             forall args . Fullfilled "_rmod_scalar(symbol)" args =>
               String -> ArgsHMap "_rmod_scalar(symbol)" args -> IO SymbolHandle
_rmod_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_rmod_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_rnn_param_concat(symbol)" =
     '[ '("num_args", AttrReq Int), '("dim", AttrOpt Int),
        '("data", AttrOpt [SymbolHandle])]

_rnn_param_concat ::
                  forall args . Fullfilled "_rnn_param_concat(symbol)" args =>
                    String ->
                      ArgsHMap "_rnn_param_concat(symbol)" args -> IO SymbolHandle
_rnn_param_concat name args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("dim",) . showValue <$> (args !? #dim :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
        array = fromMaybe [] (args !? #data :: Maybe [SymbolHandle])
      in
      do op <- nnGetOpHandle "_rnn_param_concat"
         sym <- if hasKey args #num_args then
                  mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys scalarvals
                  else
                  mxSymbolCreateAtomicSymbol (fromOpHandle op)
                    ("num_args" : scalarkeys)
                    (showValue (length array) : scalarvals)
         mxSymbolCompose sym name Nothing array
         return sym

type instance ParameterList "_rpower_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_rpower_scalar ::
               forall args . Fullfilled "_rpower_scalar(symbol)" args =>
                 String -> ArgsHMap "_rpower_scalar(symbol)" args -> IO SymbolHandle
_rpower_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_rpower_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_sample_exponential(symbol)" =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("lam", AttrOpt SymbolHandle)]

_sample_exponential ::
                    forall args . Fullfilled "_sample_exponential(symbol)" args =>
                      String ->
                        ArgsHMap "_sample_exponential(symbol)" args -> IO SymbolHandle
_sample_exponential name args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("lam",) <$> (args !? #lam :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_exponential"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_sample_gamma(symbol)" =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("alpha", AttrOpt SymbolHandle), '("beta", AttrOpt SymbolHandle)]

_sample_gamma ::
              forall args . Fullfilled "_sample_gamma(symbol)" args =>
                String -> ArgsHMap "_sample_gamma(symbol)" args -> IO SymbolHandle
_sample_gamma name args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("alpha",) <$> (args !? #alpha :: Maybe SymbolHandle),
               ("beta",) <$> (args !? #beta :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_gamma"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance
     ParameterList "_sample_generalized_negative_binomial(symbol)" =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("mu", AttrOpt SymbolHandle), '("alpha", AttrOpt SymbolHandle)]

_sample_generalized_negative_binomial ::
                                      forall args .
                                        Fullfilled "_sample_generalized_negative_binomial(symbol)"
                                          args =>
                                        String ->
                                          ArgsHMap "_sample_generalized_negative_binomial(symbol)"
                                            args
                                            -> IO SymbolHandle
_sample_generalized_negative_binomial name args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("mu",) <$> (args !? #mu :: Maybe SymbolHandle),
               ("alpha",) <$> (args !? #alpha :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_generalized_negative_binomial"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_sample_multinomial(symbol)" =
     '[ '("shape", AttrOpt [Int]), '("get_prob", AttrOpt Bool),
        '("dtype",
          AttrOpt
            (EnumType '["float16", "float32", "float64", "int32", "uint8"])),
        '("data", AttrOpt SymbolHandle)]

_sample_multinomial ::
                    forall args . Fullfilled "_sample_multinomial(symbol)" args =>
                      String ->
                        ArgsHMap "_sample_multinomial(symbol)" args -> IO SymbolHandle
_sample_multinomial name args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("get_prob",) . showValue <$> (args !? #get_prob :: Maybe Bool),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType '["float16", "float32", "float64", "int32", "uint8"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_multinomial"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_sample_negative_binomial(symbol)" =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("k", AttrOpt SymbolHandle), '("p", AttrOpt SymbolHandle)]

_sample_negative_binomial ::
                          forall args .
                            Fullfilled "_sample_negative_binomial(symbol)" args =>
                            String ->
                              ArgsHMap "_sample_negative_binomial(symbol)" args ->
                                IO SymbolHandle
_sample_negative_binomial name args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("k",) <$> (args !? #k :: Maybe SymbolHandle),
               ("p",) <$> (args !? #p :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_negative_binomial"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_sample_normal(symbol)" =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("mu", AttrOpt SymbolHandle), '("sigma", AttrOpt SymbolHandle)]

_sample_normal ::
               forall args . Fullfilled "_sample_normal(symbol)" args =>
                 String -> ArgsHMap "_sample_normal(symbol)" args -> IO SymbolHandle
_sample_normal name args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("mu",) <$> (args !? #mu :: Maybe SymbolHandle),
               ("sigma",) <$> (args !? #sigma :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_normal"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_sample_poisson(symbol)" =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("lam", AttrOpt SymbolHandle)]

_sample_poisson ::
                forall args . Fullfilled "_sample_poisson(symbol)" args =>
                  String ->
                    ArgsHMap "_sample_poisson(symbol)" args -> IO SymbolHandle
_sample_poisson name args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("lam",) <$> (args !? #lam :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_poisson"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_sample_uniform(symbol)" =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("low", AttrOpt SymbolHandle), '("high", AttrOpt SymbolHandle)]

_sample_uniform ::
                forall args . Fullfilled "_sample_uniform(symbol)" args =>
                  String ->
                    ArgsHMap "_sample_uniform(symbol)" args -> IO SymbolHandle
_sample_uniform name args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("low",) <$> (args !? #low :: Maybe SymbolHandle),
               ("high",) <$> (args !? #high :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_uniform"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_sample_unique_zipfian(symbol)" =
     '[ '("range_max", AttrReq Int), '("shape", AttrOpt [Int])]

_sample_unique_zipfian ::
                       forall args . Fullfilled "_sample_unique_zipfian(symbol)" args =>
                         String ->
                           ArgsHMap "_sample_unique_zipfian(symbol)" args -> IO SymbolHandle
_sample_unique_zipfian name args
  = let scalarArgs
          = catMaybes
              [("range_max",) . showValue <$> (args !? #range_max :: Maybe Int),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sample_unique_zipfian"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_scatter_elemwise_div(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

_scatter_elemwise_div ::
                      forall args . Fullfilled "_scatter_elemwise_div(symbol)" args =>
                        String ->
                          ArgsHMap "_scatter_elemwise_div(symbol)" args -> IO SymbolHandle
_scatter_elemwise_div name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_scatter_elemwise_div"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_scatter_minus_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_scatter_minus_scalar ::
                      forall args . Fullfilled "_scatter_minus_scalar(symbol)" args =>
                        String ->
                          ArgsHMap "_scatter_minus_scalar(symbol)" args -> IO SymbolHandle
_scatter_minus_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_scatter_minus_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_scatter_plus_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

_scatter_plus_scalar ::
                     forall args . Fullfilled "_scatter_plus_scalar(symbol)" args =>
                       String ->
                         ArgsHMap "_scatter_plus_scalar(symbol)" args -> IO SymbolHandle
_scatter_plus_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_scatter_plus_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_scatter_set_nd(symbol)" =
     '[ '("shape", AttrReq [Int]), '("lhs", AttrOpt SymbolHandle),
        '("rhs", AttrOpt SymbolHandle), '("indices", AttrOpt SymbolHandle)]

_scatter_set_nd ::
                forall args . Fullfilled "_scatter_set_nd(symbol)" args =>
                  String ->
                    ArgsHMap "_scatter_set_nd(symbol)" args -> IO SymbolHandle
_scatter_set_nd name args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle),
               ("indices",) <$> (args !? #indices :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_scatter_set_nd"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_set_value(symbol)" =
     '[ '("src", AttrOpt Float)]

_set_value ::
           forall args . Fullfilled "_set_value(symbol)" args =>
             String -> ArgsHMap "_set_value(symbol)" args -> IO SymbolHandle
_set_value name args
  = let scalarArgs
          = catMaybes
              [("src",) . showValue <$> (args !? #src :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_set_value"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_shuffle(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

_shuffle ::
         forall args . Fullfilled "_shuffle(symbol)" args =>
           String -> ArgsHMap "_shuffle(symbol)" args -> IO SymbolHandle
_shuffle name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_shuffle"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_slice_assign(symbol)" =
     '[ '("begin", AttrReq [Int]), '("end", AttrReq [Int]),
        '("step", AttrOpt [Int]), '("lhs", AttrOpt SymbolHandle),
        '("rhs", AttrOpt SymbolHandle)]

_slice_assign ::
              forall args . Fullfilled "_slice_assign(symbol)" args =>
                String -> ArgsHMap "_slice_assign(symbol)" args -> IO SymbolHandle
_slice_assign name args
  = let scalarArgs
          = catMaybes
              [("begin",) . showValue <$> (args !? #begin :: Maybe [Int]),
               ("end",) . showValue <$> (args !? #end :: Maybe [Int]),
               ("step",) . showValue <$> (args !? #step :: Maybe [Int])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_slice_assign"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_slice_assign_scalar(symbol)" =
     '[ '("scalar", AttrOpt Float), '("begin", AttrReq [Int]),
        '("end", AttrReq [Int]), '("step", AttrOpt [Int]),
        '("data", AttrOpt SymbolHandle)]

_slice_assign_scalar ::
                     forall args . Fullfilled "_slice_assign_scalar(symbol)" args =>
                       String ->
                         ArgsHMap "_slice_assign_scalar(symbol)" args -> IO SymbolHandle
_slice_assign_scalar name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float),
               ("begin",) . showValue <$> (args !? #begin :: Maybe [Int]),
               ("end",) . showValue <$> (args !? #end :: Maybe [Int]),
               ("step",) . showValue <$> (args !? #step :: Maybe [Int])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_slice_assign_scalar"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_sparse_adagrad_update(symbol)" =
     '[ '("lr", AttrReq Float), '("epsilon", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("weight", AttrOpt SymbolHandle), '("grad", AttrOpt SymbolHandle),
        '("history", AttrOpt SymbolHandle)]

_sparse_adagrad_update ::
                       forall args . Fullfilled "_sparse_adagrad_update(symbol)" args =>
                         String ->
                           ArgsHMap "_sparse_adagrad_update(symbol)" args -> IO SymbolHandle
_sparse_adagrad_update name args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("epsilon",) . showValue <$> (args !? #epsilon :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe SymbolHandle),
               ("grad",) <$> (args !? #grad :: Maybe SymbolHandle),
               ("history",) <$> (args !? #history :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sparse_adagrad_update"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_sparse_retain(symbol)" =
     '[ '("data", AttrOpt SymbolHandle),
        '("indices", AttrOpt SymbolHandle)]

_sparse_retain ::
               forall args . Fullfilled "_sparse_retain(symbol)" args =>
                 String -> ArgsHMap "_sparse_retain(symbol)" args -> IO SymbolHandle
_sparse_retain name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("indices",) <$> (args !? #indices :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_sparse_retain"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_square_sum(symbol)" =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt SymbolHandle)]

_square_sum ::
            forall args . Fullfilled "_square_sum(symbol)" args =>
              String -> ArgsHMap "_square_sum(symbol)" args -> IO SymbolHandle
_square_sum name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_square_sum"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_unravel_index(symbol)" =
     '[ '("shape", AttrOpt [Int]), '("data", AttrOpt SymbolHandle)]

_unravel_index ::
               forall args . Fullfilled "_unravel_index(symbol)" args =>
                 String -> ArgsHMap "_unravel_index(symbol)" args -> IO SymbolHandle
_unravel_index name args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_unravel_index"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_zeros(symbol)" =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt String),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"]))]

_zeros ::
       forall args . Fullfilled "_zeros(symbol)" args =>
         String -> ArgsHMap "_zeros(symbol)" args -> IO SymbolHandle
_zeros name args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe String),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["float16", "float32", "float64", "int32", "int64", "int8",
                           "uint8"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "_zeros"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "abs(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

abs ::
    forall args . Fullfilled "abs(symbol)" args =>
      String -> ArgsHMap "abs(symbol)" args -> IO SymbolHandle
abs name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "abs"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "adam_update(symbol)" =
     '[ '("lr", AttrReq Float), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("lazy_update", AttrOpt Bool),
        '("weight", AttrOpt SymbolHandle), '("grad", AttrOpt SymbolHandle),
        '("mean", AttrOpt SymbolHandle), '("var", AttrOpt SymbolHandle)]

adam_update ::
            forall args . Fullfilled "adam_update(symbol)" args =>
              String -> ArgsHMap "adam_update(symbol)" args -> IO SymbolHandle
adam_update name args
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
                 (args !? #clip_gradient :: Maybe Float),
               ("lazy_update",) . showValue <$>
                 (args !? #lazy_update :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe SymbolHandle),
               ("grad",) <$> (args !? #grad :: Maybe SymbolHandle),
               ("mean",) <$> (args !? #mean :: Maybe SymbolHandle),
               ("var",) <$> (args !? #var :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "adam_update"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "add_n(symbol)" =
     '[ '("args", AttrOpt [SymbolHandle])]

add_n ::
      forall args . Fullfilled "add_n(symbol)" args =>
        String -> ArgsHMap "add_n(symbol)" args -> IO SymbolHandle
add_n name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
        array = fromMaybe [] (args !? #args :: Maybe [SymbolHandle])
      in
      do op <- nnGetOpHandle "add_n"
         sym <- if hasKey args #num_args then
                  mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys scalarvals
                  else
                  mxSymbolCreateAtomicSymbol (fromOpHandle op)
                    ("num_args" : scalarkeys)
                    (showValue (length array) : scalarvals)
         mxSymbolCompose sym name Nothing array
         return sym

type instance ParameterList "arccos(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

arccos ::
       forall args . Fullfilled "arccos(symbol)" args =>
         String -> ArgsHMap "arccos(symbol)" args -> IO SymbolHandle
arccos name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "arccos"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "arccosh(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

arccosh ::
        forall args . Fullfilled "arccosh(symbol)" args =>
          String -> ArgsHMap "arccosh(symbol)" args -> IO SymbolHandle
arccosh name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "arccosh"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "arcsin(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

arcsin ::
       forall args . Fullfilled "arcsin(symbol)" args =>
         String -> ArgsHMap "arcsin(symbol)" args -> IO SymbolHandle
arcsin name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "arcsin"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "arcsinh(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

arcsinh ::
        forall args . Fullfilled "arcsinh(symbol)" args =>
          String -> ArgsHMap "arcsinh(symbol)" args -> IO SymbolHandle
arcsinh name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "arcsinh"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "arctan(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

arctan ::
       forall args . Fullfilled "arctan(symbol)" args =>
         String -> ArgsHMap "arctan(symbol)" args -> IO SymbolHandle
arctan name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "arctan"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "arctanh(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

arctanh ::
        forall args . Fullfilled "arctanh(symbol)" args =>
          String -> ArgsHMap "arctanh(symbol)" args -> IO SymbolHandle
arctanh name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "arctanh"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "argmax(symbol)" =
     '[ '("axis", AttrOpt (Maybe Int)), '("keepdims", AttrOpt Bool),
        '("data", AttrOpt SymbolHandle)]

argmax ::
       forall args . Fullfilled "argmax(symbol)" args =>
         String -> ArgsHMap "argmax(symbol)" args -> IO SymbolHandle
argmax name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "argmax"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "argmax_channel(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

argmax_channel ::
               forall args . Fullfilled "argmax_channel(symbol)" args =>
                 String -> ArgsHMap "argmax_channel(symbol)" args -> IO SymbolHandle
argmax_channel name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "argmax_channel"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "argmin(symbol)" =
     '[ '("axis", AttrOpt (Maybe Int)), '("keepdims", AttrOpt Bool),
        '("data", AttrOpt SymbolHandle)]

argmin ::
       forall args . Fullfilled "argmin(symbol)" args =>
         String -> ArgsHMap "argmin(symbol)" args -> IO SymbolHandle
argmin name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "argmin"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "argsort(symbol)" =
     '[ '("axis", AttrOpt (Maybe Int)), '("is_ascend", AttrOpt Bool),
        '("dtype",
          AttrOpt
            (EnumType '["float16", "float32", "float64", "int32", "uint8"])),
        '("data", AttrOpt SymbolHandle)]

argsort ::
        forall args . Fullfilled "argsort(symbol)" args =>
          String -> ArgsHMap "argsort(symbol)" args -> IO SymbolHandle
argsort name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("is_ascend",) . showValue <$> (args !? #is_ascend :: Maybe Bool),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType '["float16", "float32", "float64", "int32", "uint8"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "argsort"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "batch_dot(symbol)" =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("forward_stype",
          AttrOpt (Maybe (EnumType '["csr", "default", "row_sparse"]))),
        '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

batch_dot ::
          forall args . Fullfilled "batch_dot(symbol)" args =>
            String -> ArgsHMap "batch_dot(symbol)" args -> IO SymbolHandle
batch_dot name args
  = let scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$>
                 (args !? #transpose_a :: Maybe Bool),
               ("transpose_b",) . showValue <$>
                 (args !? #transpose_b :: Maybe Bool),
               ("forward_stype",) . showValue <$>
                 (args !? #forward_stype ::
                    Maybe (Maybe (EnumType '["csr", "default", "row_sparse"])))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "batch_dot"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "batch_take(symbol)" =
     '[ '("a", AttrOpt SymbolHandle),
        '("indices", AttrOpt SymbolHandle)]

batch_take ::
           forall args . Fullfilled "batch_take(symbol)" args =>
             String -> ArgsHMap "batch_take(symbol)" args -> IO SymbolHandle
batch_take name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("a",) <$> (args !? #a :: Maybe SymbolHandle),
               ("indices",) <$> (args !? #indices :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "batch_take"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "broadcast_add(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

broadcast_add ::
              forall args . Fullfilled "broadcast_add(symbol)" args =>
                String -> ArgsHMap "broadcast_add(symbol)" args -> IO SymbolHandle
broadcast_add name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_add"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "broadcast_axis(symbol)" =
     '[ '("axis", AttrOpt [Int]), '("size", AttrOpt [Int]),
        '("data", AttrOpt SymbolHandle)]

broadcast_axis ::
               forall args . Fullfilled "broadcast_axis(symbol)" args =>
                 String -> ArgsHMap "broadcast_axis(symbol)" args -> IO SymbolHandle
broadcast_axis name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int]),
               ("size",) . showValue <$> (args !? #size :: Maybe [Int])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_axis"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "broadcast_div(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

broadcast_div ::
              forall args . Fullfilled "broadcast_div(symbol)" args =>
                String -> ArgsHMap "broadcast_div(symbol)" args -> IO SymbolHandle
broadcast_div name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_div"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "broadcast_equal(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

broadcast_equal ::
                forall args . Fullfilled "broadcast_equal(symbol)" args =>
                  String ->
                    ArgsHMap "broadcast_equal(symbol)" args -> IO SymbolHandle
broadcast_equal name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_equal"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "broadcast_greater(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

broadcast_greater ::
                  forall args . Fullfilled "broadcast_greater(symbol)" args =>
                    String ->
                      ArgsHMap "broadcast_greater(symbol)" args -> IO SymbolHandle
broadcast_greater name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_greater"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "broadcast_greater_equal(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

broadcast_greater_equal ::
                        forall args . Fullfilled "broadcast_greater_equal(symbol)" args =>
                          String ->
                            ArgsHMap "broadcast_greater_equal(symbol)" args -> IO SymbolHandle
broadcast_greater_equal name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_greater_equal"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "broadcast_hypot(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

broadcast_hypot ::
                forall args . Fullfilled "broadcast_hypot(symbol)" args =>
                  String ->
                    ArgsHMap "broadcast_hypot(symbol)" args -> IO SymbolHandle
broadcast_hypot name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_hypot"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "broadcast_lesser(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

broadcast_lesser ::
                 forall args . Fullfilled "broadcast_lesser(symbol)" args =>
                   String ->
                     ArgsHMap "broadcast_lesser(symbol)" args -> IO SymbolHandle
broadcast_lesser name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_lesser"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "broadcast_lesser_equal(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

broadcast_lesser_equal ::
                       forall args . Fullfilled "broadcast_lesser_equal(symbol)" args =>
                         String ->
                           ArgsHMap "broadcast_lesser_equal(symbol)" args -> IO SymbolHandle
broadcast_lesser_equal name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_lesser_equal"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "broadcast_like(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

broadcast_like ::
               forall args . Fullfilled "broadcast_like(symbol)" args =>
                 String -> ArgsHMap "broadcast_like(symbol)" args -> IO SymbolHandle
broadcast_like name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_like"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "broadcast_logical_and(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

broadcast_logical_and ::
                      forall args . Fullfilled "broadcast_logical_and(symbol)" args =>
                        String ->
                          ArgsHMap "broadcast_logical_and(symbol)" args -> IO SymbolHandle
broadcast_logical_and name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_logical_and"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "broadcast_logical_or(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

broadcast_logical_or ::
                     forall args . Fullfilled "broadcast_logical_or(symbol)" args =>
                       String ->
                         ArgsHMap "broadcast_logical_or(symbol)" args -> IO SymbolHandle
broadcast_logical_or name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_logical_or"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "broadcast_logical_xor(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

broadcast_logical_xor ::
                      forall args . Fullfilled "broadcast_logical_xor(symbol)" args =>
                        String ->
                          ArgsHMap "broadcast_logical_xor(symbol)" args -> IO SymbolHandle
broadcast_logical_xor name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_logical_xor"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "broadcast_maximum(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

broadcast_maximum ::
                  forall args . Fullfilled "broadcast_maximum(symbol)" args =>
                    String ->
                      ArgsHMap "broadcast_maximum(symbol)" args -> IO SymbolHandle
broadcast_maximum name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_maximum"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "broadcast_minimum(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

broadcast_minimum ::
                  forall args . Fullfilled "broadcast_minimum(symbol)" args =>
                    String ->
                      ArgsHMap "broadcast_minimum(symbol)" args -> IO SymbolHandle
broadcast_minimum name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_minimum"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "broadcast_mod(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

broadcast_mod ::
              forall args . Fullfilled "broadcast_mod(symbol)" args =>
                String -> ArgsHMap "broadcast_mod(symbol)" args -> IO SymbolHandle
broadcast_mod name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_mod"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "broadcast_mul(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

broadcast_mul ::
              forall args . Fullfilled "broadcast_mul(symbol)" args =>
                String -> ArgsHMap "broadcast_mul(symbol)" args -> IO SymbolHandle
broadcast_mul name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_mul"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "broadcast_not_equal(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

broadcast_not_equal ::
                    forall args . Fullfilled "broadcast_not_equal(symbol)" args =>
                      String ->
                        ArgsHMap "broadcast_not_equal(symbol)" args -> IO SymbolHandle
broadcast_not_equal name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_not_equal"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "broadcast_power(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

broadcast_power ::
                forall args . Fullfilled "broadcast_power(symbol)" args =>
                  String ->
                    ArgsHMap "broadcast_power(symbol)" args -> IO SymbolHandle
broadcast_power name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_power"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "broadcast_sub(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

broadcast_sub ::
              forall args . Fullfilled "broadcast_sub(symbol)" args =>
                String -> ArgsHMap "broadcast_sub(symbol)" args -> IO SymbolHandle
broadcast_sub name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_sub"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "broadcast_to(symbol)" =
     '[ '("shape", AttrOpt [Int]), '("data", AttrOpt SymbolHandle)]

broadcast_to ::
             forall args . Fullfilled "broadcast_to(symbol)" args =>
               String -> ArgsHMap "broadcast_to(symbol)" args -> IO SymbolHandle
broadcast_to name args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "broadcast_to"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "cast_storage(symbol)" =
     '[ '("stype",
          AttrReq (EnumType '["csr", "default", "row_sparse"])),
        '("data", AttrOpt SymbolHandle)]

cast_storage ::
             forall args . Fullfilled "cast_storage(symbol)" args =>
               String -> ArgsHMap "cast_storage(symbol)" args -> IO SymbolHandle
cast_storage name args
  = let scalarArgs
          = catMaybes
              [("stype",) . showValue <$>
                 (args !? #stype ::
                    Maybe (EnumType '["csr", "default", "row_sparse"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "cast_storage"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "cbrt(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

cbrt ::
     forall args . Fullfilled "cbrt(symbol)" args =>
       String -> ArgsHMap "cbrt(symbol)" args -> IO SymbolHandle
cbrt name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "cbrt"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "ceil(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

ceil ::
     forall args . Fullfilled "ceil(symbol)" args =>
       String -> ArgsHMap "ceil(symbol)" args -> IO SymbolHandle
ceil name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "ceil"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "clip(symbol)" =
     '[ '("a_min", AttrReq Float), '("a_max", AttrReq Float),
        '("data", AttrOpt SymbolHandle)]

clip ::
     forall args . Fullfilled "clip(symbol)" args =>
       String -> ArgsHMap "clip(symbol)" args -> IO SymbolHandle
clip name args
  = let scalarArgs
          = catMaybes
              [("a_min",) . showValue <$> (args !? #a_min :: Maybe Float),
               ("a_max",) . showValue <$> (args !? #a_max :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "clip"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "cos(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

cos ::
    forall args . Fullfilled "cos(symbol)" args =>
      String -> ArgsHMap "cos(symbol)" args -> IO SymbolHandle
cos name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "cos"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "cosh(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

cosh ::
     forall args . Fullfilled "cosh(symbol)" args =>
       String -> ArgsHMap "cosh(symbol)" args -> IO SymbolHandle
cosh name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "cosh"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "degrees(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

degrees ::
        forall args . Fullfilled "degrees(symbol)" args =>
          String -> ArgsHMap "degrees(symbol)" args -> IO SymbolHandle
degrees name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "degrees"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "depth_to_space(symbol)" =
     '[ '("block_size", AttrReq Int), '("data", AttrOpt SymbolHandle)]

depth_to_space ::
               forall args . Fullfilled "depth_to_space(symbol)" args =>
                 String -> ArgsHMap "depth_to_space(symbol)" args -> IO SymbolHandle
depth_to_space name args
  = let scalarArgs
          = catMaybes
              [("block_size",) . showValue <$>
                 (args !? #block_size :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "depth_to_space"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "diag(symbol)" =
     '[ '("k", AttrOpt (Maybe Int)), '("data", AttrOpt SymbolHandle)]

diag ::
     forall args . Fullfilled "diag(symbol)" args =>
       String -> ArgsHMap "diag(symbol)" args -> IO SymbolHandle
diag name args
  = let scalarArgs
          = catMaybes
              [("k",) . showValue <$> (args !? #k :: Maybe (Maybe Int))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "diag"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "dot(symbol)" =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("forward_stype",
          AttrOpt (Maybe (EnumType '["csr", "default", "row_sparse"]))),
        '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

dot ::
    forall args . Fullfilled "dot(symbol)" args =>
      String -> ArgsHMap "dot(symbol)" args -> IO SymbolHandle
dot name args
  = let scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$>
                 (args !? #transpose_a :: Maybe Bool),
               ("transpose_b",) . showValue <$>
                 (args !? #transpose_b :: Maybe Bool),
               ("forward_stype",) . showValue <$>
                 (args !? #forward_stype ::
                    Maybe (Maybe (EnumType '["csr", "default", "row_sparse"])))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "dot"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "elemwise_add(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

elemwise_add ::
             forall args . Fullfilled "elemwise_add(symbol)" args =>
               String -> ArgsHMap "elemwise_add(symbol)" args -> IO SymbolHandle
elemwise_add name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "elemwise_add"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "elemwise_div(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

elemwise_div ::
             forall args . Fullfilled "elemwise_div(symbol)" args =>
               String -> ArgsHMap "elemwise_div(symbol)" args -> IO SymbolHandle
elemwise_div name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "elemwise_div"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "elemwise_mul(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

elemwise_mul ::
             forall args . Fullfilled "elemwise_mul(symbol)" args =>
               String -> ArgsHMap "elemwise_mul(symbol)" args -> IO SymbolHandle
elemwise_mul name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "elemwise_mul"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "elemwise_sub(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

elemwise_sub ::
             forall args . Fullfilled "elemwise_sub(symbol)" args =>
               String -> ArgsHMap "elemwise_sub(symbol)" args -> IO SymbolHandle
elemwise_sub name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "elemwise_sub"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "exp(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

exp ::
    forall args . Fullfilled "exp(symbol)" args =>
      String -> ArgsHMap "exp(symbol)" args -> IO SymbolHandle
exp name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "exp"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "expand_dims(symbol)" =
     '[ '("axis", AttrReq Int), '("data", AttrOpt SymbolHandle)]

expand_dims ::
            forall args . Fullfilled "expand_dims(symbol)" args =>
              String -> ArgsHMap "expand_dims(symbol)" args -> IO SymbolHandle
expand_dims name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "expand_dims"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "expm1(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

expm1 ::
      forall args . Fullfilled "expm1(symbol)" args =>
        String -> ArgsHMap "expm1(symbol)" args -> IO SymbolHandle
expm1 name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "expm1"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "fix(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

fix ::
    forall args . Fullfilled "fix(symbol)" args =>
      String -> ArgsHMap "fix(symbol)" args -> IO SymbolHandle
fix name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "fix"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "floor(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

floor ::
      forall args . Fullfilled "floor(symbol)" args =>
        String -> ArgsHMap "floor(symbol)" args -> IO SymbolHandle
floor name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "floor"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "ftml_update(symbol)" =
     '[ '("lr", AttrReq Float), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Double),
        '("t", AttrReq Int), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float), '("clip_grad", AttrOpt Float),
        '("weight", AttrOpt SymbolHandle), '("grad", AttrOpt SymbolHandle),
        '("d", AttrOpt SymbolHandle), '("v", AttrOpt SymbolHandle),
        '("z", AttrOpt SymbolHandle)]

ftml_update ::
            forall args . Fullfilled "ftml_update(symbol)" args =>
              String -> ArgsHMap "ftml_update(symbol)" args -> IO SymbolHandle
ftml_update name args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("beta1",) . showValue <$> (args !? #beta1 :: Maybe Float),
               ("beta2",) . showValue <$> (args !? #beta2 :: Maybe Float),
               ("epsilon",) . showValue <$> (args !? #epsilon :: Maybe Double),
               ("t",) . showValue <$> (args !? #t :: Maybe Int),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_grad",) . showValue <$> (args !? #clip_grad :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe SymbolHandle),
               ("grad",) <$> (args !? #grad :: Maybe SymbolHandle),
               ("d",) <$> (args !? #d :: Maybe SymbolHandle),
               ("v",) <$> (args !? #v :: Maybe SymbolHandle),
               ("z",) <$> (args !? #z :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "ftml_update"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "ftrl_update(symbol)" =
     '[ '("lr", AttrReq Float), '("lamda1", AttrOpt Float),
        '("beta", AttrOpt Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("weight", AttrOpt SymbolHandle), '("grad", AttrOpt SymbolHandle),
        '("z", AttrOpt SymbolHandle), '("n", AttrOpt SymbolHandle)]

ftrl_update ::
            forall args . Fullfilled "ftrl_update(symbol)" args =>
              String -> ArgsHMap "ftrl_update(symbol)" args -> IO SymbolHandle
ftrl_update name args
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
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe SymbolHandle),
               ("grad",) <$> (args !? #grad :: Maybe SymbolHandle),
               ("z",) <$> (args !? #z :: Maybe SymbolHandle),
               ("n",) <$> (args !? #n :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "ftrl_update"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "gamma(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

gamma ::
      forall args . Fullfilled "gamma(symbol)" args =>
        String -> ArgsHMap "gamma(symbol)" args -> IO SymbolHandle
gamma name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "gamma"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "gammaln(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

gammaln ::
        forall args . Fullfilled "gammaln(symbol)" args =>
          String -> ArgsHMap "gammaln(symbol)" args -> IO SymbolHandle
gammaln name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "gammaln"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "gather_nd(symbol)" =
     '[ '("data", AttrOpt SymbolHandle),
        '("indices", AttrOpt SymbolHandle)]

gather_nd ::
          forall args . Fullfilled "gather_nd(symbol)" args =>
            String -> ArgsHMap "gather_nd(symbol)" args -> IO SymbolHandle
gather_nd name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("indices",) <$> (args !? #indices :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "gather_nd"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "hard_sigmoid(symbol)" =
     '[ '("alpha", AttrOpt Float), '("beta", AttrOpt Float),
        '("data", AttrOpt SymbolHandle)]

hard_sigmoid ::
             forall args . Fullfilled "hard_sigmoid(symbol)" args =>
               String -> ArgsHMap "hard_sigmoid(symbol)" args -> IO SymbolHandle
hard_sigmoid name args
  = let scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> (args !? #alpha :: Maybe Float),
               ("beta",) . showValue <$> (args !? #beta :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "hard_sigmoid"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "khatri_rao(symbol)" =
     '[ '("args", AttrOpt [SymbolHandle])]

khatri_rao ::
           forall args . Fullfilled "khatri_rao(symbol)" args =>
             String -> ArgsHMap "khatri_rao(symbol)" args -> IO SymbolHandle
khatri_rao name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
        array = fromMaybe [] (args !? #args :: Maybe [SymbolHandle])
      in
      do op <- nnGetOpHandle "khatri_rao"
         sym <- if hasKey args #num_args then
                  mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys scalarvals
                  else
                  mxSymbolCreateAtomicSymbol (fromOpHandle op)
                    ("num_args" : scalarkeys)
                    (showValue (length array) : scalarvals)
         mxSymbolCompose sym name Nothing array
         return sym

type instance ParameterList "log(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

log ::
    forall args . Fullfilled "log(symbol)" args =>
      String -> ArgsHMap "log(symbol)" args -> IO SymbolHandle
log name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "log"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "log10(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

log10 ::
      forall args . Fullfilled "log10(symbol)" args =>
        String -> ArgsHMap "log10(symbol)" args -> IO SymbolHandle
log10 name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "log10"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "log1p(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

log1p ::
      forall args . Fullfilled "log1p(symbol)" args =>
        String -> ArgsHMap "log1p(symbol)" args -> IO SymbolHandle
log1p name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "log1p"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "log2(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

log2 ::
     forall args . Fullfilled "log2(symbol)" args =>
       String -> ArgsHMap "log2(symbol)" args -> IO SymbolHandle
log2 name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "log2"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "log_softmax(symbol)" =
     '[ '("axis", AttrOpt Int),
        '("temperature", AttrOpt (Maybe Double)),
        '("data", AttrOpt SymbolHandle)]

log_softmax ::
            forall args . Fullfilled "log_softmax(symbol)" args =>
              String -> ArgsHMap "log_softmax(symbol)" args -> IO SymbolHandle
log_softmax name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("temperature",) . showValue <$>
                 (args !? #temperature :: Maybe (Maybe Double))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "log_softmax"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "logical_not(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

logical_not ::
            forall args . Fullfilled "logical_not(symbol)" args =>
              String -> ArgsHMap "logical_not(symbol)" args -> IO SymbolHandle
logical_not name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "logical_not"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "make_loss(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

make_loss ::
          forall args . Fullfilled "make_loss(symbol)" args =>
            String -> ArgsHMap "make_loss(symbol)" args -> IO SymbolHandle
make_loss name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "make_loss"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "max(symbol)" =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt SymbolHandle)]

max ::
    forall args . Fullfilled "max(symbol)" args =>
      String -> ArgsHMap "max(symbol)" args -> IO SymbolHandle
max name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "max"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "mean(symbol)" =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt SymbolHandle)]

mean ::
     forall args . Fullfilled "mean(symbol)" args =>
       String -> ArgsHMap "mean(symbol)" args -> IO SymbolHandle
mean name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "mean"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "min(symbol)" =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt SymbolHandle)]

min ::
    forall args . Fullfilled "min(symbol)" args =>
      String -> ArgsHMap "min(symbol)" args -> IO SymbolHandle
min name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "min"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "mp_sgd_mom_update(symbol)" =
     '[ '("lr", AttrReq Float), '("momentum", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("lazy_update", AttrOpt Bool),
        '("weight", AttrOpt SymbolHandle), '("grad", AttrOpt SymbolHandle),
        '("mom", AttrOpt SymbolHandle),
        '("weight32", AttrOpt SymbolHandle)]

mp_sgd_mom_update ::
                  forall args . Fullfilled "mp_sgd_mom_update(symbol)" args =>
                    String ->
                      ArgsHMap "mp_sgd_mom_update(symbol)" args -> IO SymbolHandle
mp_sgd_mom_update name args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("lazy_update",) . showValue <$>
                 (args !? #lazy_update :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe SymbolHandle),
               ("grad",) <$> (args !? #grad :: Maybe SymbolHandle),
               ("mom",) <$> (args !? #mom :: Maybe SymbolHandle),
               ("weight32",) <$> (args !? #weight32 :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "mp_sgd_mom_update"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "mp_sgd_update(symbol)" =
     '[ '("lr", AttrReq Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("lazy_update", AttrOpt Bool),
        '("weight", AttrOpt SymbolHandle), '("grad", AttrOpt SymbolHandle),
        '("weight32", AttrOpt SymbolHandle)]

mp_sgd_update ::
              forall args . Fullfilled "mp_sgd_update(symbol)" args =>
                String -> ArgsHMap "mp_sgd_update(symbol)" args -> IO SymbolHandle
mp_sgd_update name args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("lazy_update",) . showValue <$>
                 (args !? #lazy_update :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe SymbolHandle),
               ("grad",) <$> (args !? #grad :: Maybe SymbolHandle),
               ("weight32",) <$> (args !? #weight32 :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "mp_sgd_update"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "nanprod(symbol)" =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt SymbolHandle)]

nanprod ::
        forall args . Fullfilled "nanprod(symbol)" args =>
          String -> ArgsHMap "nanprod(symbol)" args -> IO SymbolHandle
nanprod name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "nanprod"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "nansum(symbol)" =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt SymbolHandle)]

nansum ::
       forall args . Fullfilled "nansum(symbol)" args =>
         String -> ArgsHMap "nansum(symbol)" args -> IO SymbolHandle
nansum name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "nansum"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "negative(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

negative ::
         forall args . Fullfilled "negative(symbol)" args =>
           String -> ArgsHMap "negative(symbol)" args -> IO SymbolHandle
negative name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "negative"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "norm(symbol)" =
     '[ '("ord", AttrOpt Int), '("axis", AttrOpt (Maybe [Int])),
        '("keepdims", AttrOpt Bool), '("data", AttrOpt SymbolHandle)]

norm ::
     forall args . Fullfilled "norm(symbol)" args =>
       String -> ArgsHMap "norm(symbol)" args -> IO SymbolHandle
norm name args
  = let scalarArgs
          = catMaybes
              [("ord",) . showValue <$> (args !? #ord :: Maybe Int),
               ("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "norm"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "one_hot(symbol)" =
     '[ '("depth", AttrReq Int), '("on_value", AttrOpt Double),
        '("off_value", AttrOpt Double),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"])),
        '("indices", AttrOpt SymbolHandle)]

one_hot ::
        forall args . Fullfilled "one_hot(symbol)" args =>
          String -> ArgsHMap "one_hot(symbol)" args -> IO SymbolHandle
one_hot name args
  = let scalarArgs
          = catMaybes
              [("depth",) . showValue <$> (args !? #depth :: Maybe Int),
               ("on_value",) . showValue <$> (args !? #on_value :: Maybe Double),
               ("off_value",) . showValue <$>
                 (args !? #off_value :: Maybe Double),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["float16", "float32", "float64", "int32", "int64", "int8",
                           "uint8"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("indices",) <$> (args !? #indices :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "one_hot"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "ones_like(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

ones_like ::
          forall args . Fullfilled "ones_like(symbol)" args =>
            String -> ArgsHMap "ones_like(symbol)" args -> IO SymbolHandle
ones_like name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "ones_like"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "pick(symbol)" =
     '[ '("axis", AttrOpt (Maybe Int)), '("keepdims", AttrOpt Bool),
        '("mode", AttrOpt (EnumType '["clip", "wrap"])),
        '("data", AttrOpt SymbolHandle), '("index", AttrOpt SymbolHandle)]

pick ::
     forall args . Fullfilled "pick(symbol)" args =>
       String -> ArgsHMap "pick(symbol)" args -> IO SymbolHandle
pick name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("mode",) . showValue <$>
                 (args !? #mode :: Maybe (EnumType '["clip", "wrap"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("index",) <$> (args !? #index :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "pick"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "prod(symbol)" =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt SymbolHandle)]

prod ::
     forall args . Fullfilled "prod(symbol)" args =>
       String -> ArgsHMap "prod(symbol)" args -> IO SymbolHandle
prod name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "prod"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "radians(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

radians ::
        forall args . Fullfilled "radians(symbol)" args =>
          String -> ArgsHMap "radians(symbol)" args -> IO SymbolHandle
radians name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "radians"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "rcbrt(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

rcbrt ::
      forall args . Fullfilled "rcbrt(symbol)" args =>
        String -> ArgsHMap "rcbrt(symbol)" args -> IO SymbolHandle
rcbrt name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "rcbrt"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "reciprocal(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

reciprocal ::
           forall args . Fullfilled "reciprocal(symbol)" args =>
             String -> ArgsHMap "reciprocal(symbol)" args -> IO SymbolHandle
reciprocal name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "reciprocal"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "relu(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

relu ::
     forall args . Fullfilled "relu(symbol)" args =>
       String -> ArgsHMap "relu(symbol)" args -> IO SymbolHandle
relu name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "relu"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "repeat(symbol)" =
     '[ '("repeats", AttrReq Int), '("axis", AttrOpt (Maybe Int)),
        '("data", AttrOpt SymbolHandle)]

repeat ::
       forall args . Fullfilled "repeat(symbol)" args =>
         String -> ArgsHMap "repeat(symbol)" args -> IO SymbolHandle
repeat name args
  = let scalarArgs
          = catMaybes
              [("repeats",) . showValue <$> (args !? #repeats :: Maybe Int),
               ("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "repeat"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "reshape_like(symbol)" =
     '[ '("lhs", AttrOpt SymbolHandle), '("rhs", AttrOpt SymbolHandle)]

reshape_like ::
             forall args . Fullfilled "reshape_like(symbol)" args =>
               String -> ArgsHMap "reshape_like(symbol)" args -> IO SymbolHandle
reshape_like name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe SymbolHandle),
               ("rhs",) <$> (args !? #rhs :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "reshape_like"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "reverse(symbol)" =
     '[ '("axis", AttrReq [Int]), '("data", AttrOpt SymbolHandle)]

reverse ::
        forall args . Fullfilled "reverse(symbol)" args =>
          String -> ArgsHMap "reverse(symbol)" args -> IO SymbolHandle
reverse name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "reverse"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "rint(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

rint ::
     forall args . Fullfilled "rint(symbol)" args =>
       String -> ArgsHMap "rint(symbol)" args -> IO SymbolHandle
rint name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "rint"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "rmsprop_update(symbol)" =
     '[ '("lr", AttrReq Float), '("gamma1", AttrOpt Float),
        '("epsilon", AttrOpt Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("clip_weights", AttrOpt Float),
        '("weight", AttrOpt SymbolHandle), '("grad", AttrOpt SymbolHandle),
        '("n", AttrOpt SymbolHandle)]

rmsprop_update ::
               forall args . Fullfilled "rmsprop_update(symbol)" args =>
                 String -> ArgsHMap "rmsprop_update(symbol)" args -> IO SymbolHandle
rmsprop_update name args
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
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe SymbolHandle),
               ("grad",) <$> (args !? #grad :: Maybe SymbolHandle),
               ("n",) <$> (args !? #n :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "rmsprop_update"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "rmspropalex_update(symbol)" =
     '[ '("lr", AttrReq Float), '("gamma1", AttrOpt Float),
        '("gamma2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("clip_weights", AttrOpt Float),
        '("weight", AttrOpt SymbolHandle), '("grad", AttrOpt SymbolHandle),
        '("n", AttrOpt SymbolHandle), '("g", AttrOpt SymbolHandle),
        '("delta", AttrOpt SymbolHandle)]

rmspropalex_update ::
                   forall args . Fullfilled "rmspropalex_update(symbol)" args =>
                     String ->
                       ArgsHMap "rmspropalex_update(symbol)" args -> IO SymbolHandle
rmspropalex_update name args
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
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe SymbolHandle),
               ("grad",) <$> (args !? #grad :: Maybe SymbolHandle),
               ("n",) <$> (args !? #n :: Maybe SymbolHandle),
               ("g",) <$> (args !? #g :: Maybe SymbolHandle),
               ("delta",) <$> (args !? #delta :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "rmspropalex_update"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "round(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

round ::
      forall args . Fullfilled "round(symbol)" args =>
        String -> ArgsHMap "round(symbol)" args -> IO SymbolHandle
round name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "round"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "rsqrt(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

rsqrt ::
      forall args . Fullfilled "rsqrt(symbol)" args =>
        String -> ArgsHMap "rsqrt(symbol)" args -> IO SymbolHandle
rsqrt name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "rsqrt"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "scatter_nd(symbol)" =
     '[ '("shape", AttrReq [Int]), '("data", AttrOpt SymbolHandle),
        '("indices", AttrOpt SymbolHandle)]

scatter_nd ::
           forall args . Fullfilled "scatter_nd(symbol)" args =>
             String -> ArgsHMap "scatter_nd(symbol)" args -> IO SymbolHandle
scatter_nd name args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("indices",) <$> (args !? #indices :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "scatter_nd"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "sgd_mom_update(symbol)" =
     '[ '("lr", AttrReq Float), '("momentum", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("lazy_update", AttrOpt Bool),
        '("weight", AttrOpt SymbolHandle), '("grad", AttrOpt SymbolHandle),
        '("mom", AttrOpt SymbolHandle)]

sgd_mom_update ::
               forall args . Fullfilled "sgd_mom_update(symbol)" args =>
                 String -> ArgsHMap "sgd_mom_update(symbol)" args -> IO SymbolHandle
sgd_mom_update name args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("lazy_update",) . showValue <$>
                 (args !? #lazy_update :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe SymbolHandle),
               ("grad",) <$> (args !? #grad :: Maybe SymbolHandle),
               ("mom",) <$> (args !? #mom :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sgd_mom_update"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "sgd_update(symbol)" =
     '[ '("lr", AttrReq Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("lazy_update", AttrOpt Bool),
        '("weight", AttrOpt SymbolHandle), '("grad", AttrOpt SymbolHandle)]

sgd_update ::
           forall args . Fullfilled "sgd_update(symbol)" args =>
             String -> ArgsHMap "sgd_update(symbol)" args -> IO SymbolHandle
sgd_update name args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("lazy_update",) . showValue <$>
                 (args !? #lazy_update :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe SymbolHandle),
               ("grad",) <$> (args !? #grad :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sgd_update"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "shape_array(symbol)" =
     '[ '("lhs_begin", AttrOpt (Maybe Int)),
        '("lhs_end", AttrOpt (Maybe Int)),
        '("rhs_begin", AttrOpt (Maybe Int)),
        '("rhs_end", AttrOpt (Maybe Int)), '("data", AttrOpt SymbolHandle)]

shape_array ::
            forall args . Fullfilled "shape_array(symbol)" args =>
              String -> ArgsHMap "shape_array(symbol)" args -> IO SymbolHandle
shape_array name args
  = let scalarArgs
          = catMaybes
              [("lhs_begin",) . showValue <$>
                 (args !? #lhs_begin :: Maybe (Maybe Int)),
               ("lhs_end",) . showValue <$>
                 (args !? #lhs_end :: Maybe (Maybe Int)),
               ("rhs_begin",) . showValue <$>
                 (args !? #rhs_begin :: Maybe (Maybe Int)),
               ("rhs_end",) . showValue <$>
                 (args !? #rhs_end :: Maybe (Maybe Int))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "shape_array"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "sigmoid(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

sigmoid ::
        forall args . Fullfilled "sigmoid(symbol)" args =>
          String -> ArgsHMap "sigmoid(symbol)" args -> IO SymbolHandle
sigmoid name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sigmoid"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "sign(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

sign ::
     forall args . Fullfilled "sign(symbol)" args =>
       String -> ArgsHMap "sign(symbol)" args -> IO SymbolHandle
sign name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sign"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "signsgd_update(symbol)" =
     '[ '("lr", AttrReq Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("weight", AttrOpt SymbolHandle), '("grad", AttrOpt SymbolHandle)]

signsgd_update ::
               forall args . Fullfilled "signsgd_update(symbol)" args =>
                 String -> ArgsHMap "signsgd_update(symbol)" args -> IO SymbolHandle
signsgd_update name args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe SymbolHandle),
               ("grad",) <$> (args !? #grad :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "signsgd_update"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "signum_update(symbol)" =
     '[ '("lr", AttrReq Float), '("momentum", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("wd_lh", AttrOpt Float),
        '("weight", AttrOpt SymbolHandle), '("grad", AttrOpt SymbolHandle),
        '("mom", AttrOpt SymbolHandle)]

signum_update ::
              forall args . Fullfilled "signum_update(symbol)" args =>
                String -> ArgsHMap "signum_update(symbol)" args -> IO SymbolHandle
signum_update name args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("wd_lh",) . showValue <$> (args !? #wd_lh :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("weight",) <$> (args !? #weight :: Maybe SymbolHandle),
               ("grad",) <$> (args !? #grad :: Maybe SymbolHandle),
               ("mom",) <$> (args !? #mom :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "signum_update"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "sin(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

sin ::
    forall args . Fullfilled "sin(symbol)" args =>
      String -> ArgsHMap "sin(symbol)" args -> IO SymbolHandle
sin name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sin"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "sinh(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

sinh ::
     forall args . Fullfilled "sinh(symbol)" args =>
       String -> ArgsHMap "sinh(symbol)" args -> IO SymbolHandle
sinh name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sinh"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "size_array(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

size_array ::
           forall args . Fullfilled "size_array(symbol)" args =>
             String -> ArgsHMap "size_array(symbol)" args -> IO SymbolHandle
size_array name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "size_array"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "slice(symbol)" =
     '[ '("begin", AttrReq [Int]), '("end", AttrReq [Int]),
        '("step", AttrOpt [Int]), '("data", AttrOpt SymbolHandle)]

slice ::
      forall args . Fullfilled "slice(symbol)" args =>
        String -> ArgsHMap "slice(symbol)" args -> IO SymbolHandle
slice name args
  = let scalarArgs
          = catMaybes
              [("begin",) . showValue <$> (args !? #begin :: Maybe [Int]),
               ("end",) . showValue <$> (args !? #end :: Maybe [Int]),
               ("step",) . showValue <$> (args !? #step :: Maybe [Int])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "slice"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "slice_axis(symbol)" =
     '[ '("axis", AttrReq Int), '("begin", AttrReq Int),
        '("end", AttrReq (Maybe Int)), '("data", AttrOpt SymbolHandle)]

slice_axis ::
           forall args . Fullfilled "slice_axis(symbol)" args =>
             String -> ArgsHMap "slice_axis(symbol)" args -> IO SymbolHandle
slice_axis name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("begin",) . showValue <$> (args !? #begin :: Maybe Int),
               ("end",) . showValue <$> (args !? #end :: Maybe (Maybe Int))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "slice_axis"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "slice_like(symbol)" =
     '[ '("axes", AttrOpt [Int]), '("data", AttrOpt SymbolHandle),
        '("shape_like", AttrOpt SymbolHandle)]

slice_like ::
           forall args . Fullfilled "slice_like(symbol)" args =>
             String -> ArgsHMap "slice_like(symbol)" args -> IO SymbolHandle
slice_like name args
  = let scalarArgs
          = catMaybes
              [("axes",) . showValue <$> (args !? #axes :: Maybe [Int])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("shape_like",) <$> (args !? #shape_like :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "slice_like"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "smooth_l1(symbol)" =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt SymbolHandle)]

smooth_l1 ::
          forall args . Fullfilled "smooth_l1(symbol)" args =>
            String -> ArgsHMap "smooth_l1(symbol)" args -> IO SymbolHandle
smooth_l1 name args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "smooth_l1"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "softmax(symbol)" =
     '[ '("axis", AttrOpt Int),
        '("temperature", AttrOpt (Maybe Double)),
        '("data", AttrOpt SymbolHandle)]

softmax ::
        forall args . Fullfilled "softmax(symbol)" args =>
          String -> ArgsHMap "softmax(symbol)" args -> IO SymbolHandle
softmax name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("temperature",) . showValue <$>
                 (args !? #temperature :: Maybe (Maybe Double))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "softmax"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "softmax_cross_entropy(symbol)" =
     '[ '("data", AttrOpt SymbolHandle),
        '("label", AttrOpt SymbolHandle)]

softmax_cross_entropy ::
                      forall args . Fullfilled "softmax_cross_entropy(symbol)" args =>
                        String ->
                          ArgsHMap "softmax_cross_entropy(symbol)" args -> IO SymbolHandle
softmax_cross_entropy name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe SymbolHandle),
               ("label",) <$> (args !? #label :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "softmax_cross_entropy"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "softsign(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

softsign ::
         forall args . Fullfilled "softsign(symbol)" args =>
           String -> ArgsHMap "softsign(symbol)" args -> IO SymbolHandle
softsign name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "softsign"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "sort(symbol)" =
     '[ '("axis", AttrOpt (Maybe Int)), '("is_ascend", AttrOpt Bool),
        '("data", AttrOpt SymbolHandle)]

sort ::
     forall args . Fullfilled "sort(symbol)" args =>
       String -> ArgsHMap "sort(symbol)" args -> IO SymbolHandle
sort name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("is_ascend",) . showValue <$> (args !? #is_ascend :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sort"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "space_to_depth(symbol)" =
     '[ '("block_size", AttrReq Int), '("data", AttrOpt SymbolHandle)]

space_to_depth ::
               forall args . Fullfilled "space_to_depth(symbol)" args =>
                 String -> ArgsHMap "space_to_depth(symbol)" args -> IO SymbolHandle
space_to_depth name args
  = let scalarArgs
          = catMaybes
              [("block_size",) . showValue <$>
                 (args !? #block_size :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "space_to_depth"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "sqrt(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

sqrt ::
     forall args . Fullfilled "sqrt(symbol)" args =>
       String -> ArgsHMap "sqrt(symbol)" args -> IO SymbolHandle
sqrt name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sqrt"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "square(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

square ::
       forall args . Fullfilled "square(symbol)" args =>
         String -> ArgsHMap "square(symbol)" args -> IO SymbolHandle
square name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "square"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "squeeze(symbol)" =
     '[ '("axis", AttrOpt (Maybe [Int])),
        '("data", AttrOpt [SymbolHandle])]

squeeze ::
        forall args . Fullfilled "squeeze(symbol)" args =>
          String -> ArgsHMap "squeeze(symbol)" args -> IO SymbolHandle
squeeze name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
        array = fromMaybe [] (args !? #data :: Maybe [SymbolHandle])
      in
      do op <- nnGetOpHandle "squeeze"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name Nothing array
         return sym

type instance ParameterList "stack(symbol)" =
     '[ '("axis", AttrOpt Int), '("num_args", AttrReq Int),
        '("data", AttrOpt [SymbolHandle])]

stack ::
      forall args . Fullfilled "stack(symbol)" args =>
        String -> ArgsHMap "stack(symbol)" args -> IO SymbolHandle
stack name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("num_args",) . showValue <$> (args !? #num_args :: Maybe Int)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs = catMaybes []
        (tensorkeys, tensorvals) = unzip tensorArgs
        array = fromMaybe [] (args !? #data :: Maybe [SymbolHandle])
      in
      do op <- nnGetOpHandle "stack"
         sym <- if hasKey args #num_args then
                  mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys scalarvals
                  else
                  mxSymbolCreateAtomicSymbol (fromOpHandle op)
                    ("num_args" : scalarkeys)
                    (showValue (length array) : scalarvals)
         mxSymbolCompose sym name Nothing array
         return sym

type instance ParameterList "sum(symbol)" =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt SymbolHandle)]

sum ::
    forall args . Fullfilled "sum(symbol)" args =>
      String -> ArgsHMap "sum(symbol)" args -> IO SymbolHandle
sum name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "sum"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "take(symbol)" =
     '[ '("axis", AttrOpt Int),
        '("mode", AttrOpt (EnumType '["clip", "raise", "wrap"])),
        '("a", AttrOpt SymbolHandle), '("indices", AttrOpt SymbolHandle)]

take ::
     forall args . Fullfilled "take(symbol)" args =>
       String -> ArgsHMap "take(symbol)" args -> IO SymbolHandle
take name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("mode",) . showValue <$>
                 (args !? #mode :: Maybe (EnumType '["clip", "raise", "wrap"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("a",) <$> (args !? #a :: Maybe SymbolHandle),
               ("indices",) <$> (args !? #indices :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "take"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "tan(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

tan ::
    forall args . Fullfilled "tan(symbol)" args =>
      String -> ArgsHMap "tan(symbol)" args -> IO SymbolHandle
tan name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "tan"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "tanh(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

tanh ::
     forall args . Fullfilled "tanh(symbol)" args =>
       String -> ArgsHMap "tanh(symbol)" args -> IO SymbolHandle
tanh name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "tanh"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "tile(symbol)" =
     '[ '("reps", AttrReq [Int]), '("data", AttrOpt SymbolHandle)]

tile ::
     forall args . Fullfilled "tile(symbol)" args =>
       String -> ArgsHMap "tile(symbol)" args -> IO SymbolHandle
tile name args
  = let scalarArgs
          = catMaybes
              [("reps",) . showValue <$> (args !? #reps :: Maybe [Int])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "tile"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "topk(symbol)" =
     '[ '("axis", AttrOpt (Maybe Int)), '("k", AttrOpt Int),
        '("ret_typ",
          AttrOpt (EnumType '["both", "indices", "mask", "value"])),
        '("is_ascend", AttrOpt Bool),
        '("dtype",
          AttrOpt
            (EnumType '["float16", "float32", "float64", "int32", "uint8"])),
        '("data", AttrOpt SymbolHandle)]

topk ::
     forall args . Fullfilled "topk(symbol)" args =>
       String -> ArgsHMap "topk(symbol)" args -> IO SymbolHandle
topk name args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("k",) . showValue <$> (args !? #k :: Maybe Int),
               ("ret_typ",) . showValue <$>
                 (args !? #ret_typ ::
                    Maybe (EnumType '["both", "indices", "mask", "value"])),
               ("is_ascend",) . showValue <$> (args !? #is_ascend :: Maybe Bool),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType '["float16", "float32", "float64", "int32", "uint8"]))]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "topk"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "transpose(symbol)" =
     '[ '("axes", AttrOpt [Int]), '("data", AttrOpt SymbolHandle)]

transpose ::
          forall args . Fullfilled "transpose(symbol)" args =>
            String -> ArgsHMap "transpose(symbol)" args -> IO SymbolHandle
transpose name args
  = let scalarArgs
          = catMaybes
              [("axes",) . showValue <$> (args !? #axes :: Maybe [Int])]
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "transpose"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "trunc(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

trunc ::
      forall args . Fullfilled "trunc(symbol)" args =>
        String -> ArgsHMap "trunc(symbol)" args -> IO SymbolHandle
trunc name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "trunc"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "_where(symbol)" =
     '[ '("condition", AttrOpt SymbolHandle),
        '("x", AttrOpt SymbolHandle), '("y", AttrOpt SymbolHandle)]

_where ::
       forall args . Fullfilled "_where(symbol)" args =>
         String -> ArgsHMap "_where(symbol)" args -> IO SymbolHandle
_where name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes
              [("condition",) <$> (args !? #condition :: Maybe SymbolHandle),
               ("x",) <$> (args !? #x :: Maybe SymbolHandle),
               ("y",) <$> (args !? #y :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "where"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym

type instance ParameterList "zeros_like(symbol)" =
     '[ '("data", AttrOpt SymbolHandle)]

zeros_like ::
           forall args . Fullfilled "zeros_like(symbol)" args =>
             String -> ArgsHMap "zeros_like(symbol)" args -> IO SymbolHandle
zeros_like name args
  = let scalarArgs = catMaybes []
        (scalarkeys, scalarvals) = unzip scalarArgs
        tensorArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe SymbolHandle)]
        (tensorkeys, tensorvals) = unzip tensorArgs
      in
      do op <- nnGetOpHandle "zeros_like"
         sym <- mxSymbolCreateAtomicSymbol (fromOpHandle op) scalarkeys
                  scalarvals
         mxSymbolCompose sym name (Just tensorkeys) tensorvals
         return sym