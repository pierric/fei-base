module MXNet.Base.Operators.Tensor where
import RIO
import RIO.List
import MXNet.Base.Raw
import MXNet.Base.Spec.Operator
import MXNet.Base.Spec.HMap
import MXNet.Base.Tensor.Class
import MXNet.Base.Types (DType)
import Data.Maybe (catMaybes, fromMaybe)

type instance ParameterList "_Activation" '(t, u) =
     '[ '("act_type",
          AttrReq
            (EnumType '["relu", "sigmoid", "softrelu", "softsign", "tanh"])),
        '("data", AttrOpt (t u))]

_Activation ::
            forall a t u .
              (Tensor t, Fullfilled "_Activation" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "_Activation" '(t, u) a -> TensorApply (t u)
_Activation args
  = let scalarArgs
          = catMaybes
              [("act_type",) . showValue <$>
                 (args !? #act_type ::
                    Maybe
                      (EnumType '["relu", "sigmoid", "softrelu", "softsign", "tanh"]))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "Activation" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_BatchNorm" '(t, u) =
     '[ '("eps", AttrOpt Double), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool), '("axis", AttrOpt Int),
        '("cudnn_off", AttrOpt Bool),
        '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)),
        '("data", AttrOpt (t u)), '("gamma", AttrOpt (t u)),
        '("beta", AttrOpt (t u)), '("moving_mean", AttrOpt (t u)),
        '("moving_var", AttrOpt (t u))]

_BatchNorm ::
           forall a t u .
             (Tensor t, Fullfilled "_BatchNorm" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "_BatchNorm" '(t, u) a -> TensorApply (t u)
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
               ("cudnn_off",) . showValue <$> (args !? #cudnn_off :: Maybe Bool),
               ("min_calib_range",) . showValue <$>
                 (args !? #min_calib_range :: Maybe (Maybe Float)),
               ("max_calib_range",) . showValue <$>
                 (args !? #max_calib_range :: Maybe (Maybe Float))]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("gamma",) . toRaw <$> (args !? #gamma :: Maybe (t u)),
               ("beta",) . toRaw <$> (args !? #beta :: Maybe (t u)),
               ("moving_mean",) . toRaw <$> (args !? #moving_mean :: Maybe (t u)),
               ("moving_var",) . toRaw <$> (args !? #moving_var :: Maybe (t u))]
      in
      applyRaw "BatchNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_BatchNorm_v1" '(t, u) =
     '[ '("eps", AttrOpt Float), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool), '("data", AttrOpt (t u)),
        '("gamma", AttrOpt (t u)), '("beta", AttrOpt (t u))]

_BatchNorm_v1 ::
              forall a t u .
                (Tensor t, Fullfilled "_BatchNorm_v1" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "_BatchNorm_v1" '(t, u) a -> TensorApply (t u)
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
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("gamma",) . toRaw <$> (args !? #gamma :: Maybe (t u)),
               ("beta",) . toRaw <$> (args !? #beta :: Maybe (t u))]
      in
      applyRaw "BatchNorm_v1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_BilinearSampler" '(t, u) =
     '[ '("cudnn_off", AttrOpt (Maybe Bool)), '("data", AttrOpt (t u)),
        '("grid", AttrOpt (t u))]

_BilinearSampler ::
                 forall a t u .
                   (Tensor t, Fullfilled "_BilinearSampler" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "_BilinearSampler" '(t, u) a -> TensorApply (t u)
_BilinearSampler args
  = let scalarArgs
          = catMaybes
              [("cudnn_off",) . showValue <$>
                 (args !? #cudnn_off :: Maybe (Maybe Bool))]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("grid",) . toRaw <$> (args !? #grid :: Maybe (t u))]
      in
      applyRaw "BilinearSampler" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_BlockGrad" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_BlockGrad ::
           forall a t u .
             (Tensor t, Fullfilled "_BlockGrad" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "_BlockGrad" '(t, u) a -> TensorApply (t u)
_BlockGrad args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "BlockGrad" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_CTCLoss" '(t, u) =
     '[ '("use_data_lengths", AttrOpt Bool),
        '("use_label_lengths", AttrOpt Bool),
        '("blank_label", AttrOpt (EnumType '["first", "last"])),
        '("data", AttrOpt (t u)), '("label", AttrOpt (t u)),
        '("data_lengths", AttrOpt (t u)),
        '("label_lengths", AttrOpt (t u))]

_CTCLoss ::
         forall a t u .
           (Tensor t, Fullfilled "_CTCLoss" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "_CTCLoss" '(t, u) a -> TensorApply (t u)
_CTCLoss args
  = let scalarArgs
          = catMaybes
              [("use_data_lengths",) . showValue <$>
                 (args !? #use_data_lengths :: Maybe Bool),
               ("use_label_lengths",) . showValue <$>
                 (args !? #use_label_lengths :: Maybe Bool),
               ("blank_label",) . showValue <$>
                 (args !? #blank_label :: Maybe (EnumType '["first", "last"]))]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("label",) . toRaw <$> (args !? #label :: Maybe (t u)),
               ("data_lengths",) . toRaw <$>
                 (args !? #data_lengths :: Maybe (t u)),
               ("label_lengths",) . toRaw <$>
                 (args !? #label_lengths :: Maybe (t u))]
      in
      applyRaw "CTCLoss" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_Cast" '(t, u) =
     '[ '("dtype",
          AttrReq
            (EnumType
               '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                 "int64", "int8", "uint8"])),
        '("data", AttrOpt (t u))]

_Cast ::
      forall a t v u .
        (Tensor t, Fullfilled "_Cast" '(t, u) a, HasCallStack, DType v,
         DType u) =>
        ArgsHMap "_Cast" '(t, u) a -> TensorApply (t v)
_Cast args
  = let scalarArgs
          = catMaybes
              [("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                           "int64", "int8", "uint8"]))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "Cast" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_Concat" '(t, u) =
     '[ '("num_args", AttrReq Int), '("dim", AttrOpt Int),
        '("data", AttrOpt [t u])]

_Concat ::
        forall a t u .
          (Tensor t, Fullfilled "_Concat" '(t, u) a, HasCallStack,
           DType u) =>
          ArgsHMap "_Concat" '(t, u) a -> TensorApply (t u)
_Concat args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("dim",) . showValue <$> (args !? #dim :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "Concat" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_Convolution" '(t, u) =
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
        '("data", AttrOpt (t u)), '("weight", AttrOpt (t u)),
        '("bias", AttrOpt (t u))]

_Convolution ::
             forall a t u .
               (Tensor t, Fullfilled "_Convolution" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "_Convolution" '(t, u) a -> TensorApply (t u)
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
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("bias",) . toRaw <$> (args !? #bias :: Maybe (t u))]
      in
      applyRaw "Convolution" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_Convolution_v1" '(t, u) =
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
        '("data", AttrOpt (t u)), '("weight", AttrOpt (t u)),
        '("bias", AttrOpt (t u))]

_Convolution_v1 ::
                forall a t u .
                  (Tensor t, Fullfilled "_Convolution_v1" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "_Convolution_v1" '(t, u) a -> TensorApply (t u)
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
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("bias",) . toRaw <$> (args !? #bias :: Maybe (t u))]
      in
      applyRaw "Convolution_v1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_Correlation" '(t, u) =
     '[ '("kernel_size", AttrOpt Int),
        '("max_displacement", AttrOpt Int), '("stride1", AttrOpt Int),
        '("stride2", AttrOpt Int), '("pad_size", AttrOpt Int),
        '("is_multiply", AttrOpt Bool), '("data1", AttrOpt (t u)),
        '("data2", AttrOpt (t u))]

_Correlation ::
             forall a t u .
               (Tensor t, Fullfilled "_Correlation" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "_Correlation" '(t, u) a -> TensorApply (t u)
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
        tensorKeyArgs
          = catMaybes
              [("data1",) . toRaw <$> (args !? #data1 :: Maybe (t u)),
               ("data2",) . toRaw <$> (args !? #data2 :: Maybe (t u))]
      in
      applyRaw "Correlation" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_Crop" '(t, u) =
     '[ '("num_args", AttrReq Int), '("offset", AttrOpt [Int]),
        '("h_w", AttrOpt [Int]), '("center_crop", AttrOpt Bool),
        '("data", AttrOpt [t u])]

_Crop ::
      forall a t u .
        (Tensor t, Fullfilled "_Crop" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "_Crop" '(t, u) a -> TensorApply (t u)
_Crop args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("offset",) . showValue <$> (args !? #offset :: Maybe [Int]),
               ("h_w",) . showValue <$> (args !? #h_w :: Maybe [Int]),
               ("center_crop",) . showValue <$>
                 (args !? #center_crop :: Maybe Bool)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "Crop" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_CuDNNBatchNorm" '(t, u) =
     '[ '("eps", AttrOpt Double), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool), '("axis", AttrOpt Int),
        '("cudnn_off", AttrOpt Bool),
        '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)),
        '("data", AttrOpt (t u)), '("gamma", AttrOpt (t u)),
        '("beta", AttrOpt (t u)), '("moving_mean", AttrOpt (t u)),
        '("moving_var", AttrOpt (t u))]

_CuDNNBatchNorm ::
                forall a t u .
                  (Tensor t, Fullfilled "_CuDNNBatchNorm" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "_CuDNNBatchNorm" '(t, u) a -> TensorApply (t u)
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
               ("cudnn_off",) . showValue <$> (args !? #cudnn_off :: Maybe Bool),
               ("min_calib_range",) . showValue <$>
                 (args !? #min_calib_range :: Maybe (Maybe Float)),
               ("max_calib_range",) . showValue <$>
                 (args !? #max_calib_range :: Maybe (Maybe Float))]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("gamma",) . toRaw <$> (args !? #gamma :: Maybe (t u)),
               ("beta",) . toRaw <$> (args !? #beta :: Maybe (t u)),
               ("moving_mean",) . toRaw <$> (args !? #moving_mean :: Maybe (t u)),
               ("moving_var",) . toRaw <$> (args !? #moving_var :: Maybe (t u))]
      in
      applyRaw "CuDNNBatchNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_Custom" '(t, u) =
     '[ '("op_type", AttrOpt Text), '("data", AttrOpt [t u])]

_Custom ::
        forall a t u .
          (Tensor t, Fullfilled "_Custom" '(t, u) a, HasCallStack, DType u,
           PopKey (ArgOf "_Custom" '(t, u)) a "data",
           Dump (PopResult (ArgOf "_Custom" '(t, u)) a "data")) =>
          ArgsHMap "_Custom" '(t, u) a -> TensorApply (t u)
_Custom args
  = let scalarArgs = dump (pop args #data)
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "Custom" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_Deconvolution" '(t, u) =
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
        '("data", AttrOpt (t u)), '("weight", AttrOpt (t u)),
        '("bias", AttrOpt (t u))]

_Deconvolution ::
               forall a t u .
                 (Tensor t, Fullfilled "_Deconvolution" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "_Deconvolution" '(t, u) a -> TensorApply (t u)
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
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("bias",) . toRaw <$> (args !? #bias :: Maybe (t u))]
      in
      applyRaw "Deconvolution" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_Dropout" '(t, u) =
     '[ '("p", AttrOpt Float),
        '("mode", AttrOpt (EnumType '["always", "training"])),
        '("axes", AttrOpt [Int]), '("cudnn_off", AttrOpt (Maybe Bool)),
        '("data", AttrOpt (t u))]

_Dropout ::
         forall a t u .
           (Tensor t, Fullfilled "_Dropout" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "_Dropout" '(t, u) a -> TensorApply (t u)
_Dropout args
  = let scalarArgs
          = catMaybes
              [("p",) . showValue <$> (args !? #p :: Maybe Float),
               ("mode",) . showValue <$>
                 (args !? #mode :: Maybe (EnumType '["always", "training"])),
               ("axes",) . showValue <$> (args !? #axes :: Maybe [Int]),
               ("cudnn_off",) . showValue <$>
                 (args !? #cudnn_off :: Maybe (Maybe Bool))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "Dropout" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_Embedding" '(t, u) =
     '[ '("input_dim", AttrReq Int), '("output_dim", AttrReq Int),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"])),
        '("sparse_grad", AttrOpt Bool), '("data", AttrOpt (t u)),
        '("weight", AttrOpt (t u))]

_Embedding ::
           forall a t u .
             (Tensor t, Fullfilled "_Embedding" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "_Embedding" '(t, u) a -> TensorApply (t u)
_Embedding args
  = let scalarArgs
          = catMaybes
              [("input_dim",) . showValue <$> (args !? #input_dim :: Maybe Int),
               ("output_dim",) . showValue <$> (args !? #output_dim :: Maybe Int),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                           "int8", "uint8"])),
               ("sparse_grad",) . showValue <$>
                 (args !? #sparse_grad :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("weight",) . toRaw <$> (args !? #weight :: Maybe (t u))]
      in
      applyRaw "Embedding" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_Flatten" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_Flatten ::
         forall a t u .
           (Tensor t, Fullfilled "_Flatten" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "_Flatten" '(t, u) a -> TensorApply (t u)
_Flatten args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "Flatten" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_FullyConnected" '(t, u) =
     '[ '("num_hidden", AttrReq Int), '("no_bias", AttrOpt Bool),
        '("flatten", AttrOpt Bool), '("data", AttrOpt (t u)),
        '("weight", AttrOpt (t u)), '("bias", AttrOpt (t u))]

_FullyConnected ::
                forall a t u .
                  (Tensor t, Fullfilled "_FullyConnected" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "_FullyConnected" '(t, u) a -> TensorApply (t u)
_FullyConnected args
  = let scalarArgs
          = catMaybes
              [("num_hidden",) . showValue <$>
                 (args !? #num_hidden :: Maybe Int),
               ("no_bias",) . showValue <$> (args !? #no_bias :: Maybe Bool),
               ("flatten",) . showValue <$> (args !? #flatten :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("bias",) . toRaw <$> (args !? #bias :: Maybe (t u))]
      in
      applyRaw "FullyConnected" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_GridGenerator" '(t, u) =
     '[ '("transform_type", AttrReq (EnumType '["affine", "warp"])),
        '("target_shape", AttrOpt [Int]), '("data", AttrOpt (t u))]

_GridGenerator ::
               forall a t u .
                 (Tensor t, Fullfilled "_GridGenerator" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "_GridGenerator" '(t, u) a -> TensorApply (t u)
_GridGenerator args
  = let scalarArgs
          = catMaybes
              [("transform_type",) . showValue <$>
                 (args !? #transform_type :: Maybe (EnumType '["affine", "warp"])),
               ("target_shape",) . showValue <$>
                 (args !? #target_shape :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "GridGenerator" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_GroupNorm" '(t, u) =
     '[ '("num_groups", AttrOpt Int), '("eps", AttrOpt Float),
        '("output_mean_var", AttrOpt Bool), '("data", AttrOpt (t u)),
        '("gamma", AttrOpt (t u)), '("beta", AttrOpt (t u))]

_GroupNorm ::
           forall a t u .
             (Tensor t, Fullfilled "_GroupNorm" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "_GroupNorm" '(t, u) a -> TensorApply (t u)
_GroupNorm args
  = let scalarArgs
          = catMaybes
              [("num_groups",) . showValue <$>
                 (args !? #num_groups :: Maybe Int),
               ("eps",) . showValue <$> (args !? #eps :: Maybe Float),
               ("output_mean_var",) . showValue <$>
                 (args !? #output_mean_var :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("gamma",) . toRaw <$> (args !? #gamma :: Maybe (t u)),
               ("beta",) . toRaw <$> (args !? #beta :: Maybe (t u))]
      in
      applyRaw "GroupNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_IdentityAttachKLSparseReg" '(t, u) =
     '[ '("sparseness_target", AttrOpt Float),
        '("penalty", AttrOpt Float), '("momentum", AttrOpt Float),
        '("data", AttrOpt (t u))]

_IdentityAttachKLSparseReg ::
                           forall a t u .
                             (Tensor t, Fullfilled "_IdentityAttachKLSparseReg" '(t, u) a,
                              HasCallStack, DType u) =>
                             ArgsHMap "_IdentityAttachKLSparseReg" '(t, u) a ->
                               TensorApply (t u)
_IdentityAttachKLSparseReg args
  = let scalarArgs
          = catMaybes
              [("sparseness_target",) . showValue <$>
                 (args !? #sparseness_target :: Maybe Float),
               ("penalty",) . showValue <$> (args !? #penalty :: Maybe Float),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "IdentityAttachKLSparseReg" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_InstanceNorm" '(t, u) =
     '[ '("eps", AttrOpt Float), '("data", AttrOpt (t u)),
        '("gamma", AttrOpt (t u)), '("beta", AttrOpt (t u))]

_InstanceNorm ::
              forall a t u .
                (Tensor t, Fullfilled "_InstanceNorm" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "_InstanceNorm" '(t, u) a -> TensorApply (t u)
_InstanceNorm args
  = let scalarArgs
          = catMaybes
              [("eps",) . showValue <$> (args !? #eps :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("gamma",) . toRaw <$> (args !? #gamma :: Maybe (t u)),
               ("beta",) . toRaw <$> (args !? #beta :: Maybe (t u))]
      in
      applyRaw "InstanceNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_L2Normalization" '(t, u) =
     '[ '("eps", AttrOpt Float),
        '("mode", AttrOpt (EnumType '["channel", "instance", "spatial"])),
        '("data", AttrOpt (t u))]

_L2Normalization ::
                 forall a t u .
                   (Tensor t, Fullfilled "_L2Normalization" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "_L2Normalization" '(t, u) a -> TensorApply (t u)
_L2Normalization args
  = let scalarArgs
          = catMaybes
              [("eps",) . showValue <$> (args !? #eps :: Maybe Float),
               ("mode",) . showValue <$>
                 (args !? #mode ::
                    Maybe (EnumType '["channel", "instance", "spatial"]))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "L2Normalization" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_LRN" '(t, u) =
     '[ '("alpha", AttrOpt Float), '("beta", AttrOpt Float),
        '("knorm", AttrOpt Float), '("nsize", AttrReq Int),
        '("data", AttrOpt (t u))]

_LRN ::
     forall a t u .
       (Tensor t, Fullfilled "_LRN" '(t, u) a, HasCallStack, DType u) =>
       ArgsHMap "_LRN" '(t, u) a -> TensorApply (t u)
_LRN args
  = let scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> (args !? #alpha :: Maybe Float),
               ("beta",) . showValue <$> (args !? #beta :: Maybe Float),
               ("knorm",) . showValue <$> (args !? #knorm :: Maybe Float),
               ("nsize",) . showValue <$> (args !? #nsize :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "LRN" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_LayerNorm" '(t, u) =
     '[ '("axis", AttrOpt Int), '("eps", AttrOpt Float),
        '("output_mean_var", AttrOpt Bool), '("data", AttrOpt (t u)),
        '("gamma", AttrOpt (t u)), '("beta", AttrOpt (t u))]

_LayerNorm ::
           forall a t u .
             (Tensor t, Fullfilled "_LayerNorm" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "_LayerNorm" '(t, u) a -> TensorApply (t u)
_LayerNorm args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("eps",) . showValue <$> (args !? #eps :: Maybe Float),
               ("output_mean_var",) . showValue <$>
                 (args !? #output_mean_var :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("gamma",) . toRaw <$> (args !? #gamma :: Maybe (t u)),
               ("beta",) . toRaw <$> (args !? #beta :: Maybe (t u))]
      in
      applyRaw "LayerNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_LeakyReLU" '(t, u) =
     '[ '("act_type",
          AttrOpt
            (EnumType '["elu", "gelu", "leaky", "prelu", "rrelu", "selu"])),
        '("slope", AttrOpt Float), '("lower_bound", AttrOpt Float),
        '("upper_bound", AttrOpt Float), '("data", AttrOpt (t u)),
        '("gamma", AttrOpt (t u))]

_LeakyReLU ::
           forall a t u .
             (Tensor t, Fullfilled "_LeakyReLU" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "_LeakyReLU" '(t, u) a -> TensorApply (t u)
_LeakyReLU args
  = let scalarArgs
          = catMaybes
              [("act_type",) . showValue <$>
                 (args !? #act_type ::
                    Maybe
                      (EnumType '["elu", "gelu", "leaky", "prelu", "rrelu", "selu"])),
               ("slope",) . showValue <$> (args !? #slope :: Maybe Float),
               ("lower_bound",) . showValue <$>
                 (args !? #lower_bound :: Maybe Float),
               ("upper_bound",) . showValue <$>
                 (args !? #upper_bound :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("gamma",) . toRaw <$> (args !? #gamma :: Maybe (t u))]
      in
      applyRaw "LeakyReLU" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_LinearRegressionOutput" '(t, u) =
     '[ '("grad_scale", AttrOpt Float), '("data", AttrOpt (t u)),
        '("label", AttrOpt (t u))]

_LinearRegressionOutput ::
                        forall a t u .
                          (Tensor t, Fullfilled "_LinearRegressionOutput" '(t, u) a,
                           HasCallStack, DType u) =>
                          ArgsHMap "_LinearRegressionOutput" '(t, u) a -> TensorApply (t u)
_LinearRegressionOutput args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("label",) . toRaw <$> (args !? #label :: Maybe (t u))]
      in
      applyRaw "LinearRegressionOutput" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_LogisticRegressionOutput" '(t, u) =
     '[ '("grad_scale", AttrOpt Float), '("data", AttrOpt (t u)),
        '("label", AttrOpt (t u))]

_LogisticRegressionOutput ::
                          forall a t u .
                            (Tensor t, Fullfilled "_LogisticRegressionOutput" '(t, u) a,
                             HasCallStack, DType u) =>
                            ArgsHMap "_LogisticRegressionOutput" '(t, u) a -> TensorApply (t u)
_LogisticRegressionOutput args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("label",) . toRaw <$> (args !? #label :: Maybe (t u))]
      in
      applyRaw "LogisticRegressionOutput" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_MAERegressionOutput" '(t, u) =
     '[ '("grad_scale", AttrOpt Float), '("data", AttrOpt (t u)),
        '("label", AttrOpt (t u))]

_MAERegressionOutput ::
                     forall a t u .
                       (Tensor t, Fullfilled "_MAERegressionOutput" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "_MAERegressionOutput" '(t, u) a -> TensorApply (t u)
_MAERegressionOutput args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("label",) . toRaw <$> (args !? #label :: Maybe (t u))]
      in
      applyRaw "MAERegressionOutput" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_MakeLoss" '(t, u) =
     '[ '("grad_scale", AttrOpt Float),
        '("valid_thresh", AttrOpt Float),
        '("normalization", AttrOpt (EnumType '["batch", "null", "valid"])),
        '("data", AttrOpt (t u))]

_MakeLoss ::
          forall a t u .
            (Tensor t, Fullfilled "_MakeLoss" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "_MakeLoss" '(t, u) a -> TensorApply (t u)
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
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "MakeLoss" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_Pad" '(t, u) =
     '[ '("mode", AttrReq (EnumType '["constant", "edge", "reflect"])),
        '("pad_width", AttrReq [Int]), '("constant_value", AttrOpt Double),
        '("data", AttrOpt (t u))]

_Pad ::
     forall a t u .
       (Tensor t, Fullfilled "_Pad" '(t, u) a, HasCallStack, DType u) =>
       ArgsHMap "_Pad" '(t, u) a -> TensorApply (t u)
_Pad args
  = let scalarArgs
          = catMaybes
              [("mode",) . showValue <$>
                 (args !? #mode ::
                    Maybe (EnumType '["constant", "edge", "reflect"])),
               ("pad_width",) . showValue <$> (args !? #pad_width :: Maybe [Int]),
               ("constant_value",) . showValue <$>
                 (args !? #constant_value :: Maybe Double)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "Pad" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_Pooling" '(t, u) =
     '[ '("kernel", AttrOpt [Int]),
        '("pool_type", AttrOpt (EnumType '["avg", "lp", "max", "sum"])),
        '("global_pool", AttrOpt Bool), '("cudnn_off", AttrOpt Bool),
        '("pooling_convention",
          AttrOpt (EnumType '["full", "same", "valid"])),
        '("stride", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("p_value", AttrOpt (Maybe Int)),
        '("count_include_pad", AttrOpt (Maybe Bool)),
        '("layout",
          AttrOpt
            (Maybe
               (EnumType '["NCDHW", "NCHW", "NCW", "NDHWC", "NHWC", "NWC"]))),
        '("data", AttrOpt (t u))]

_Pooling ::
         forall a t u .
           (Tensor t, Fullfilled "_Pooling" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "_Pooling" '(t, u) a -> TensorApply (t u)
_Pooling args
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
                    Maybe (EnumType '["full", "same", "valid"])),
               ("stride",) . showValue <$> (args !? #stride :: Maybe [Int]),
               ("pad",) . showValue <$> (args !? #pad :: Maybe [Int]),
               ("p_value",) . showValue <$>
                 (args !? #p_value :: Maybe (Maybe Int)),
               ("count_include_pad",) . showValue <$>
                 (args !? #count_include_pad :: Maybe (Maybe Bool)),
               ("layout",) . showValue <$>
                 (args !? #layout ::
                    Maybe
                      (Maybe
                         (EnumType '["NCDHW", "NCHW", "NCW", "NDHWC", "NHWC", "NWC"])))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "Pooling" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_Pooling_v1" '(t, u) =
     '[ '("kernel", AttrOpt [Int]),
        '("pool_type", AttrOpt (EnumType '["avg", "max", "sum"])),
        '("global_pool", AttrOpt Bool),
        '("pooling_convention", AttrOpt (EnumType '["full", "valid"])),
        '("stride", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("data", AttrOpt (t u))]

_Pooling_v1 ::
            forall a t u .
              (Tensor t, Fullfilled "_Pooling_v1" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "_Pooling_v1" '(t, u) a -> TensorApply (t u)
_Pooling_v1 args
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
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "Pooling_v1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_RNN" '(t, u) =
     '[ '("state_size", AttrReq Int), '("num_layers", AttrReq Int),
        '("bidirectional", AttrOpt Bool),
        '("mode",
          AttrReq (EnumType '["gru", "lstm", "rnn_relu", "rnn_tanh"])),
        '("p", AttrOpt Float), '("state_outputs", AttrOpt Bool),
        '("projection_size", AttrOpt (Maybe Int)),
        '("lstm_state_clip_min", AttrOpt (Maybe Double)),
        '("lstm_state_clip_max", AttrOpt (Maybe Double)),
        '("lstm_state_clip_nan", AttrOpt Bool),
        '("use_sequence_length", AttrOpt Bool), '("data", AttrOpt (t u)),
        '("parameters", AttrOpt (t u)), '("state", AttrOpt (t u)),
        '("state_cell", AttrOpt (t u)),
        '("sequence_length", AttrOpt (t u))]

_RNN ::
     forall a t u .
       (Tensor t, Fullfilled "_RNN" '(t, u) a, HasCallStack, DType u) =>
       ArgsHMap "_RNN" '(t, u) a -> TensorApply (t u)
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
                 (args !? #state_outputs :: Maybe Bool),
               ("projection_size",) . showValue <$>
                 (args !? #projection_size :: Maybe (Maybe Int)),
               ("lstm_state_clip_min",) . showValue <$>
                 (args !? #lstm_state_clip_min :: Maybe (Maybe Double)),
               ("lstm_state_clip_max",) . showValue <$>
                 (args !? #lstm_state_clip_max :: Maybe (Maybe Double)),
               ("lstm_state_clip_nan",) . showValue <$>
                 (args !? #lstm_state_clip_nan :: Maybe Bool),
               ("use_sequence_length",) . showValue <$>
                 (args !? #use_sequence_length :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("parameters",) . toRaw <$> (args !? #parameters :: Maybe (t u)),
               ("state",) . toRaw <$> (args !? #state :: Maybe (t u)),
               ("state_cell",) . toRaw <$> (args !? #state_cell :: Maybe (t u)),
               ("sequence_length",) . toRaw <$>
                 (args !? #sequence_length :: Maybe (t u))]
      in
      applyRaw "RNN" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_ROIPooling" '(t, u) =
     '[ '("pooled_size", AttrReq [Int]),
        '("spatial_scale", AttrReq Float), '("data", AttrOpt (t u)),
        '("rois", AttrOpt (t u))]

_ROIPooling ::
            forall a t u .
              (Tensor t, Fullfilled "_ROIPooling" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "_ROIPooling" '(t, u) a -> TensorApply (t u)
_ROIPooling args
  = let scalarArgs
          = catMaybes
              [("pooled_size",) . showValue <$>
                 (args !? #pooled_size :: Maybe [Int]),
               ("spatial_scale",) . showValue <$>
                 (args !? #spatial_scale :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("rois",) . toRaw <$> (args !? #rois :: Maybe (t u))]
      in
      applyRaw "ROIPooling" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_Reshape" '(t, u) =
     '[ '("shape", AttrOpt [Int]), '("reverse", AttrOpt Bool),
        '("target_shape", AttrOpt [Int]), '("keep_highest", AttrOpt Bool),
        '("data", AttrOpt (t u))]

_Reshape ::
         forall a t u .
           (Tensor t, Fullfilled "_Reshape" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "_Reshape" '(t, u) a -> TensorApply (t u)
_Reshape args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("reverse",) . showValue <$> (args !? #reverse :: Maybe Bool),
               ("target_shape",) . showValue <$>
                 (args !? #target_shape :: Maybe [Int]),
               ("keep_highest",) . showValue <$>
                 (args !? #keep_highest :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "Reshape" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_SVMOutput" '(t, u) =
     '[ '("margin", AttrOpt Float),
        '("regularization_coefficient", AttrOpt Float),
        '("use_linear", AttrOpt Bool), '("data", AttrOpt (t u)),
        '("label", AttrOpt (t u))]

_SVMOutput ::
           forall a t u .
             (Tensor t, Fullfilled "_SVMOutput" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "_SVMOutput" '(t, u) a -> TensorApply (t u)
_SVMOutput args
  = let scalarArgs
          = catMaybes
              [("margin",) . showValue <$> (args !? #margin :: Maybe Float),
               ("regularization_coefficient",) . showValue <$>
                 (args !? #regularization_coefficient :: Maybe Float),
               ("use_linear",) . showValue <$>
                 (args !? #use_linear :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("label",) . toRaw <$> (args !? #label :: Maybe (t u))]
      in
      applyRaw "SVMOutput" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_SequenceLast" '(t, u) =
     '[ '("use_sequence_length", AttrOpt Bool), '("axis", AttrOpt Int),
        '("data", AttrOpt (t u)), '("sequence_length", AttrOpt (t u))]

_SequenceLast ::
              forall a t u .
                (Tensor t, Fullfilled "_SequenceLast" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "_SequenceLast" '(t, u) a -> TensorApply (t u)
_SequenceLast args
  = let scalarArgs
          = catMaybes
              [("use_sequence_length",) . showValue <$>
                 (args !? #use_sequence_length :: Maybe Bool),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("sequence_length",) . toRaw <$>
                 (args !? #sequence_length :: Maybe (t u))]
      in
      applyRaw "SequenceLast" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_SequenceMask" '(t, u) =
     '[ '("use_sequence_length", AttrOpt Bool),
        '("value", AttrOpt Float), '("axis", AttrOpt Int),
        '("data", AttrOpt (t u)), '("sequence_length", AttrOpt (t u))]

_SequenceMask ::
              forall a t u .
                (Tensor t, Fullfilled "_SequenceMask" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "_SequenceMask" '(t, u) a -> TensorApply (t u)
_SequenceMask args
  = let scalarArgs
          = catMaybes
              [("use_sequence_length",) . showValue <$>
                 (args !? #use_sequence_length :: Maybe Bool),
               ("value",) . showValue <$> (args !? #value :: Maybe Float),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("sequence_length",) . toRaw <$>
                 (args !? #sequence_length :: Maybe (t u))]
      in
      applyRaw "SequenceMask" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_SequenceReverse" '(t, u) =
     '[ '("use_sequence_length", AttrOpt Bool), '("axis", AttrOpt Int),
        '("data", AttrOpt (t u)), '("sequence_length", AttrOpt (t u))]

_SequenceReverse ::
                 forall a t u .
                   (Tensor t, Fullfilled "_SequenceReverse" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "_SequenceReverse" '(t, u) a -> TensorApply (t u)
_SequenceReverse args
  = let scalarArgs
          = catMaybes
              [("use_sequence_length",) . showValue <$>
                 (args !? #use_sequence_length :: Maybe Bool),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("sequence_length",) . toRaw <$>
                 (args !? #sequence_length :: Maybe (t u))]
      in
      applyRaw "SequenceReverse" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_SliceChannel" '(t, u) =
     '[ '("num_outputs", AttrReq Int), '("axis", AttrOpt Int),
        '("squeeze_axis", AttrOpt Bool), '("data", AttrOpt (t u))]

_SliceChannel ::
              forall a t u .
                (Tensor t, Fullfilled "_SliceChannel" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "_SliceChannel" '(t, u) a -> TensorApply (t u)
_SliceChannel args
  = let scalarArgs
          = catMaybes
              [("num_outputs",) . showValue <$>
                 (args !? #num_outputs :: Maybe Int),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("squeeze_axis",) . showValue <$>
                 (args !? #squeeze_axis :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "SliceChannel" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_SoftmaxActivation" '(t, u) =
     '[ '("mode", AttrOpt (EnumType '["channel", "instance"])),
        '("data", AttrOpt (t u))]

_SoftmaxActivation ::
                   forall a t u .
                     (Tensor t, Fullfilled "_SoftmaxActivation" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "_SoftmaxActivation" '(t, u) a -> TensorApply (t u)
_SoftmaxActivation args
  = let scalarArgs
          = catMaybes
              [("mode",) . showValue <$>
                 (args !? #mode :: Maybe (EnumType '["channel", "instance"]))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "SoftmaxActivation" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_SoftmaxOutput" '(t, u) =
     '[ '("grad_scale", AttrOpt Float),
        '("ignore_label", AttrOpt Float), '("multi_output", AttrOpt Bool),
        '("use_ignore", AttrOpt Bool), '("preserve_shape", AttrOpt Bool),
        '("normalization", AttrOpt (EnumType '["batch", "null", "valid"])),
        '("out_grad", AttrOpt Bool), '("smooth_alpha", AttrOpt Float),
        '("data", AttrOpt (t u)), '("label", AttrOpt (t u))]

_SoftmaxOutput ::
               forall a t u .
                 (Tensor t, Fullfilled "_SoftmaxOutput" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "_SoftmaxOutput" '(t, u) a -> TensorApply (t u)
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
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("label",) . toRaw <$> (args !? #label :: Maybe (t u))]
      in
      applyRaw "SoftmaxOutput" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_SpatialTransformer" '(t, u) =
     '[ '("target_shape", AttrOpt [Int]),
        '("transform_type", AttrReq (EnumType '["affine"])),
        '("sampler_type", AttrReq (EnumType '["bilinear"])),
        '("cudnn_off", AttrOpt (Maybe Bool)), '("data", AttrOpt (t u)),
        '("loc", AttrOpt (t u))]

_SpatialTransformer ::
                    forall a t u .
                      (Tensor t, Fullfilled "_SpatialTransformer" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "_SpatialTransformer" '(t, u) a -> TensorApply (t u)
_SpatialTransformer args
  = let scalarArgs
          = catMaybes
              [("target_shape",) . showValue <$>
                 (args !? #target_shape :: Maybe [Int]),
               ("transform_type",) . showValue <$>
                 (args !? #transform_type :: Maybe (EnumType '["affine"])),
               ("sampler_type",) . showValue <$>
                 (args !? #sampler_type :: Maybe (EnumType '["bilinear"])),
               ("cudnn_off",) . showValue <$>
                 (args !? #cudnn_off :: Maybe (Maybe Bool))]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("loc",) . toRaw <$> (args !? #loc :: Maybe (t u))]
      in
      applyRaw "SpatialTransformer" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_SwapAxis" '(t, u) =
     '[ '("dim1", AttrOpt Int), '("dim2", AttrOpt Int),
        '("data", AttrOpt (t u))]

_SwapAxis ::
          forall a t u .
            (Tensor t, Fullfilled "_SwapAxis" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "_SwapAxis" '(t, u) a -> TensorApply (t u)
_SwapAxis args
  = let scalarArgs
          = catMaybes
              [("dim1",) . showValue <$> (args !? #dim1 :: Maybe Int),
               ("dim2",) . showValue <$> (args !? #dim2 :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "SwapAxis" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_UpSampling" '(t, u) =
     '[ '("scale", AttrReq Int), '("num_filter", AttrOpt Int),
        '("sample_type", AttrReq (EnumType '["bilinear", "nearest"])),
        '("multi_input_mode", AttrOpt (EnumType '["concat", "sum"])),
        '("num_args", AttrReq Int), '("workspace", AttrOpt Int),
        '("data", AttrOpt [t u])]

_UpSampling ::
            forall a t u .
              (Tensor t, Fullfilled "_UpSampling" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "_UpSampling" '(t, u) a -> TensorApply (t u)
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
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "UpSampling" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__CachedOp" '(t, u) =
     '[ '("data", AttrOpt [t u])]

__CachedOp ::
           forall a t u .
             (Tensor t, Fullfilled "__CachedOp" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__CachedOp" '(t, u) a -> TensorApply (t u)
__CachedOp args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "_CachedOp" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__CachedOpThreadSafe" '(t, u) =
     '[ '("data", AttrOpt [t u])]

__CachedOpThreadSafe ::
                     forall a t u .
                       (Tensor t, Fullfilled "__CachedOpThreadSafe" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "__CachedOpThreadSafe" '(t, u) a -> TensorApply (t u)
__CachedOpThreadSafe args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "_CachedOpThreadSafe" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__FusedOp" '(t, u) =
     '[ '("data", AttrOpt [t u])]

__FusedOp ::
          forall a t u .
            (Tensor t, Fullfilled "__FusedOp" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__FusedOp" '(t, u) a -> TensorApply (t u)
__FusedOp args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "_FusedOp" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__NoGradient" '() = '[]

__NoGradient ::
             forall a t u .
               (Tensor t, Fullfilled "__NoGradient" '() a, HasCallStack,
                DType u) =>
               ArgsHMap "__NoGradient" '() a -> TensorApply (t u)
__NoGradient args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_NoGradient" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__adamw_update" '(t, u) =
     '[ '("lr", AttrReq Float), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wd", AttrOpt Float), '("eta", AttrReq Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u)), '("mean", AttrOpt (t u)),
        '("var", AttrOpt (t u)), '("rescale_grad", AttrOpt (t u))]

__adamw_update ::
               forall a t u .
                 (Tensor t, Fullfilled "__adamw_update" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__adamw_update" '(t, u) a -> TensorApply (t u)
__adamw_update args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("beta1",) . showValue <$> (args !? #beta1 :: Maybe Float),
               ("beta2",) . showValue <$> (args !? #beta2 :: Maybe Float),
               ("epsilon",) . showValue <$> (args !? #epsilon :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("eta",) . showValue <$> (args !? #eta :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("grad",) . toRaw <$> (args !? #grad :: Maybe (t u)),
               ("mean",) . toRaw <$> (args !? #mean :: Maybe (t u)),
               ("var",) . toRaw <$> (args !? #var :: Maybe (t u)),
               ("rescale_grad",) . toRaw <$>
                 (args !? #rescale_grad :: Maybe (t u))]
      in
      applyRaw "_adamw_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__arange" '() =
     '[ '("start", AttrReq Double), '("stop", AttrOpt (Maybe Double)),
        '("step", AttrOpt Double), '("repeat", AttrOpt Int),
        '("infer_range", AttrOpt Bool), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"]))]

__arange ::
         forall a t u .
           (Tensor t, Fullfilled "__arange" '() a, HasCallStack, DType u) =>
           ArgsHMap "__arange" '() a -> TensorApply (t u)
__arange args
  = let scalarArgs
          = catMaybes
              [("start",) . showValue <$> (args !? #start :: Maybe Double),
               ("stop",) . showValue <$> (args !? #stop :: Maybe (Maybe Double)),
               ("step",) . showValue <$> (args !? #step :: Maybe Double),
               ("repeat",) . showValue <$> (args !? #repeat :: Maybe Int),
               ("infer_range",) . showValue <$>
                 (args !? #infer_range :: Maybe Bool),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                           "int8", "uint8"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_arange" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_Activation" '() = '[]

__backward_Activation ::
                      forall a t u .
                        (Tensor t, Fullfilled "__backward_Activation" '() a, HasCallStack,
                         DType u) =>
                        ArgsHMap "__backward_Activation" '() a -> TensorApply (t u)
__backward_Activation args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Activation" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_BatchNorm" '() = '[]

__backward_BatchNorm ::
                     forall a t u .
                       (Tensor t, Fullfilled "__backward_BatchNorm" '() a, HasCallStack,
                        DType u) =>
                       ArgsHMap "__backward_BatchNorm" '() a -> TensorApply (t u)
__backward_BatchNorm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_BatchNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_BatchNorm_v1" '() = '[]

__backward_BatchNorm_v1 ::
                        forall a t u .
                          (Tensor t, Fullfilled "__backward_BatchNorm_v1" '() a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__backward_BatchNorm_v1" '() a -> TensorApply (t u)
__backward_BatchNorm_v1 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_BatchNorm_v1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_BilinearSampler" '() = '[]

__backward_BilinearSampler ::
                           forall a t u .
                             (Tensor t, Fullfilled "__backward_BilinearSampler" '() a,
                              HasCallStack, DType u) =>
                             ArgsHMap "__backward_BilinearSampler" '() a -> TensorApply (t u)
__backward_BilinearSampler args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_BilinearSampler" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_CachedOp" '() = '[]

__backward_CachedOp ::
                    forall a t u .
                      (Tensor t, Fullfilled "__backward_CachedOp" '() a, HasCallStack,
                       DType u) =>
                      ArgsHMap "__backward_CachedOp" '() a -> TensorApply (t u)
__backward_CachedOp args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_CachedOp" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_Concat" '() = '[]

__backward_Concat ::
                  forall a t u .
                    (Tensor t, Fullfilled "__backward_Concat" '() a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__backward_Concat" '() a -> TensorApply (t u)
__backward_Concat args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Concat" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_Convolution" '() = '[]

__backward_Convolution ::
                       forall a t u .
                         (Tensor t, Fullfilled "__backward_Convolution" '() a, HasCallStack,
                          DType u) =>
                         ArgsHMap "__backward_Convolution" '() a -> TensorApply (t u)
__backward_Convolution args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Convolution" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_Convolution_v1" '() = '[]

__backward_Convolution_v1 ::
                          forall a t u .
                            (Tensor t, Fullfilled "__backward_Convolution_v1" '() a,
                             HasCallStack, DType u) =>
                            ArgsHMap "__backward_Convolution_v1" '() a -> TensorApply (t u)
__backward_Convolution_v1 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Convolution_v1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_Correlation" '() = '[]

__backward_Correlation ::
                       forall a t u .
                         (Tensor t, Fullfilled "__backward_Correlation" '() a, HasCallStack,
                          DType u) =>
                         ArgsHMap "__backward_Correlation" '() a -> TensorApply (t u)
__backward_Correlation args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Correlation" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_Crop" '() = '[]

__backward_Crop ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_Crop" '() a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_Crop" '() a -> TensorApply (t u)
__backward_Crop args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Crop" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_CuDNNBatchNorm" '() = '[]

__backward_CuDNNBatchNorm ::
                          forall a t u .
                            (Tensor t, Fullfilled "__backward_CuDNNBatchNorm" '() a,
                             HasCallStack, DType u) =>
                            ArgsHMap "__backward_CuDNNBatchNorm" '() a -> TensorApply (t u)
__backward_CuDNNBatchNorm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_CuDNNBatchNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_Custom" '() = '[]

__backward_Custom ::
                  forall a t u .
                    (Tensor t, Fullfilled "__backward_Custom" '() a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__backward_Custom" '() a -> TensorApply (t u)
__backward_Custom args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Custom" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_CustomFunction" '() = '[]

__backward_CustomFunction ::
                          forall a t u .
                            (Tensor t, Fullfilled "__backward_CustomFunction" '() a,
                             HasCallStack, DType u) =>
                            ArgsHMap "__backward_CustomFunction" '() a -> TensorApply (t u)
__backward_CustomFunction args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_CustomFunction" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_Deconvolution" '() = '[]

__backward_Deconvolution ::
                         forall a t u .
                           (Tensor t, Fullfilled "__backward_Deconvolution" '() a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__backward_Deconvolution" '() a -> TensorApply (t u)
__backward_Deconvolution args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Deconvolution" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_Dropout" '() = '[]

__backward_Dropout ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_Dropout" '() a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_Dropout" '() a -> TensorApply (t u)
__backward_Dropout args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Dropout" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_Embedding" '() = '[]

__backward_Embedding ::
                     forall a t u .
                       (Tensor t, Fullfilled "__backward_Embedding" '() a, HasCallStack,
                        DType u) =>
                       ArgsHMap "__backward_Embedding" '() a -> TensorApply (t u)
__backward_Embedding args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Embedding" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_FullyConnected" '() = '[]

__backward_FullyConnected ::
                          forall a t u .
                            (Tensor t, Fullfilled "__backward_FullyConnected" '() a,
                             HasCallStack, DType u) =>
                            ArgsHMap "__backward_FullyConnected" '() a -> TensorApply (t u)
__backward_FullyConnected args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_FullyConnected" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_GridGenerator" '() = '[]

__backward_GridGenerator ::
                         forall a t u .
                           (Tensor t, Fullfilled "__backward_GridGenerator" '() a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__backward_GridGenerator" '() a -> TensorApply (t u)
__backward_GridGenerator args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_GridGenerator" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_GroupNorm" '() = '[]

__backward_GroupNorm ::
                     forall a t u .
                       (Tensor t, Fullfilled "__backward_GroupNorm" '() a, HasCallStack,
                        DType u) =>
                       ArgsHMap "__backward_GroupNorm" '() a -> TensorApply (t u)
__backward_GroupNorm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_GroupNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__backward_IdentityAttachKLSparseReg" '() = '[]

__backward_IdentityAttachKLSparseReg ::
                                     forall a t u .
                                       (Tensor t,
                                        Fullfilled "__backward_IdentityAttachKLSparseReg" '() a,
                                        HasCallStack, DType u) =>
                                       ArgsHMap "__backward_IdentityAttachKLSparseReg" '() a ->
                                         TensorApply (t u)
__backward_IdentityAttachKLSparseReg args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_IdentityAttachKLSparseReg" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_InstanceNorm" '() = '[]

__backward_InstanceNorm ::
                        forall a t u .
                          (Tensor t, Fullfilled "__backward_InstanceNorm" '() a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__backward_InstanceNorm" '() a -> TensorApply (t u)
__backward_InstanceNorm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_InstanceNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_L2Normalization" '() = '[]

__backward_L2Normalization ::
                           forall a t u .
                             (Tensor t, Fullfilled "__backward_L2Normalization" '() a,
                              HasCallStack, DType u) =>
                             ArgsHMap "__backward_L2Normalization" '() a -> TensorApply (t u)
__backward_L2Normalization args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_L2Normalization" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_LRN" '() = '[]

__backward_LRN ::
               forall a t u .
                 (Tensor t, Fullfilled "__backward_LRN" '() a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__backward_LRN" '() a -> TensorApply (t u)
__backward_LRN args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_LRN" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_LayerNorm" '() = '[]

__backward_LayerNorm ::
                     forall a t u .
                       (Tensor t, Fullfilled "__backward_LayerNorm" '() a, HasCallStack,
                        DType u) =>
                       ArgsHMap "__backward_LayerNorm" '() a -> TensorApply (t u)
__backward_LayerNorm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_LayerNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_LeakyReLU" '() = '[]

__backward_LeakyReLU ::
                     forall a t u .
                       (Tensor t, Fullfilled "__backward_LeakyReLU" '() a, HasCallStack,
                        DType u) =>
                       ArgsHMap "__backward_LeakyReLU" '() a -> TensorApply (t u)
__backward_LeakyReLU args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_LeakyReLU" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_MakeLoss" '() = '[]

__backward_MakeLoss ::
                    forall a t u .
                      (Tensor t, Fullfilled "__backward_MakeLoss" '() a, HasCallStack,
                       DType u) =>
                      ArgsHMap "__backward_MakeLoss" '() a -> TensorApply (t u)
__backward_MakeLoss args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_MakeLoss" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_Pad" '() = '[]

__backward_Pad ::
               forall a t u .
                 (Tensor t, Fullfilled "__backward_Pad" '() a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__backward_Pad" '() a -> TensorApply (t u)
__backward_Pad args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Pad" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_Pooling" '() = '[]

__backward_Pooling ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_Pooling" '() a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_Pooling" '() a -> TensorApply (t u)
__backward_Pooling args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Pooling" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_Pooling_v1" '() = '[]

__backward_Pooling_v1 ::
                      forall a t u .
                        (Tensor t, Fullfilled "__backward_Pooling_v1" '() a, HasCallStack,
                         DType u) =>
                        ArgsHMap "__backward_Pooling_v1" '() a -> TensorApply (t u)
__backward_Pooling_v1 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_Pooling_v1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_RNN" '() = '[]

__backward_RNN ::
               forall a t u .
                 (Tensor t, Fullfilled "__backward_RNN" '() a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__backward_RNN" '() a -> TensorApply (t u)
__backward_RNN args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_RNN" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_ROIAlign" '() = '[]

__backward_ROIAlign ::
                    forall a t u .
                      (Tensor t, Fullfilled "__backward_ROIAlign" '() a, HasCallStack,
                       DType u) =>
                      ArgsHMap "__backward_ROIAlign" '() a -> TensorApply (t u)
__backward_ROIAlign args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_ROIAlign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_ROIPooling" '() = '[]

__backward_ROIPooling ::
                      forall a t u .
                        (Tensor t, Fullfilled "__backward_ROIPooling" '() a, HasCallStack,
                         DType u) =>
                        ArgsHMap "__backward_ROIPooling" '() a -> TensorApply (t u)
__backward_ROIPooling args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_ROIPooling" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_RROIAlign" '() = '[]

__backward_RROIAlign ::
                     forall a t u .
                       (Tensor t, Fullfilled "__backward_RROIAlign" '() a, HasCallStack,
                        DType u) =>
                       ArgsHMap "__backward_RROIAlign" '() a -> TensorApply (t u)
__backward_RROIAlign args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_RROIAlign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_SVMOutput" '() = '[]

__backward_SVMOutput ::
                     forall a t u .
                       (Tensor t, Fullfilled "__backward_SVMOutput" '() a, HasCallStack,
                        DType u) =>
                       ArgsHMap "__backward_SVMOutput" '() a -> TensorApply (t u)
__backward_SVMOutput args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_SVMOutput" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_SequenceLast" '() = '[]

__backward_SequenceLast ::
                        forall a t u .
                          (Tensor t, Fullfilled "__backward_SequenceLast" '() a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__backward_SequenceLast" '() a -> TensorApply (t u)
__backward_SequenceLast args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_SequenceLast" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_SequenceMask" '() = '[]

__backward_SequenceMask ::
                        forall a t u .
                          (Tensor t, Fullfilled "__backward_SequenceMask" '() a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__backward_SequenceMask" '() a -> TensorApply (t u)
__backward_SequenceMask args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_SequenceMask" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_SequenceReverse" '() = '[]

__backward_SequenceReverse ::
                           forall a t u .
                             (Tensor t, Fullfilled "__backward_SequenceReverse" '() a,
                              HasCallStack, DType u) =>
                             ArgsHMap "__backward_SequenceReverse" '() a -> TensorApply (t u)
__backward_SequenceReverse args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_SequenceReverse" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_SliceChannel" '() = '[]

__backward_SliceChannel ::
                        forall a t u .
                          (Tensor t, Fullfilled "__backward_SliceChannel" '() a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__backward_SliceChannel" '() a -> TensorApply (t u)
__backward_SliceChannel args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_SliceChannel" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_SoftmaxActivation" '() =
     '[]

__backward_SoftmaxActivation ::
                             forall a t u .
                               (Tensor t, Fullfilled "__backward_SoftmaxActivation" '() a,
                                HasCallStack, DType u) =>
                               ArgsHMap "__backward_SoftmaxActivation" '() a -> TensorApply (t u)
__backward_SoftmaxActivation args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_SoftmaxActivation" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_SoftmaxOutput" '() = '[]

__backward_SoftmaxOutput ::
                         forall a t u .
                           (Tensor t, Fullfilled "__backward_SoftmaxOutput" '() a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__backward_SoftmaxOutput" '() a -> TensorApply (t u)
__backward_SoftmaxOutput args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_SoftmaxOutput" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_SparseEmbedding" '() = '[]

__backward_SparseEmbedding ::
                           forall a t u .
                             (Tensor t, Fullfilled "__backward_SparseEmbedding" '() a,
                              HasCallStack, DType u) =>
                             ArgsHMap "__backward_SparseEmbedding" '() a -> TensorApply (t u)
__backward_SparseEmbedding args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_SparseEmbedding" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_SpatialTransformer" '() =
     '[]

__backward_SpatialTransformer ::
                              forall a t u .
                                (Tensor t, Fullfilled "__backward_SpatialTransformer" '() a,
                                 HasCallStack, DType u) =>
                                ArgsHMap "__backward_SpatialTransformer" '() a -> TensorApply (t u)
__backward_SpatialTransformer args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_SpatialTransformer" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_SwapAxis" '() = '[]

__backward_SwapAxis ::
                    forall a t u .
                      (Tensor t, Fullfilled "__backward_SwapAxis" '() a, HasCallStack,
                       DType u) =>
                      ArgsHMap "__backward_SwapAxis" '() a -> TensorApply (t u)
__backward_SwapAxis args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_SwapAxis" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_UpSampling" '() = '[]

__backward_UpSampling ::
                      forall a t u .
                        (Tensor t, Fullfilled "__backward_UpSampling" '() a, HasCallStack,
                         DType u) =>
                        ArgsHMap "__backward_UpSampling" '() a -> TensorApply (t u)
__backward_UpSampling args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_UpSampling" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward__CrossDeviceCopy" '() = '[]

__backward__CrossDeviceCopy ::
                            forall a t u .
                              (Tensor t, Fullfilled "__backward__CrossDeviceCopy" '() a,
                               HasCallStack, DType u) =>
                              ArgsHMap "__backward__CrossDeviceCopy" '() a -> TensorApply (t u)
__backward__CrossDeviceCopy args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__CrossDeviceCopy" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward__NDArray" '() = '[]

__backward__NDArray ::
                    forall a t u .
                      (Tensor t, Fullfilled "__backward__NDArray" '() a, HasCallStack,
                       DType u) =>
                      ArgsHMap "__backward__NDArray" '() a -> TensorApply (t u)
__backward__NDArray args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__NDArray" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward__Native" '() = '[]

__backward__Native ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward__Native" '() a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward__Native" '() a -> TensorApply (t u)
__backward__Native args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__Native" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__backward__contrib_DeformableConvolution" '() = '[]

__backward__contrib_DeformableConvolution ::
                                          forall a t u .
                                            (Tensor t,
                                             Fullfilled "__backward__contrib_DeformableConvolution"
                                               '()
                                               a,
                                             HasCallStack, DType u) =>
                                            ArgsHMap "__backward__contrib_DeformableConvolution" '()
                                              a
                                              -> TensorApply (t u)
__backward__contrib_DeformableConvolution args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_DeformableConvolution" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__backward__contrib_DeformablePSROIPooling" '() =
     '[]

__backward__contrib_DeformablePSROIPooling ::
                                           forall a t u .
                                             (Tensor t,
                                              Fullfilled
                                                "__backward__contrib_DeformablePSROIPooling"
                                                '()
                                                a,
                                              HasCallStack, DType u) =>
                                             ArgsHMap "__backward__contrib_DeformablePSROIPooling"
                                               '()
                                               a
                                               -> TensorApply (t u)
__backward__contrib_DeformablePSROIPooling args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_DeformablePSROIPooling" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__backward__contrib_ModulatedDeformableConvolution"
       '()
     = '[]

__backward__contrib_ModulatedDeformableConvolution ::
                                                   forall a t u .
                                                     (Tensor t,
                                                      Fullfilled
                                                        "__backward__contrib_ModulatedDeformableConvolution"
                                                        '()
                                                        a,
                                                      HasCallStack, DType u) =>
                                                     ArgsHMap
                                                       "__backward__contrib_ModulatedDeformableConvolution"
                                                       '()
                                                       a
                                                       -> TensorApply (t u)
__backward__contrib_ModulatedDeformableConvolution args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_ModulatedDeformableConvolution"
        scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__backward__contrib_MultiBoxDetection" '() = '[]

__backward__contrib_MultiBoxDetection ::
                                      forall a t u .
                                        (Tensor t,
                                         Fullfilled "__backward__contrib_MultiBoxDetection" '() a,
                                         HasCallStack, DType u) =>
                                        ArgsHMap "__backward__contrib_MultiBoxDetection" '() a ->
                                          TensorApply (t u)
__backward__contrib_MultiBoxDetection args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_MultiBoxDetection" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward__contrib_MultiBoxPrior" '()
     = '[]

__backward__contrib_MultiBoxPrior ::
                                  forall a t u .
                                    (Tensor t, Fullfilled "__backward__contrib_MultiBoxPrior" '() a,
                                     HasCallStack, DType u) =>
                                    ArgsHMap "__backward__contrib_MultiBoxPrior" '() a ->
                                      TensorApply (t u)
__backward__contrib_MultiBoxPrior args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_MultiBoxPrior" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__backward__contrib_MultiBoxTarget" '() = '[]

__backward__contrib_MultiBoxTarget ::
                                   forall a t u .
                                     (Tensor t,
                                      Fullfilled "__backward__contrib_MultiBoxTarget" '() a,
                                      HasCallStack, DType u) =>
                                     ArgsHMap "__backward__contrib_MultiBoxTarget" '() a ->
                                       TensorApply (t u)
__backward__contrib_MultiBoxTarget args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_MultiBoxTarget" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward__contrib_MultiProposal" '()
     = '[]

__backward__contrib_MultiProposal ::
                                  forall a t u .
                                    (Tensor t, Fullfilled "__backward__contrib_MultiProposal" '() a,
                                     HasCallStack, DType u) =>
                                    ArgsHMap "__backward__contrib_MultiProposal" '() a ->
                                      TensorApply (t u)
__backward__contrib_MultiProposal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_MultiProposal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward__contrib_PSROIPooling" '()
     = '[]

__backward__contrib_PSROIPooling ::
                                 forall a t u .
                                   (Tensor t, Fullfilled "__backward__contrib_PSROIPooling" '() a,
                                    HasCallStack, DType u) =>
                                   ArgsHMap "__backward__contrib_PSROIPooling" '() a ->
                                     TensorApply (t u)
__backward__contrib_PSROIPooling args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_PSROIPooling" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward__contrib_Proposal" '() =
     '[]

__backward__contrib_Proposal ::
                             forall a t u .
                               (Tensor t, Fullfilled "__backward__contrib_Proposal" '() a,
                                HasCallStack, DType u) =>
                               ArgsHMap "__backward__contrib_Proposal" '() a -> TensorApply (t u)
__backward__contrib_Proposal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_Proposal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward__contrib_SyncBatchNorm" '()
     = '[]

__backward__contrib_SyncBatchNorm ::
                                  forall a t u .
                                    (Tensor t, Fullfilled "__backward__contrib_SyncBatchNorm" '() a,
                                     HasCallStack, DType u) =>
                                    ArgsHMap "__backward__contrib_SyncBatchNorm" '() a ->
                                      TensorApply (t u)
__backward__contrib_SyncBatchNorm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_SyncBatchNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward__contrib_count_sketch" '()
     = '[]

__backward__contrib_count_sketch ::
                                 forall a t u .
                                   (Tensor t, Fullfilled "__backward__contrib_count_sketch" '() a,
                                    HasCallStack, DType u) =>
                                   ArgsHMap "__backward__contrib_count_sketch" '() a ->
                                     TensorApply (t u)
__backward__contrib_count_sketch args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_count_sketch" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward__contrib_fft" '() = '[]

__backward__contrib_fft ::
                        forall a t u .
                          (Tensor t, Fullfilled "__backward__contrib_fft" '() a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__backward__contrib_fft" '() a -> TensorApply (t u)
__backward__contrib_fft args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_fft" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward__contrib_ifft" '() = '[]

__backward__contrib_ifft ::
                         forall a t u .
                           (Tensor t, Fullfilled "__backward__contrib_ifft" '() a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__backward__contrib_ifft" '() a -> TensorApply (t u)
__backward__contrib_ifft args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward__contrib_ifft" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_abs" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_abs ::
               forall a t u .
                 (Tensor t, Fullfilled "__backward_abs" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__backward_abs" '(t, u) a -> TensorApply (t u)
__backward_abs args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_abs" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_add" '() = '[]

__backward_add ::
               forall a t u .
                 (Tensor t, Fullfilled "__backward_add" '() a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__backward_add" '() a -> TensorApply (t u)
__backward_add args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_add" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_amp_cast" '() = '[]

__backward_amp_cast ::
                    forall a t u .
                      (Tensor t, Fullfilled "__backward_amp_cast" '() a, HasCallStack,
                       DType u) =>
                      ArgsHMap "__backward_amp_cast" '() a -> TensorApply (t u)
__backward_amp_cast args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_amp_cast" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_amp_multicast" '(t, u) =
     '[ '("num_outputs", AttrReq Int), '("cast_narrow", AttrOpt Bool),
        '("grad", AttrOpt [t u])]

__backward_amp_multicast ::
                         forall a t u .
                           (Tensor t, Fullfilled "__backward_amp_multicast" '(t, u) a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__backward_amp_multicast" '(t, u) a -> TensorApply (t u)
__backward_amp_multicast args
  = let scalarArgs
          = catMaybes
              [("num_outputs",) . showValue <$>
                 (args !? #num_outputs :: Maybe Int),
               ("cast_narrow",) . showValue <$>
                 (args !? #cast_narrow :: Maybe Bool)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #grad :: Maybe [RawTensor t])
      in
      applyRaw "_backward_amp_multicast" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__backward_arccos" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_arccos ::
                  forall a t u .
                    (Tensor t, Fullfilled "__backward_arccos" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__backward_arccos" '(t, u) a -> TensorApply (t u)
__backward_arccos args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_arccos" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_arccosh" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_arccosh ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_arccosh" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_arccosh" '(t, u) a -> TensorApply (t u)
__backward_arccosh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_arccosh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_arcsin" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_arcsin ::
                  forall a t u .
                    (Tensor t, Fullfilled "__backward_arcsin" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__backward_arcsin" '(t, u) a -> TensorApply (t u)
__backward_arcsin args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_arcsin" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_arcsinh" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_arcsinh ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_arcsinh" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_arcsinh" '(t, u) a -> TensorApply (t u)
__backward_arcsinh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_arcsinh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_arctan" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_arctan ::
                  forall a t u .
                    (Tensor t, Fullfilled "__backward_arctan" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__backward_arctan" '(t, u) a -> TensorApply (t u)
__backward_arctan args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_arctan" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_arctanh" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_arctanh ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_arctanh" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_arctanh" '(t, u) a -> TensorApply (t u)
__backward_arctanh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_arctanh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__backward_backward_FullyConnected" '() = '[]

__backward_backward_FullyConnected ::
                                   forall a t u .
                                     (Tensor t,
                                      Fullfilled "__backward_backward_FullyConnected" '() a,
                                      HasCallStack, DType u) =>
                                     ArgsHMap "__backward_backward_FullyConnected" '() a ->
                                       TensorApply (t u)
__backward_backward_FullyConnected args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_backward_FullyConnected" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_broadcast_add" '() = '[]

__backward_broadcast_add ::
                         forall a t u .
                           (Tensor t, Fullfilled "__backward_broadcast_add" '() a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__backward_broadcast_add" '() a -> TensorApply (t u)
__backward_broadcast_add args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_add" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_broadcast_div" '() = '[]

__backward_broadcast_div ::
                         forall a t u .
                           (Tensor t, Fullfilled "__backward_broadcast_div" '() a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__backward_broadcast_div" '() a -> TensorApply (t u)
__backward_broadcast_div args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_div" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_broadcast_exponential" '()
     =
     '[ '("scale", AttrOpt (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text)]

__backward_broadcast_exponential ::
                                 forall a t u .
                                   (Tensor t, Fullfilled "__backward_broadcast_exponential" '() a,
                                    HasCallStack, DType u) =>
                                   ArgsHMap "__backward_broadcast_exponential" '() a ->
                                     TensorApply (t u)
__backward_broadcast_exponential args
  = let scalarArgs
          = catMaybes
              [("scale",) . showValue <$>
                 (args !? #scale :: Maybe (Maybe Float)),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int])),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text)]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_exponential" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_broadcast_gumbel" '() =
     '[ '("loc", AttrReq (Maybe Float)),
        '("scale", AttrReq (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text)]

__backward_broadcast_gumbel ::
                            forall a t u .
                              (Tensor t, Fullfilled "__backward_broadcast_gumbel" '() a,
                               HasCallStack, DType u) =>
                              ArgsHMap "__backward_broadcast_gumbel" '() a -> TensorApply (t u)
__backward_broadcast_gumbel args
  = let scalarArgs
          = catMaybes
              [("loc",) . showValue <$> (args !? #loc :: Maybe (Maybe Float)),
               ("scale",) . showValue <$> (args !? #scale :: Maybe (Maybe Float)),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int])),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text)]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_gumbel" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_broadcast_hypot" '() = '[]

__backward_broadcast_hypot ::
                           forall a t u .
                             (Tensor t, Fullfilled "__backward_broadcast_hypot" '() a,
                              HasCallStack, DType u) =>
                             ArgsHMap "__backward_broadcast_hypot" '() a -> TensorApply (t u)
__backward_broadcast_hypot args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_hypot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_broadcast_logistic" '() =
     '[ '("loc", AttrReq (Maybe Float)),
        '("scale", AttrReq (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text)]

__backward_broadcast_logistic ::
                              forall a t u .
                                (Tensor t, Fullfilled "__backward_broadcast_logistic" '() a,
                                 HasCallStack, DType u) =>
                                ArgsHMap "__backward_broadcast_logistic" '() a -> TensorApply (t u)
__backward_broadcast_logistic args
  = let scalarArgs
          = catMaybes
              [("loc",) . showValue <$> (args !? #loc :: Maybe (Maybe Float)),
               ("scale",) . showValue <$> (args !? #scale :: Maybe (Maybe Float)),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int])),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text)]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_logistic" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_broadcast_maximum" '() =
     '[]

__backward_broadcast_maximum ::
                             forall a t u .
                               (Tensor t, Fullfilled "__backward_broadcast_maximum" '() a,
                                HasCallStack, DType u) =>
                               ArgsHMap "__backward_broadcast_maximum" '() a -> TensorApply (t u)
__backward_broadcast_maximum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_maximum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_broadcast_minimum" '() =
     '[]

__backward_broadcast_minimum ::
                             forall a t u .
                               (Tensor t, Fullfilled "__backward_broadcast_minimum" '() a,
                                HasCallStack, DType u) =>
                               ArgsHMap "__backward_broadcast_minimum" '() a -> TensorApply (t u)
__backward_broadcast_minimum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_minimum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_broadcast_mod" '() = '[]

__backward_broadcast_mod ::
                         forall a t u .
                           (Tensor t, Fullfilled "__backward_broadcast_mod" '() a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__backward_broadcast_mod" '() a -> TensorApply (t u)
__backward_broadcast_mod args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_mod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_broadcast_mul" '() = '[]

__backward_broadcast_mul ::
                         forall a t u .
                           (Tensor t, Fullfilled "__backward_broadcast_mul" '() a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__backward_broadcast_mul" '() a -> TensorApply (t u)
__backward_broadcast_mul args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_mul" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_broadcast_normal" '() =
     '[ '("loc", AttrReq (Maybe Float)),
        '("scale", AttrReq (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text),
        '("dtype", AttrOpt (EnumType '["float16", "float32", "float64"]))]

__backward_broadcast_normal ::
                            forall a t u .
                              (Tensor t, Fullfilled "__backward_broadcast_normal" '() a,
                               HasCallStack, DType u) =>
                              ArgsHMap "__backward_broadcast_normal" '() a -> TensorApply (t u)
__backward_broadcast_normal args
  = let scalarArgs
          = catMaybes
              [("loc",) . showValue <$> (args !? #loc :: Maybe (Maybe Float)),
               ("scale",) . showValue <$> (args !? #scale :: Maybe (Maybe Float)),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int])),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["float16", "float32", "float64"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_normal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_broadcast_pareto" '() =
     '[ '("a", AttrOpt (Maybe Float)), '("size", AttrOpt (Maybe [Int])),
        '("ctx", AttrOpt Text)]

__backward_broadcast_pareto ::
                            forall a t u .
                              (Tensor t, Fullfilled "__backward_broadcast_pareto" '() a,
                               HasCallStack, DType u) =>
                              ArgsHMap "__backward_broadcast_pareto" '() a -> TensorApply (t u)
__backward_broadcast_pareto args
  = let scalarArgs
          = catMaybes
              [("a",) . showValue <$> (args !? #a :: Maybe (Maybe Float)),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int])),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text)]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_pareto" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_broadcast_power" '() = '[]

__backward_broadcast_power ::
                           forall a t u .
                             (Tensor t, Fullfilled "__backward_broadcast_power" '() a,
                              HasCallStack, DType u) =>
                             ArgsHMap "__backward_broadcast_power" '() a -> TensorApply (t u)
__backward_broadcast_power args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_power" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_broadcast_rayleigh" '() =
     '[ '("scale", AttrOpt (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text)]

__backward_broadcast_rayleigh ::
                              forall a t u .
                                (Tensor t, Fullfilled "__backward_broadcast_rayleigh" '() a,
                                 HasCallStack, DType u) =>
                                ArgsHMap "__backward_broadcast_rayleigh" '() a -> TensorApply (t u)
__backward_broadcast_rayleigh args
  = let scalarArgs
          = catMaybes
              [("scale",) . showValue <$>
                 (args !? #scale :: Maybe (Maybe Float)),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int])),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text)]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_rayleigh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_broadcast_sub" '() = '[]

__backward_broadcast_sub ::
                         forall a t u .
                           (Tensor t, Fullfilled "__backward_broadcast_sub" '() a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__backward_broadcast_sub" '() a -> TensorApply (t u)
__backward_broadcast_sub args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_sub" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_broadcast_weibull" '() =
     '[ '("a", AttrOpt (Maybe Float)), '("size", AttrOpt (Maybe [Int])),
        '("ctx", AttrOpt Text)]

__backward_broadcast_weibull ::
                             forall a t u .
                               (Tensor t, Fullfilled "__backward_broadcast_weibull" '() a,
                                HasCallStack, DType u) =>
                               ArgsHMap "__backward_broadcast_weibull" '() a -> TensorApply (t u)
__backward_broadcast_weibull args
  = let scalarArgs
          = catMaybes
              [("a",) . showValue <$> (args !? #a :: Maybe (Maybe Float)),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int])),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text)]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_broadcast_weibull" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_cast" '() = '[]

__backward_cast ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_cast" '() a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_cast" '() a -> TensorApply (t u)
__backward_cast args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_cast" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_cbrt" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_cbrt ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_cbrt" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_cbrt" '(t, u) a -> TensorApply (t u)
__backward_cbrt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_cbrt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_clip" '() = '[]

__backward_clip ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_clip" '() a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_clip" '() a -> TensorApply (t u)
__backward_clip args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_clip" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_col2im" '() = '[]

__backward_col2im ::
                  forall a t u .
                    (Tensor t, Fullfilled "__backward_col2im" '() a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__backward_col2im" '() a -> TensorApply (t u)
__backward_col2im args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_col2im" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_cond" '() = '[]

__backward_cond ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_cond" '() a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_cond" '() a -> TensorApply (t u)
__backward_cond args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_cond" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__backward_contrib_AdaptiveAvgPooling2D" '() = '[]

__backward_contrib_AdaptiveAvgPooling2D ::
                                        forall a t u .
                                          (Tensor t,
                                           Fullfilled "__backward_contrib_AdaptiveAvgPooling2D" '()
                                             a,
                                           HasCallStack, DType u) =>
                                          ArgsHMap "__backward_contrib_AdaptiveAvgPooling2D" '() a
                                            -> TensorApply (t u)
__backward_contrib_AdaptiveAvgPooling2D args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_contrib_AdaptiveAvgPooling2D" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__backward_contrib_BatchNormWithReLU" '() = '[]

__backward_contrib_BatchNormWithReLU ::
                                     forall a t u .
                                       (Tensor t,
                                        Fullfilled "__backward_contrib_BatchNormWithReLU" '() a,
                                        HasCallStack, DType u) =>
                                       ArgsHMap "__backward_contrib_BatchNormWithReLU" '() a ->
                                         TensorApply (t u)
__backward_contrib_BatchNormWithReLU args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_contrib_BatchNormWithReLU" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__backward_contrib_BilinearResize2D" '() = '[]

__backward_contrib_BilinearResize2D ::
                                    forall a t u .
                                      (Tensor t,
                                       Fullfilled "__backward_contrib_BilinearResize2D" '() a,
                                       HasCallStack, DType u) =>
                                      ArgsHMap "__backward_contrib_BilinearResize2D" '() a ->
                                        TensorApply (t u)
__backward_contrib_BilinearResize2D args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_contrib_BilinearResize2D" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__backward_contrib_bipartite_matching" '() =
     '[ '("is_ascend", AttrOpt Bool), '("threshold", AttrReq Float),
        '("topk", AttrOpt Int)]

__backward_contrib_bipartite_matching ::
                                      forall a t u .
                                        (Tensor t,
                                         Fullfilled "__backward_contrib_bipartite_matching" '() a,
                                         HasCallStack, DType u) =>
                                        ArgsHMap "__backward_contrib_bipartite_matching" '() a ->
                                          TensorApply (t u)
__backward_contrib_bipartite_matching args
  = let scalarArgs
          = catMaybes
              [("is_ascend",) . showValue <$> (args !? #is_ascend :: Maybe Bool),
               ("threshold",) . showValue <$> (args !? #threshold :: Maybe Float),
               ("topk",) . showValue <$> (args !? #topk :: Maybe Int)]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_contrib_bipartite_matching" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_contrib_boolean_mask" '() =
     '[ '("axis", AttrOpt Int)]

__backward_contrib_boolean_mask ::
                                forall a t u .
                                  (Tensor t, Fullfilled "__backward_contrib_boolean_mask" '() a,
                                   HasCallStack, DType u) =>
                                  ArgsHMap "__backward_contrib_boolean_mask" '() a ->
                                    TensorApply (t u)
__backward_contrib_boolean_mask args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_contrib_boolean_mask" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_contrib_box_iou" '() =
     '[ '("format", AttrOpt (EnumType '["center", "corner"]))]

__backward_contrib_box_iou ::
                           forall a t u .
                             (Tensor t, Fullfilled "__backward_contrib_box_iou" '() a,
                              HasCallStack, DType u) =>
                             ArgsHMap "__backward_contrib_box_iou" '() a -> TensorApply (t u)
__backward_contrib_box_iou args
  = let scalarArgs
          = catMaybes
              [("format",) . showValue <$>
                 (args !? #format :: Maybe (EnumType '["center", "corner"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_contrib_box_iou" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_contrib_box_nms" '() =
     '[ '("overlap_thresh", AttrOpt Float),
        '("valid_thresh", AttrOpt Float), '("topk", AttrOpt Int),
        '("coord_start", AttrOpt Int), '("score_index", AttrOpt Int),
        '("id_index", AttrOpt Int), '("background_id", AttrOpt Int),
        '("force_suppress", AttrOpt Bool),
        '("in_format", AttrOpt (EnumType '["center", "corner"])),
        '("out_format", AttrOpt (EnumType '["center", "corner"]))]

__backward_contrib_box_nms ::
                           forall a t u .
                             (Tensor t, Fullfilled "__backward_contrib_box_nms" '() a,
                              HasCallStack, DType u) =>
                             ArgsHMap "__backward_contrib_box_nms" '() a -> TensorApply (t u)
__backward_contrib_box_nms args
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
               ("background_id",) . showValue <$>
                 (args !? #background_id :: Maybe Int),
               ("force_suppress",) . showValue <$>
                 (args !? #force_suppress :: Maybe Bool),
               ("in_format",) . showValue <$>
                 (args !? #in_format :: Maybe (EnumType '["center", "corner"])),
               ("out_format",) . showValue <$>
                 (args !? #out_format :: Maybe (EnumType '["center", "corner"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_contrib_box_nms" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_copy" '() = '[]

__backward_copy ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_copy" '() a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_copy" '() a -> TensorApply (t u)
__backward_copy args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_copy" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_cos" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_cos ::
               forall a t u .
                 (Tensor t, Fullfilled "__backward_cos" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__backward_cos" '(t, u) a -> TensorApply (t u)
__backward_cos args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_cos" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_cosh" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_cosh ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_cosh" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_cosh" '(t, u) a -> TensorApply (t u)
__backward_cosh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_cosh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_ctc_loss" '() = '[]

__backward_ctc_loss ::
                    forall a t u .
                      (Tensor t, Fullfilled "__backward_ctc_loss" '() a, HasCallStack,
                       DType u) =>
                      ArgsHMap "__backward_ctc_loss" '() a -> TensorApply (t u)
__backward_ctc_loss args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_ctc_loss" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_degrees" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_degrees ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_degrees" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_degrees" '(t, u) a -> TensorApply (t u)
__backward_degrees args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_degrees" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_diag" '() = '[]

__backward_diag ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_diag" '() a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_diag" '() a -> TensorApply (t u)
__backward_diag args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_diag" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_div" '() = '[]

__backward_div ::
               forall a t u .
                 (Tensor t, Fullfilled "__backward_div" '() a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__backward_div" '() a -> TensorApply (t u)
__backward_div args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_div" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_div_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__backward_div_scalar ::
                      forall a t u .
                        (Tensor t, Fullfilled "__backward_div_scalar" '(t, u) a,
                         HasCallStack, DType u) =>
                        ArgsHMap "__backward_div_scalar" '(t, u) a -> TensorApply (t u)
__backward_div_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_backward_div_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_dot" '() =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("forward_stype",
          AttrOpt (Maybe (EnumType '["csr", "default", "row_sparse"])))]

__backward_dot ::
               forall a t u .
                 (Tensor t, Fullfilled "__backward_dot" '() a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__backward_dot" '() a -> TensorApply (t u)
__backward_dot args
  = let scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$>
                 (args !? #transpose_a :: Maybe Bool),
               ("transpose_b",) . showValue <$>
                 (args !? #transpose_b :: Maybe Bool),
               ("forward_stype",) . showValue <$>
                 (args !? #forward_stype ::
                    Maybe (Maybe (EnumType '["csr", "default", "row_sparse"])))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_dot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_erf" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_erf ::
               forall a t u .
                 (Tensor t, Fullfilled "__backward_erf" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__backward_erf" '(t, u) a -> TensorApply (t u)
__backward_erf args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_erf" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_erfinv" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_erfinv ::
                  forall a t u .
                    (Tensor t, Fullfilled "__backward_erfinv" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__backward_erfinv" '(t, u) a -> TensorApply (t u)
__backward_erfinv args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_erfinv" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_expm1" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_expm1 ::
                 forall a t u .
                   (Tensor t, Fullfilled "__backward_expm1" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__backward_expm1" '(t, u) a -> TensorApply (t u)
__backward_expm1 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_expm1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_foreach" '() = '[]

__backward_foreach ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_foreach" '() a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_foreach" '() a -> TensorApply (t u)
__backward_foreach args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_foreach" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_gamma" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_gamma ::
                 forall a t u .
                   (Tensor t, Fullfilled "__backward_gamma" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__backward_gamma" '(t, u) a -> TensorApply (t u)
__backward_gamma args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_gamma" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_gammaln" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_gammaln ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_gammaln" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_gammaln" '(t, u) a -> TensorApply (t u)
__backward_gammaln args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_gammaln" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_gather_nd" '(t, u) =
     '[ '("shape", AttrReq [Int]), '("data", AttrOpt (t u)),
        '("indices", AttrOpt (t u))]

__backward_gather_nd ::
                     forall a t u .
                       (Tensor t, Fullfilled "__backward_gather_nd" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "__backward_gather_nd" '(t, u) a -> TensorApply (t u)
__backward_gather_nd args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("indices",) . toRaw <$> (args !? #indices :: Maybe (t u))]
      in
      applyRaw "_backward_gather_nd" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_hard_sigmoid" '() = '[]

__backward_hard_sigmoid ::
                        forall a t u .
                          (Tensor t, Fullfilled "__backward_hard_sigmoid" '() a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__backward_hard_sigmoid" '() a -> TensorApply (t u)
__backward_hard_sigmoid args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_hard_sigmoid" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_hypot" '() = '[]

__backward_hypot ::
                 forall a t u .
                   (Tensor t, Fullfilled "__backward_hypot" '() a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__backward_hypot" '() a -> TensorApply (t u)
__backward_hypot args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_hypot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_hypot_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_hypot_scalar ::
                        forall a t u .
                          (Tensor t, Fullfilled "__backward_hypot_scalar" '(t, u) a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__backward_hypot_scalar" '(t, u) a -> TensorApply (t u)
__backward_hypot_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_hypot_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_im2col" '() = '[]

__backward_im2col ::
                  forall a t u .
                    (Tensor t, Fullfilled "__backward_im2col" '() a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__backward_im2col" '() a -> TensorApply (t u)
__backward_im2col args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_im2col" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_image_crop" '() = '[]

__backward_image_crop ::
                      forall a t u .
                        (Tensor t, Fullfilled "__backward_image_crop" '() a, HasCallStack,
                         DType u) =>
                        ArgsHMap "__backward_image_crop" '() a -> TensorApply (t u)
__backward_image_crop args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_image_crop" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_image_normalize" '() = '[]

__backward_image_normalize ::
                           forall a t u .
                             (Tensor t, Fullfilled "__backward_image_normalize" '() a,
                              HasCallStack, DType u) =>
                             ArgsHMap "__backward_image_normalize" '() a -> TensorApply (t u)
__backward_image_normalize args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_image_normalize" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__backward_interleaved_matmul_encdec_qk" '() = '[]

__backward_interleaved_matmul_encdec_qk ::
                                        forall a t u .
                                          (Tensor t,
                                           Fullfilled "__backward_interleaved_matmul_encdec_qk" '()
                                             a,
                                           HasCallStack, DType u) =>
                                          ArgsHMap "__backward_interleaved_matmul_encdec_qk" '() a
                                            -> TensorApply (t u)
__backward_interleaved_matmul_encdec_qk args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_interleaved_matmul_encdec_qk" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__backward_interleaved_matmul_encdec_valatt" '() =
     '[]

__backward_interleaved_matmul_encdec_valatt ::
                                            forall a t u .
                                              (Tensor t,
                                               Fullfilled
                                                 "__backward_interleaved_matmul_encdec_valatt"
                                                 '()
                                                 a,
                                               HasCallStack, DType u) =>
                                              ArgsHMap "__backward_interleaved_matmul_encdec_valatt"
                                                '()
                                                a
                                                -> TensorApply (t u)
__backward_interleaved_matmul_encdec_valatt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_interleaved_matmul_encdec_valatt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__backward_interleaved_matmul_selfatt_qk" '() = '[]

__backward_interleaved_matmul_selfatt_qk ::
                                         forall a t u .
                                           (Tensor t,
                                            Fullfilled "__backward_interleaved_matmul_selfatt_qk"
                                              '()
                                              a,
                                            HasCallStack, DType u) =>
                                           ArgsHMap "__backward_interleaved_matmul_selfatt_qk" '() a
                                             -> TensorApply (t u)
__backward_interleaved_matmul_selfatt_qk args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_interleaved_matmul_selfatt_qk" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__backward_interleaved_matmul_selfatt_valatt" '() =
     '[]

__backward_interleaved_matmul_selfatt_valatt ::
                                             forall a t u .
                                               (Tensor t,
                                                Fullfilled
                                                  "__backward_interleaved_matmul_selfatt_valatt"
                                                  '()
                                                  a,
                                                HasCallStack, DType u) =>
                                               ArgsHMap
                                                 "__backward_interleaved_matmul_selfatt_valatt"
                                                 '()
                                                 a
                                                 -> TensorApply (t u)
__backward_interleaved_matmul_selfatt_valatt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_interleaved_matmul_selfatt_valatt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_linalg_det" '() = '[]

__backward_linalg_det ::
                      forall a t u .
                        (Tensor t, Fullfilled "__backward_linalg_det" '() a, HasCallStack,
                         DType u) =>
                        ArgsHMap "__backward_linalg_det" '() a -> TensorApply (t u)
__backward_linalg_det args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_det" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_linalg_extractdiag" '() =
     '[]

__backward_linalg_extractdiag ::
                              forall a t u .
                                (Tensor t, Fullfilled "__backward_linalg_extractdiag" '() a,
                                 HasCallStack, DType u) =>
                                ArgsHMap "__backward_linalg_extractdiag" '() a -> TensorApply (t u)
__backward_linalg_extractdiag args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_extractdiag" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_linalg_extracttrian" '() =
     '[]

__backward_linalg_extracttrian ::
                               forall a t u .
                                 (Tensor t, Fullfilled "__backward_linalg_extracttrian" '() a,
                                  HasCallStack, DType u) =>
                                 ArgsHMap "__backward_linalg_extracttrian" '() a ->
                                   TensorApply (t u)
__backward_linalg_extracttrian args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_extracttrian" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_linalg_gelqf" '() = '[]

__backward_linalg_gelqf ::
                        forall a t u .
                          (Tensor t, Fullfilled "__backward_linalg_gelqf" '() a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__backward_linalg_gelqf" '() a -> TensorApply (t u)
__backward_linalg_gelqf args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_gelqf" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_linalg_gemm" '() = '[]

__backward_linalg_gemm ::
                       forall a t u .
                         (Tensor t, Fullfilled "__backward_linalg_gemm" '() a, HasCallStack,
                          DType u) =>
                         ArgsHMap "__backward_linalg_gemm" '() a -> TensorApply (t u)
__backward_linalg_gemm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_gemm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_linalg_gemm2" '() = '[]

__backward_linalg_gemm2 ::
                        forall a t u .
                          (Tensor t, Fullfilled "__backward_linalg_gemm2" '() a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__backward_linalg_gemm2" '() a -> TensorApply (t u)
__backward_linalg_gemm2 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_gemm2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_linalg_inverse" '() = '[]

__backward_linalg_inverse ::
                          forall a t u .
                            (Tensor t, Fullfilled "__backward_linalg_inverse" '() a,
                             HasCallStack, DType u) =>
                            ArgsHMap "__backward_linalg_inverse" '() a -> TensorApply (t u)
__backward_linalg_inverse args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_inverse" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_linalg_makediag" '() = '[]

__backward_linalg_makediag ::
                           forall a t u .
                             (Tensor t, Fullfilled "__backward_linalg_makediag" '() a,
                              HasCallStack, DType u) =>
                             ArgsHMap "__backward_linalg_makediag" '() a -> TensorApply (t u)
__backward_linalg_makediag args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_makediag" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_linalg_maketrian" '() = '[]

__backward_linalg_maketrian ::
                            forall a t u .
                              (Tensor t, Fullfilled "__backward_linalg_maketrian" '() a,
                               HasCallStack, DType u) =>
                              ArgsHMap "__backward_linalg_maketrian" '() a -> TensorApply (t u)
__backward_linalg_maketrian args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_maketrian" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_linalg_potrf" '() = '[]

__backward_linalg_potrf ::
                        forall a t u .
                          (Tensor t, Fullfilled "__backward_linalg_potrf" '() a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__backward_linalg_potrf" '() a -> TensorApply (t u)
__backward_linalg_potrf args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_potrf" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_linalg_potri" '() = '[]

__backward_linalg_potri ::
                        forall a t u .
                          (Tensor t, Fullfilled "__backward_linalg_potri" '() a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__backward_linalg_potri" '() a -> TensorApply (t u)
__backward_linalg_potri args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_potri" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_linalg_slogdet" '() = '[]

__backward_linalg_slogdet ::
                          forall a t u .
                            (Tensor t, Fullfilled "__backward_linalg_slogdet" '() a,
                             HasCallStack, DType u) =>
                            ArgsHMap "__backward_linalg_slogdet" '() a -> TensorApply (t u)
__backward_linalg_slogdet args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_slogdet" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_linalg_sumlogdiag" '() =
     '[]

__backward_linalg_sumlogdiag ::
                             forall a t u .
                               (Tensor t, Fullfilled "__backward_linalg_sumlogdiag" '() a,
                                HasCallStack, DType u) =>
                               ArgsHMap "__backward_linalg_sumlogdiag" '() a -> TensorApply (t u)
__backward_linalg_sumlogdiag args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_sumlogdiag" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_linalg_syevd" '() = '[]

__backward_linalg_syevd ::
                        forall a t u .
                          (Tensor t, Fullfilled "__backward_linalg_syevd" '() a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__backward_linalg_syevd" '() a -> TensorApply (t u)
__backward_linalg_syevd args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_syevd" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_linalg_syrk" '() = '[]

__backward_linalg_syrk ::
                       forall a t u .
                         (Tensor t, Fullfilled "__backward_linalg_syrk" '() a, HasCallStack,
                          DType u) =>
                         ArgsHMap "__backward_linalg_syrk" '() a -> TensorApply (t u)
__backward_linalg_syrk args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_syrk" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_linalg_trmm" '() = '[]

__backward_linalg_trmm ::
                       forall a t u .
                         (Tensor t, Fullfilled "__backward_linalg_trmm" '() a, HasCallStack,
                          DType u) =>
                         ArgsHMap "__backward_linalg_trmm" '() a -> TensorApply (t u)
__backward_linalg_trmm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_trmm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_linalg_trsm" '() = '[]

__backward_linalg_trsm ::
                       forall a t u .
                         (Tensor t, Fullfilled "__backward_linalg_trsm" '() a, HasCallStack,
                          DType u) =>
                         ArgsHMap "__backward_linalg_trsm" '() a -> TensorApply (t u)
__backward_linalg_trsm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linalg_trsm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_linear_reg_out" '() = '[]

__backward_linear_reg_out ::
                          forall a t u .
                            (Tensor t, Fullfilled "__backward_linear_reg_out" '() a,
                             HasCallStack, DType u) =>
                            ArgsHMap "__backward_linear_reg_out" '() a -> TensorApply (t u)
__backward_linear_reg_out args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_linear_reg_out" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_log" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_log ::
               forall a t u .
                 (Tensor t, Fullfilled "__backward_log" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__backward_log" '(t, u) a -> TensorApply (t u)
__backward_log args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_log" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_log10" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_log10 ::
                 forall a t u .
                   (Tensor t, Fullfilled "__backward_log10" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__backward_log10" '(t, u) a -> TensorApply (t u)
__backward_log10 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_log10" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_log1p" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_log1p ::
                 forall a t u .
                   (Tensor t, Fullfilled "__backward_log1p" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__backward_log1p" '(t, u) a -> TensorApply (t u)
__backward_log1p args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_log1p" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_log2" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_log2 ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_log2" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_log2" '(t, u) a -> TensorApply (t u)
__backward_log2 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_log2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_log_softmax" '(t, u) =
     '[ '("args", AttrOpt [t u])]

__backward_log_softmax ::
                       forall a t u .
                         (Tensor t, Fullfilled "__backward_log_softmax" '(t, u) a,
                          HasCallStack, DType u) =>
                         ArgsHMap "__backward_log_softmax" '(t, u) a -> TensorApply (t u)
__backward_log_softmax args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #args :: Maybe [RawTensor t])
      in
      applyRaw "_backward_log_softmax" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__backward_logistic_reg_out" '() = '[]

__backward_logistic_reg_out ::
                            forall a t u .
                              (Tensor t, Fullfilled "__backward_logistic_reg_out" '() a,
                               HasCallStack, DType u) =>
                              ArgsHMap "__backward_logistic_reg_out" '() a -> TensorApply (t u)
__backward_logistic_reg_out args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_logistic_reg_out" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_mae_reg_out" '() = '[]

__backward_mae_reg_out ::
                       forall a t u .
                         (Tensor t, Fullfilled "__backward_mae_reg_out" '() a, HasCallStack,
                          DType u) =>
                         ArgsHMap "__backward_mae_reg_out" '() a -> TensorApply (t u)
__backward_mae_reg_out args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_mae_reg_out" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_max" '() = '[]

__backward_max ::
               forall a t u .
                 (Tensor t, Fullfilled "__backward_max" '() a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__backward_max" '() a -> TensorApply (t u)
__backward_max args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_max" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_maximum" '() = '[]

__backward_maximum ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_maximum" '() a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_maximum" '() a -> TensorApply (t u)
__backward_maximum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_maximum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_maximum_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_maximum_scalar ::
                          forall a t u .
                            (Tensor t, Fullfilled "__backward_maximum_scalar" '(t, u) a,
                             HasCallStack, DType u) =>
                            ArgsHMap "__backward_maximum_scalar" '(t, u) a -> TensorApply (t u)
__backward_maximum_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_maximum_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_mean" '() = '[]

__backward_mean ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_mean" '() a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_mean" '() a -> TensorApply (t u)
__backward_mean args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_mean" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_min" '() = '[]

__backward_min ::
               forall a t u .
                 (Tensor t, Fullfilled "__backward_min" '() a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__backward_min" '() a -> TensorApply (t u)
__backward_min args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_min" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_minimum" '() = '[]

__backward_minimum ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_minimum" '() a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_minimum" '() a -> TensorApply (t u)
__backward_minimum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_minimum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_minimum_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_minimum_scalar ::
                          forall a t u .
                            (Tensor t, Fullfilled "__backward_minimum_scalar" '(t, u) a,
                             HasCallStack, DType u) =>
                            ArgsHMap "__backward_minimum_scalar" '(t, u) a -> TensorApply (t u)
__backward_minimum_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_minimum_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_mod" '() = '[]

__backward_mod ::
               forall a t u .
                 (Tensor t, Fullfilled "__backward_mod" '() a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__backward_mod" '() a -> TensorApply (t u)
__backward_mod args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_mod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_mod_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_mod_scalar ::
                      forall a t u .
                        (Tensor t, Fullfilled "__backward_mod_scalar" '(t, u) a,
                         HasCallStack, DType u) =>
                        ArgsHMap "__backward_mod_scalar" '(t, u) a -> TensorApply (t u)
__backward_mod_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_mod_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_moments" '() = '[]

__backward_moments ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_moments" '() a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_moments" '() a -> TensorApply (t u)
__backward_moments args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_moments" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_mul" '() = '[]

__backward_mul ::
               forall a t u .
                 (Tensor t, Fullfilled "__backward_mul" '() a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__backward_mul" '() a -> TensorApply (t u)
__backward_mul args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_mul" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_mul_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__backward_mul_scalar ::
                      forall a t u .
                        (Tensor t, Fullfilled "__backward_mul_scalar" '(t, u) a,
                         HasCallStack, DType u) =>
                        ArgsHMap "__backward_mul_scalar" '(t, u) a -> TensorApply (t u)
__backward_mul_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_backward_mul_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_nanprod" '() = '[]

__backward_nanprod ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_nanprod" '() a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_nanprod" '() a -> TensorApply (t u)
__backward_nanprod args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_nanprod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_nansum" '() = '[]

__backward_nansum ::
                  forall a t u .
                    (Tensor t, Fullfilled "__backward_nansum" '() a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__backward_nansum" '() a -> TensorApply (t u)
__backward_nansum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_nansum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_norm" '() = '[]

__backward_norm ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_norm" '() a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_norm" '() a -> TensorApply (t u)
__backward_norm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_norm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_average" '() = '[]

__backward_np_average ::
                      forall a t u .
                        (Tensor t, Fullfilled "__backward_np_average" '() a, HasCallStack,
                         DType u) =>
                        ArgsHMap "__backward_np_average" '() a -> TensorApply (t u)
__backward_np_average args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_average" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_broadcast_to" '() = '[]

__backward_np_broadcast_to ::
                           forall a t u .
                             (Tensor t, Fullfilled "__backward_np_broadcast_to" '() a,
                              HasCallStack, DType u) =>
                             ArgsHMap "__backward_np_broadcast_to" '() a -> TensorApply (t u)
__backward_np_broadcast_to args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_broadcast_to" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_column_stack" '() = '[]

__backward_np_column_stack ::
                           forall a t u .
                             (Tensor t, Fullfilled "__backward_np_column_stack" '() a,
                              HasCallStack, DType u) =>
                             ArgsHMap "__backward_np_column_stack" '() a -> TensorApply (t u)
__backward_np_column_stack args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_column_stack" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_concat" '() = '[]

__backward_np_concat ::
                     forall a t u .
                       (Tensor t, Fullfilled "__backward_np_concat" '() a, HasCallStack,
                        DType u) =>
                       ArgsHMap "__backward_np_concat" '() a -> TensorApply (t u)
__backward_np_concat args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_concat" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_cumsum" '() = '[]

__backward_np_cumsum ::
                     forall a t u .
                       (Tensor t, Fullfilled "__backward_np_cumsum" '() a, HasCallStack,
                        DType u) =>
                       ArgsHMap "__backward_np_cumsum" '() a -> TensorApply (t u)
__backward_np_cumsum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_cumsum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_diag" '() = '[]

__backward_np_diag ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_np_diag" '() a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_np_diag" '() a -> TensorApply (t u)
__backward_np_diag args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_diag" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_diagflat" '() = '[]

__backward_np_diagflat ::
                       forall a t u .
                         (Tensor t, Fullfilled "__backward_np_diagflat" '() a, HasCallStack,
                          DType u) =>
                         ArgsHMap "__backward_np_diagflat" '() a -> TensorApply (t u)
__backward_np_diagflat args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_diagflat" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_diagonal" '() = '[]

__backward_np_diagonal ::
                       forall a t u .
                         (Tensor t, Fullfilled "__backward_np_diagonal" '() a, HasCallStack,
                          DType u) =>
                         ArgsHMap "__backward_np_diagonal" '() a -> TensorApply (t u)
__backward_np_diagonal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_diagonal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_dot" '() = '[]

__backward_np_dot ::
                  forall a t u .
                    (Tensor t, Fullfilled "__backward_np_dot" '() a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__backward_np_dot" '() a -> TensorApply (t u)
__backward_np_dot args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_dot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_dstack" '() = '[]

__backward_np_dstack ::
                     forall a t u .
                       (Tensor t, Fullfilled "__backward_np_dstack" '() a, HasCallStack,
                        DType u) =>
                       ArgsHMap "__backward_np_dstack" '() a -> TensorApply (t u)
__backward_np_dstack args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_dstack" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_hstack" '() = '[]

__backward_np_hstack ::
                     forall a t u .
                       (Tensor t, Fullfilled "__backward_np_hstack" '() a, HasCallStack,
                        DType u) =>
                       ArgsHMap "__backward_np_hstack" '() a -> TensorApply (t u)
__backward_np_hstack args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_hstack" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_matmul" '() = '[]

__backward_np_matmul ::
                     forall a t u .
                       (Tensor t, Fullfilled "__backward_np_matmul" '() a, HasCallStack,
                        DType u) =>
                       ArgsHMap "__backward_np_matmul" '() a -> TensorApply (t u)
__backward_np_matmul args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_matmul" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_max" '() = '[]

__backward_np_max ::
                  forall a t u .
                    (Tensor t, Fullfilled "__backward_np_max" '() a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__backward_np_max" '() a -> TensorApply (t u)
__backward_np_max args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_max" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_mean" '() = '[]

__backward_np_mean ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_np_mean" '() a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_np_mean" '() a -> TensorApply (t u)
__backward_np_mean args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_mean" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_min" '() = '[]

__backward_np_min ::
                  forall a t u .
                    (Tensor t, Fullfilled "__backward_np_min" '() a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__backward_np_min" '() a -> TensorApply (t u)
__backward_np_min args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_min" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_prod" '() = '[]

__backward_np_prod ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_np_prod" '() a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_np_prod" '() a -> TensorApply (t u)
__backward_np_prod args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_prod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_sum" '() = '[]

__backward_np_sum ::
                  forall a t u .
                    (Tensor t, Fullfilled "__backward_np_sum" '() a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__backward_np_sum" '() a -> TensorApply (t u)
__backward_np_sum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_sum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_trace" '() = '[]

__backward_np_trace ::
                    forall a t u .
                      (Tensor t, Fullfilled "__backward_np_trace" '() a, HasCallStack,
                       DType u) =>
                      ArgsHMap "__backward_np_trace" '() a -> TensorApply (t u)
__backward_np_trace args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_trace" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_vstack" '() = '[]

__backward_np_vstack ::
                     forall a t u .
                       (Tensor t, Fullfilled "__backward_np_vstack" '() a, HasCallStack,
                        DType u) =>
                       ArgsHMap "__backward_np_vstack" '() a -> TensorApply (t u)
__backward_np_vstack args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_vstack" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_where" '() = '[]

__backward_np_where ::
                    forall a t u .
                      (Tensor t, Fullfilled "__backward_np_where" '() a, HasCallStack,
                       DType u) =>
                      ArgsHMap "__backward_np_where" '() a -> TensorApply (t u)
__backward_np_where args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_where" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_where_lscalar" '() = '[]

__backward_np_where_lscalar ::
                            forall a t u .
                              (Tensor t, Fullfilled "__backward_np_where_lscalar" '() a,
                               HasCallStack, DType u) =>
                              ArgsHMap "__backward_np_where_lscalar" '() a -> TensorApply (t u)
__backward_np_where_lscalar args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_where_lscalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_np_where_rscalar" '() = '[]

__backward_np_where_rscalar ::
                            forall a t u .
                              (Tensor t, Fullfilled "__backward_np_where_rscalar" '() a,
                               HasCallStack, DType u) =>
                              ArgsHMap "__backward_np_where_rscalar" '() a -> TensorApply (t u)
__backward_np_where_rscalar args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_np_where_rscalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_arctan2" '() = '[]

__backward_npi_arctan2 ::
                       forall a t u .
                         (Tensor t, Fullfilled "__backward_npi_arctan2" '() a, HasCallStack,
                          DType u) =>
                         ArgsHMap "__backward_npi_arctan2" '() a -> TensorApply (t u)
__backward_npi_arctan2 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_arctan2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_arctan2_scalar" '(t, u)
     =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_npi_arctan2_scalar ::
                              forall a t u .
                                (Tensor t, Fullfilled "__backward_npi_arctan2_scalar" '(t, u) a,
                                 HasCallStack, DType u) =>
                                ArgsHMap "__backward_npi_arctan2_scalar" '(t, u) a ->
                                  TensorApply (t u)
__backward_npi_arctan2_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_npi_arctan2_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_broadcast_add" '() =
     '[]

__backward_npi_broadcast_add ::
                             forall a t u .
                               (Tensor t, Fullfilled "__backward_npi_broadcast_add" '() a,
                                HasCallStack, DType u) =>
                               ArgsHMap "__backward_npi_broadcast_add" '() a -> TensorApply (t u)
__backward_npi_broadcast_add args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_broadcast_add" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_broadcast_div" '() =
     '[]

__backward_npi_broadcast_div ::
                             forall a t u .
                               (Tensor t, Fullfilled "__backward_npi_broadcast_div" '() a,
                                HasCallStack, DType u) =>
                               ArgsHMap "__backward_npi_broadcast_div" '() a -> TensorApply (t u)
__backward_npi_broadcast_div args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_broadcast_div" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_broadcast_mod" '() =
     '[]

__backward_npi_broadcast_mod ::
                             forall a t u .
                               (Tensor t, Fullfilled "__backward_npi_broadcast_mod" '() a,
                                HasCallStack, DType u) =>
                               ArgsHMap "__backward_npi_broadcast_mod" '() a -> TensorApply (t u)
__backward_npi_broadcast_mod args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_broadcast_mod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_broadcast_mul" '() =
     '[]

__backward_npi_broadcast_mul ::
                             forall a t u .
                               (Tensor t, Fullfilled "__backward_npi_broadcast_mul" '() a,
                                HasCallStack, DType u) =>
                               ArgsHMap "__backward_npi_broadcast_mul" '() a -> TensorApply (t u)
__backward_npi_broadcast_mul args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_broadcast_mul" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_broadcast_power" '() =
     '[]

__backward_npi_broadcast_power ::
                               forall a t u .
                                 (Tensor t, Fullfilled "__backward_npi_broadcast_power" '() a,
                                  HasCallStack, DType u) =>
                                 ArgsHMap "__backward_npi_broadcast_power" '() a ->
                                   TensorApply (t u)
__backward_npi_broadcast_power args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_broadcast_power" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_broadcast_sub" '() =
     '[]

__backward_npi_broadcast_sub ::
                             forall a t u .
                               (Tensor t, Fullfilled "__backward_npi_broadcast_sub" '() a,
                                HasCallStack, DType u) =>
                               ArgsHMap "__backward_npi_broadcast_sub" '() a -> TensorApply (t u)
__backward_npi_broadcast_sub args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_broadcast_sub" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_copysign" '() = '[]

__backward_npi_copysign ::
                        forall a t u .
                          (Tensor t, Fullfilled "__backward_npi_copysign" '() a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__backward_npi_copysign" '() a -> TensorApply (t u)
__backward_npi_copysign args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_copysign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__backward_npi_copysign_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__backward_npi_copysign_scalar ::
                               forall a t u .
                                 (Tensor t, Fullfilled "__backward_npi_copysign_scalar" '(t, u) a,
                                  HasCallStack, DType u) =>
                                 ArgsHMap "__backward_npi_copysign_scalar" '(t, u) a ->
                                   TensorApply (t u)
__backward_npi_copysign_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_backward_npi_copysign_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_diff" '() = '[]

__backward_npi_diff ::
                    forall a t u .
                      (Tensor t, Fullfilled "__backward_npi_diff" '() a, HasCallStack,
                       DType u) =>
                      ArgsHMap "__backward_npi_diff" '() a -> TensorApply (t u)
__backward_npi_diff args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_diff" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_einsum" '() = '[]

__backward_npi_einsum ::
                      forall a t u .
                        (Tensor t, Fullfilled "__backward_npi_einsum" '() a, HasCallStack,
                         DType u) =>
                        ArgsHMap "__backward_npi_einsum" '() a -> TensorApply (t u)
__backward_npi_einsum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_einsum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_flip" '() = '[]

__backward_npi_flip ::
                    forall a t u .
                      (Tensor t, Fullfilled "__backward_npi_flip" '() a, HasCallStack,
                       DType u) =>
                      ArgsHMap "__backward_npi_flip" '() a -> TensorApply (t u)
__backward_npi_flip args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_flip" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_hypot" '() = '[]

__backward_npi_hypot ::
                     forall a t u .
                       (Tensor t, Fullfilled "__backward_npi_hypot" '() a, HasCallStack,
                        DType u) =>
                       ArgsHMap "__backward_npi_hypot" '() a -> TensorApply (t u)
__backward_npi_hypot args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_hypot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_ldexp" '() = '[]

__backward_npi_ldexp ::
                     forall a t u .
                       (Tensor t, Fullfilled "__backward_npi_ldexp" '() a, HasCallStack,
                        DType u) =>
                       ArgsHMap "__backward_npi_ldexp" '() a -> TensorApply (t u)
__backward_npi_ldexp args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_ldexp" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_ldexp_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_npi_ldexp_scalar ::
                            forall a t u .
                              (Tensor t, Fullfilled "__backward_npi_ldexp_scalar" '(t, u) a,
                               HasCallStack, DType u) =>
                              ArgsHMap "__backward_npi_ldexp_scalar" '(t, u) a ->
                                TensorApply (t u)
__backward_npi_ldexp_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_npi_ldexp_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_norm" '() = '[]

__backward_npi_norm ::
                    forall a t u .
                      (Tensor t, Fullfilled "__backward_npi_norm" '() a, HasCallStack,
                       DType u) =>
                      ArgsHMap "__backward_npi_norm" '() a -> TensorApply (t u)
__backward_npi_norm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_norm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_pad" '() = '[]

__backward_npi_pad ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_npi_pad" '() a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_npi_pad" '() a -> TensorApply (t u)
__backward_npi_pad args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_pad" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__backward_npi_rarctan2_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_npi_rarctan2_scalar ::
                               forall a t u .
                                 (Tensor t, Fullfilled "__backward_npi_rarctan2_scalar" '(t, u) a,
                                  HasCallStack, DType u) =>
                                 ArgsHMap "__backward_npi_rarctan2_scalar" '(t, u) a ->
                                   TensorApply (t u)
__backward_npi_rarctan2_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_npi_rarctan2_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__backward_npi_rcopysign_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__backward_npi_rcopysign_scalar ::
                                forall a t u .
                                  (Tensor t, Fullfilled "__backward_npi_rcopysign_scalar" '(t, u) a,
                                   HasCallStack, DType u) =>
                                  ArgsHMap "__backward_npi_rcopysign_scalar" '(t, u) a ->
                                    TensorApply (t u)
__backward_npi_rcopysign_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_backward_npi_rcopysign_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_rldexp_scalar" '(t, u)
     =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_npi_rldexp_scalar ::
                             forall a t u .
                               (Tensor t, Fullfilled "__backward_npi_rldexp_scalar" '(t, u) a,
                                HasCallStack, DType u) =>
                               ArgsHMap "__backward_npi_rldexp_scalar" '(t, u) a ->
                                 TensorApply (t u)
__backward_npi_rldexp_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_npi_rldexp_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_solve" '() = '[]

__backward_npi_solve ::
                     forall a t u .
                       (Tensor t, Fullfilled "__backward_npi_solve" '() a, HasCallStack,
                        DType u) =>
                       ArgsHMap "__backward_npi_solve" '() a -> TensorApply (t u)
__backward_npi_solve args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_solve" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_svd" '() = '[]

__backward_npi_svd ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_npi_svd" '() a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_npi_svd" '() a -> TensorApply (t u)
__backward_npi_svd args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_svd" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_tensordot" '() = '[]

__backward_npi_tensordot ::
                         forall a t u .
                           (Tensor t, Fullfilled "__backward_npi_tensordot" '() a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__backward_npi_tensordot" '() a -> TensorApply (t u)
__backward_npi_tensordot args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_tensordot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_tensordot_int_axes" '()
     = '[]

__backward_npi_tensordot_int_axes ::
                                  forall a t u .
                                    (Tensor t, Fullfilled "__backward_npi_tensordot_int_axes" '() a,
                                     HasCallStack, DType u) =>
                                    ArgsHMap "__backward_npi_tensordot_int_axes" '() a ->
                                      TensorApply (t u)
__backward_npi_tensordot_int_axes args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_tensordot_int_axes" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_tensorinv" '() = '[]

__backward_npi_tensorinv ::
                         forall a t u .
                           (Tensor t, Fullfilled "__backward_npi_tensorinv" '() a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__backward_npi_tensorinv" '() a -> TensorApply (t u)
__backward_npi_tensorinv args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_tensorinv" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_npi_tensorsolve" '() = '[]

__backward_npi_tensorsolve ::
                           forall a t u .
                             (Tensor t, Fullfilled "__backward_npi_tensorsolve" '() a,
                              HasCallStack, DType u) =>
                             ArgsHMap "__backward_npi_tensorsolve" '() a -> TensorApply (t u)
__backward_npi_tensorsolve args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_npi_tensorsolve" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_pdf_dirichlet" '() = '[]

__backward_pdf_dirichlet ::
                         forall a t u .
                           (Tensor t, Fullfilled "__backward_pdf_dirichlet" '() a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__backward_pdf_dirichlet" '() a -> TensorApply (t u)
__backward_pdf_dirichlet args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_pdf_dirichlet" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_pdf_exponential" '() = '[]

__backward_pdf_exponential ::
                           forall a t u .
                             (Tensor t, Fullfilled "__backward_pdf_exponential" '() a,
                              HasCallStack, DType u) =>
                             ArgsHMap "__backward_pdf_exponential" '() a -> TensorApply (t u)
__backward_pdf_exponential args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_pdf_exponential" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_pdf_gamma" '() = '[]

__backward_pdf_gamma ::
                     forall a t u .
                       (Tensor t, Fullfilled "__backward_pdf_gamma" '() a, HasCallStack,
                        DType u) =>
                       ArgsHMap "__backward_pdf_gamma" '() a -> TensorApply (t u)
__backward_pdf_gamma args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_pdf_gamma" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__backward_pdf_generalized_negative_binomial" '() =
     '[]

__backward_pdf_generalized_negative_binomial ::
                                             forall a t u .
                                               (Tensor t,
                                                Fullfilled
                                                  "__backward_pdf_generalized_negative_binomial"
                                                  '()
                                                  a,
                                                HasCallStack, DType u) =>
                                               ArgsHMap
                                                 "__backward_pdf_generalized_negative_binomial"
                                                 '()
                                                 a
                                                 -> TensorApply (t u)
__backward_pdf_generalized_negative_binomial args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_pdf_generalized_negative_binomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_pdf_negative_binomial" '()
     = '[]

__backward_pdf_negative_binomial ::
                                 forall a t u .
                                   (Tensor t, Fullfilled "__backward_pdf_negative_binomial" '() a,
                                    HasCallStack, DType u) =>
                                   ArgsHMap "__backward_pdf_negative_binomial" '() a ->
                                     TensorApply (t u)
__backward_pdf_negative_binomial args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_pdf_negative_binomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_pdf_normal" '() = '[]

__backward_pdf_normal ::
                      forall a t u .
                        (Tensor t, Fullfilled "__backward_pdf_normal" '() a, HasCallStack,
                         DType u) =>
                        ArgsHMap "__backward_pdf_normal" '() a -> TensorApply (t u)
__backward_pdf_normal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_pdf_normal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_pdf_poisson" '() = '[]

__backward_pdf_poisson ::
                       forall a t u .
                         (Tensor t, Fullfilled "__backward_pdf_poisson" '() a, HasCallStack,
                          DType u) =>
                         ArgsHMap "__backward_pdf_poisson" '() a -> TensorApply (t u)
__backward_pdf_poisson args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_pdf_poisson" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_pdf_uniform" '() = '[]

__backward_pdf_uniform ::
                       forall a t u .
                         (Tensor t, Fullfilled "__backward_pdf_uniform" '() a, HasCallStack,
                          DType u) =>
                         ArgsHMap "__backward_pdf_uniform" '() a -> TensorApply (t u)
__backward_pdf_uniform args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_pdf_uniform" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_pick" '() = '[]

__backward_pick ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_pick" '() a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_pick" '() a -> TensorApply (t u)
__backward_pick args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_pick" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_power" '() = '[]

__backward_power ::
                 forall a t u .
                   (Tensor t, Fullfilled "__backward_power" '() a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__backward_power" '() a -> TensorApply (t u)
__backward_power args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_power" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_power_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_power_scalar ::
                        forall a t u .
                          (Tensor t, Fullfilled "__backward_power_scalar" '(t, u) a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__backward_power_scalar" '(t, u) a -> TensorApply (t u)
__backward_power_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_power_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_prod" '() = '[]

__backward_prod ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_prod" '() a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_prod" '() a -> TensorApply (t u)
__backward_prod args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_prod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_radians" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_radians ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_radians" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_radians" '(t, u) a -> TensorApply (t u)
__backward_radians args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_radians" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_rcbrt" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_rcbrt ::
                 forall a t u .
                   (Tensor t, Fullfilled "__backward_rcbrt" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__backward_rcbrt" '(t, u) a -> TensorApply (t u)
__backward_rcbrt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_rcbrt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_rdiv_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_rdiv_scalar ::
                       forall a t u .
                         (Tensor t, Fullfilled "__backward_rdiv_scalar" '(t, u) a,
                          HasCallStack, DType u) =>
                         ArgsHMap "__backward_rdiv_scalar" '(t, u) a -> TensorApply (t u)
__backward_rdiv_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_rdiv_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_reciprocal" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_reciprocal ::
                      forall a t u .
                        (Tensor t, Fullfilled "__backward_reciprocal" '(t, u) a,
                         HasCallStack, DType u) =>
                        ArgsHMap "__backward_reciprocal" '(t, u) a -> TensorApply (t u)
__backward_reciprocal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_reciprocal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_relu" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_relu ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_relu" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_relu" '(t, u) a -> TensorApply (t u)
__backward_relu args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_relu" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_repeat" '() = '[]

__backward_repeat ::
                  forall a t u .
                    (Tensor t, Fullfilled "__backward_repeat" '() a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__backward_repeat" '() a -> TensorApply (t u)
__backward_repeat args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_repeat" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_reshape" '() = '[]

__backward_reshape ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_reshape" '() a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_reshape" '() a -> TensorApply (t u)
__backward_reshape args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_reshape" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_reverse" '() = '[]

__backward_reverse ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_reverse" '() a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_reverse" '() a -> TensorApply (t u)
__backward_reverse args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_reverse" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_rmod_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_rmod_scalar ::
                       forall a t u .
                         (Tensor t, Fullfilled "__backward_rmod_scalar" '(t, u) a,
                          HasCallStack, DType u) =>
                         ArgsHMap "__backward_rmod_scalar" '(t, u) a -> TensorApply (t u)
__backward_rmod_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_rmod_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_rpower_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_rpower_scalar ::
                         forall a t u .
                           (Tensor t, Fullfilled "__backward_rpower_scalar" '(t, u) a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__backward_rpower_scalar" '(t, u) a -> TensorApply (t u)
__backward_rpower_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_rpower_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_rsqrt" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_rsqrt ::
                 forall a t u .
                   (Tensor t, Fullfilled "__backward_rsqrt" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__backward_rsqrt" '(t, u) a -> TensorApply (t u)
__backward_rsqrt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_rsqrt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_sample_multinomial" '() =
     '[]

__backward_sample_multinomial ::
                              forall a t u .
                                (Tensor t, Fullfilled "__backward_sample_multinomial" '() a,
                                 HasCallStack, DType u) =>
                                ArgsHMap "__backward_sample_multinomial" '() a -> TensorApply (t u)
__backward_sample_multinomial args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_sample_multinomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_sigmoid" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_sigmoid ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_sigmoid" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_sigmoid" '(t, u) a -> TensorApply (t u)
__backward_sigmoid args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_sigmoid" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_sign" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_sign ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_sign" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_sign" '(t, u) a -> TensorApply (t u)
__backward_sign args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_sign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_sin" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_sin ::
               forall a t u .
                 (Tensor t, Fullfilled "__backward_sin" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__backward_sin" '(t, u) a -> TensorApply (t u)
__backward_sin args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_sin" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_sinh" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_sinh ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_sinh" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_sinh" '(t, u) a -> TensorApply (t u)
__backward_sinh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_sinh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_slice" '() = '[]

__backward_slice ::
                 forall a t u .
                   (Tensor t, Fullfilled "__backward_slice" '() a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__backward_slice" '() a -> TensorApply (t u)
__backward_slice args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_slice" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_slice_axis" '() = '[]

__backward_slice_axis ::
                      forall a t u .
                        (Tensor t, Fullfilled "__backward_slice_axis" '() a, HasCallStack,
                         DType u) =>
                        ArgsHMap "__backward_slice_axis" '() a -> TensorApply (t u)
__backward_slice_axis args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_slice_axis" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_slice_like" '() = '[]

__backward_slice_like ::
                      forall a t u .
                        (Tensor t, Fullfilled "__backward_slice_like" '() a, HasCallStack,
                         DType u) =>
                        ArgsHMap "__backward_slice_like" '() a -> TensorApply (t u)
__backward_slice_like args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_slice_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_smooth_l1" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_smooth_l1 ::
                     forall a t u .
                       (Tensor t, Fullfilled "__backward_smooth_l1" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "__backward_smooth_l1" '(t, u) a -> TensorApply (t u)
__backward_smooth_l1 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_smooth_l1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_softmax" '(t, u) =
     '[ '("args", AttrOpt [t u])]

__backward_softmax ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_softmax" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_softmax" '(t, u) a -> TensorApply (t u)
__backward_softmax args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #args :: Maybe [RawTensor t])
      in applyRaw "_backward_softmax" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__backward_softmax_cross_entropy" '()
     = '[]

__backward_softmax_cross_entropy ::
                                 forall a t u .
                                   (Tensor t, Fullfilled "__backward_softmax_cross_entropy" '() a,
                                    HasCallStack, DType u) =>
                                   ArgsHMap "__backward_softmax_cross_entropy" '() a ->
                                     TensorApply (t u)
__backward_softmax_cross_entropy args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_softmax_cross_entropy" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_softmin" '(t, u) =
     '[ '("args", AttrOpt [t u])]

__backward_softmin ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_softmin" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_softmin" '(t, u) a -> TensorApply (t u)
__backward_softmin args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #args :: Maybe [RawTensor t])
      in applyRaw "_backward_softmin" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__backward_softsign" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_softsign ::
                    forall a t u .
                      (Tensor t, Fullfilled "__backward_softsign" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "__backward_softsign" '(t, u) a -> TensorApply (t u)
__backward_softsign args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_softsign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_sparse_retain" '() = '[]

__backward_sparse_retain ::
                         forall a t u .
                           (Tensor t, Fullfilled "__backward_sparse_retain" '() a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__backward_sparse_retain" '() a -> TensorApply (t u)
__backward_sparse_retain args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_sparse_retain" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_sqrt" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_sqrt ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_sqrt" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_sqrt" '(t, u) a -> TensorApply (t u)
__backward_sqrt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_sqrt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_square" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_square ::
                  forall a t u .
                    (Tensor t, Fullfilled "__backward_square" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__backward_square" '(t, u) a -> TensorApply (t u)
__backward_square args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_square" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_square_sum" '() = '[]

__backward_square_sum ::
                      forall a t u .
                        (Tensor t, Fullfilled "__backward_square_sum" '() a, HasCallStack,
                         DType u) =>
                        ArgsHMap "__backward_square_sum" '() a -> TensorApply (t u)
__backward_square_sum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_square_sum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_squeeze" '() = '[]

__backward_squeeze ::
                   forall a t u .
                     (Tensor t, Fullfilled "__backward_squeeze" '() a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__backward_squeeze" '() a -> TensorApply (t u)
__backward_squeeze args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_squeeze" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_stack" '() = '[]

__backward_stack ::
                 forall a t u .
                   (Tensor t, Fullfilled "__backward_stack" '() a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__backward_stack" '() a -> TensorApply (t u)
__backward_stack args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_stack" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_sub" '() = '[]

__backward_sub ::
               forall a t u .
                 (Tensor t, Fullfilled "__backward_sub" '() a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__backward_sub" '() a -> TensorApply (t u)
__backward_sub args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_sub" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_sum" '() = '[]

__backward_sum ::
               forall a t u .
                 (Tensor t, Fullfilled "__backward_sum" '() a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__backward_sum" '() a -> TensorApply (t u)
__backward_sum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_sum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_take" '() = '[]

__backward_take ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_take" '() a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_take" '() a -> TensorApply (t u)
__backward_take args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_take" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_tan" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_tan ::
               forall a t u .
                 (Tensor t, Fullfilled "__backward_tan" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__backward_tan" '(t, u) a -> TensorApply (t u)
__backward_tan args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_tan" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_tanh" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__backward_tanh ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_tanh" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_tanh" '(t, u) a -> TensorApply (t u)
__backward_tanh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_backward_tanh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_tile" '() = '[]

__backward_tile ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_tile" '() a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_tile" '() a -> TensorApply (t u)
__backward_tile args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_tile" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_topk" '() = '[]

__backward_topk ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_topk" '() a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_topk" '() a -> TensorApply (t u)
__backward_topk args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_topk" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_tril" '() = '[]

__backward_tril ::
                forall a t u .
                  (Tensor t, Fullfilled "__backward_tril" '() a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__backward_tril" '() a -> TensorApply (t u)
__backward_tril args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_tril" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_where" '() = '[]

__backward_where ::
                 forall a t u .
                   (Tensor t, Fullfilled "__backward_where" '() a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__backward_where" '() a -> TensorApply (t u)
__backward_where args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_where" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__backward_while_loop" '() = '[]

__backward_while_loop ::
                      forall a t u .
                        (Tensor t, Fullfilled "__backward_while_loop" '() a, HasCallStack,
                         DType u) =>
                        ArgsHMap "__backward_while_loop" '() a -> TensorApply (t u)
__backward_while_loop args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_backward_while_loop" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__broadcast_backward" '() = '[]

__broadcast_backward ::
                     forall a t u .
                       (Tensor t, Fullfilled "__broadcast_backward" '() a, HasCallStack,
                        DType u) =>
                       ArgsHMap "__broadcast_backward" '() a -> TensorApply (t u)
__broadcast_backward args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_broadcast_backward" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__contrib_AdaptiveAvgPooling2D" '(t, u) =
     '[ '("output_size", AttrOpt [Int]), '("data", AttrOpt (t u))]

__contrib_AdaptiveAvgPooling2D ::
                               forall a t u .
                                 (Tensor t, Fullfilled "__contrib_AdaptiveAvgPooling2D" '(t, u) a,
                                  HasCallStack, DType u) =>
                                 ArgsHMap "__contrib_AdaptiveAvgPooling2D" '(t, u) a ->
                                   TensorApply (t u)
__contrib_AdaptiveAvgPooling2D args
  = let scalarArgs
          = catMaybes
              [("output_size",) . showValue <$>
                 (args !? #output_size :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_contrib_AdaptiveAvgPooling2D" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_BatchNormWithReLU" '(t, u) =
     '[ '("eps", AttrOpt Double), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool), '("axis", AttrOpt Int),
        '("cudnn_off", AttrOpt Bool),
        '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)),
        '("data", AttrOpt (t u)), '("gamma", AttrOpt (t u)),
        '("beta", AttrOpt (t u)), '("moving_mean", AttrOpt (t u)),
        '("moving_var", AttrOpt (t u))]

__contrib_BatchNormWithReLU ::
                            forall a t u .
                              (Tensor t, Fullfilled "__contrib_BatchNormWithReLU" '(t, u) a,
                               HasCallStack, DType u) =>
                              ArgsHMap "__contrib_BatchNormWithReLU" '(t, u) a ->
                                TensorApply (t u)
__contrib_BatchNormWithReLU args
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
               ("cudnn_off",) . showValue <$> (args !? #cudnn_off :: Maybe Bool),
               ("min_calib_range",) . showValue <$>
                 (args !? #min_calib_range :: Maybe (Maybe Float)),
               ("max_calib_range",) . showValue <$>
                 (args !? #max_calib_range :: Maybe (Maybe Float))]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("gamma",) . toRaw <$> (args !? #gamma :: Maybe (t u)),
               ("beta",) . toRaw <$> (args !? #beta :: Maybe (t u)),
               ("moving_mean",) . toRaw <$> (args !? #moving_mean :: Maybe (t u)),
               ("moving_var",) . toRaw <$> (args !? #moving_var :: Maybe (t u))]
      in
      applyRaw "_contrib_BatchNormWithReLU" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_BilinearResize2D" '(t, u) =
     '[ '("height", AttrOpt Int), '("width", AttrOpt Int),
        '("scale_height", AttrOpt (Maybe Float)),
        '("scale_width", AttrOpt (Maybe Float)),
        '("mode",
          AttrOpt
            (EnumType
               '["like", "odd_scale", "size", "to_even_down", "to_even_up",
                 "to_odd_down", "to_odd_up"])),
        '("align_corners", AttrOpt Bool), '("data", AttrOpt (t u)),
        '("like", AttrOpt (t u))]

__contrib_BilinearResize2D ::
                           forall a t u .
                             (Tensor t, Fullfilled "__contrib_BilinearResize2D" '(t, u) a,
                              HasCallStack, DType u) =>
                             ArgsHMap "__contrib_BilinearResize2D" '(t, u) a ->
                               TensorApply (t u)
__contrib_BilinearResize2D args
  = let scalarArgs
          = catMaybes
              [("height",) . showValue <$> (args !? #height :: Maybe Int),
               ("width",) . showValue <$> (args !? #width :: Maybe Int),
               ("scale_height",) . showValue <$>
                 (args !? #scale_height :: Maybe (Maybe Float)),
               ("scale_width",) . showValue <$>
                 (args !? #scale_width :: Maybe (Maybe Float)),
               ("mode",) . showValue <$>
                 (args !? #mode ::
                    Maybe
                      (EnumType
                         '["like", "odd_scale", "size", "to_even_down", "to_even_up",
                           "to_odd_down", "to_odd_up"])),
               ("align_corners",) . showValue <$>
                 (args !? #align_corners :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("like",) . toRaw <$> (args !? #like :: Maybe (t u))]
      in
      applyRaw "_contrib_BilinearResize2D" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__contrib_DeformableConvolution" '(t, u) =
     '[ '("kernel", AttrReq [Int]), '("stride", AttrOpt [Int]),
        '("dilate", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("num_filter", AttrReq Int), '("num_group", AttrOpt Int),
        '("num_deformable_group", AttrOpt Int),
        '("workspace", AttrOpt Int), '("no_bias", AttrOpt Bool),
        '("layout", AttrOpt (Maybe (EnumType '["NCDHW", "NCHW", "NCW"]))),
        '("data", AttrOpt (t u)), '("offset", AttrOpt (t u)),
        '("weight", AttrOpt (t u)), '("bias", AttrOpt (t u))]

__contrib_DeformableConvolution ::
                                forall a t u .
                                  (Tensor t, Fullfilled "__contrib_DeformableConvolution" '(t, u) a,
                                   HasCallStack, DType u) =>
                                  ArgsHMap "__contrib_DeformableConvolution" '(t, u) a ->
                                    TensorApply (t u)
__contrib_DeformableConvolution args
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
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("offset",) . toRaw <$> (args !? #offset :: Maybe (t u)),
               ("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("bias",) . toRaw <$> (args !? #bias :: Maybe (t u))]
      in
      applyRaw "_contrib_DeformableConvolution" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__contrib_DeformablePSROIPooling" '(t, u) =
     '[ '("spatial_scale", AttrReq Float), '("output_dim", AttrReq Int),
        '("group_size", AttrReq Int), '("pooled_size", AttrReq Int),
        '("part_size", AttrOpt Int), '("sample_per_part", AttrOpt Int),
        '("trans_std", AttrOpt Float), '("no_trans", AttrOpt Bool),
        '("data", AttrOpt (t u)), '("rois", AttrOpt (t u)),
        '("trans", AttrOpt (t u))]

__contrib_DeformablePSROIPooling ::
                                 forall a t u .
                                   (Tensor t,
                                    Fullfilled "__contrib_DeformablePSROIPooling" '(t, u) a,
                                    HasCallStack, DType u) =>
                                   ArgsHMap "__contrib_DeformablePSROIPooling" '(t, u) a ->
                                     TensorApply (t u)
__contrib_DeformablePSROIPooling args
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
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("rois",) . toRaw <$> (args !? #rois :: Maybe (t u)),
               ("trans",) . toRaw <$> (args !? #trans :: Maybe (t u))]
      in
      applyRaw "_contrib_DeformablePSROIPooling" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__contrib_ModulatedDeformableConvolution" '(t, u) =
     '[ '("kernel", AttrReq [Int]), '("stride", AttrOpt [Int]),
        '("dilate", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("num_filter", AttrReq Int), '("num_group", AttrOpt Int),
        '("num_deformable_group", AttrOpt Int),
        '("workspace", AttrOpt Int), '("no_bias", AttrOpt Bool),
        '("im2col_step", AttrOpt Int),
        '("layout", AttrOpt (Maybe (EnumType '["NCDHW", "NCHW", "NCW"]))),
        '("data", AttrOpt (t u)), '("offset", AttrOpt (t u)),
        '("mask", AttrOpt (t u)), '("weight", AttrOpt (t u)),
        '("bias", AttrOpt (t u))]

__contrib_ModulatedDeformableConvolution ::
                                         forall a t u .
                                           (Tensor t,
                                            Fullfilled "__contrib_ModulatedDeformableConvolution"
                                              '(t, u)
                                              a,
                                            HasCallStack, DType u) =>
                                           ArgsHMap "__contrib_ModulatedDeformableConvolution"
                                             '(t, u)
                                             a
                                             -> TensorApply (t u)
__contrib_ModulatedDeformableConvolution args
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
               ("im2col_step",) . showValue <$>
                 (args !? #im2col_step :: Maybe Int),
               ("layout",) . showValue <$>
                 (args !? #layout ::
                    Maybe (Maybe (EnumType '["NCDHW", "NCHW", "NCW"])))]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("offset",) . toRaw <$> (args !? #offset :: Maybe (t u)),
               ("mask",) . toRaw <$> (args !? #mask :: Maybe (t u)),
               ("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("bias",) . toRaw <$> (args !? #bias :: Maybe (t u))]
      in
      applyRaw "_contrib_ModulatedDeformableConvolution" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_MultiBoxDetection" '(t, u) =
     '[ '("clip", AttrOpt Bool), '("threshold", AttrOpt Float),
        '("background_id", AttrOpt Int), '("nms_threshold", AttrOpt Float),
        '("force_suppress", AttrOpt Bool), '("variances", AttrOpt [Float]),
        '("nms_topk", AttrOpt Int), '("cls_prob", AttrOpt (t u)),
        '("loc_pred", AttrOpt (t u)), '("anchor", AttrOpt (t u))]

__contrib_MultiBoxDetection ::
                            forall a t u .
                              (Tensor t, Fullfilled "__contrib_MultiBoxDetection" '(t, u) a,
                               HasCallStack, DType u) =>
                              ArgsHMap "__contrib_MultiBoxDetection" '(t, u) a ->
                                TensorApply (t u)
__contrib_MultiBoxDetection args
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
        tensorKeyArgs
          = catMaybes
              [("cls_prob",) . toRaw <$> (args !? #cls_prob :: Maybe (t u)),
               ("loc_pred",) . toRaw <$> (args !? #loc_pred :: Maybe (t u)),
               ("anchor",) . toRaw <$> (args !? #anchor :: Maybe (t u))]
      in
      applyRaw "_contrib_MultiBoxDetection" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_MultiBoxPrior" '(t, u) =
     '[ '("sizes", AttrOpt [Float]), '("ratios", AttrOpt [Float]),
        '("clip", AttrOpt Bool), '("steps", AttrOpt [Float]),
        '("offsets", AttrOpt [Float]), '("data", AttrOpt (t u))]

__contrib_MultiBoxPrior ::
                        forall a t u .
                          (Tensor t, Fullfilled "__contrib_MultiBoxPrior" '(t, u) a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__contrib_MultiBoxPrior" '(t, u) a -> TensorApply (t u)
__contrib_MultiBoxPrior args
  = let scalarArgs
          = catMaybes
              [("sizes",) . showValue <$> (args !? #sizes :: Maybe [Float]),
               ("ratios",) . showValue <$> (args !? #ratios :: Maybe [Float]),
               ("clip",) . showValue <$> (args !? #clip :: Maybe Bool),
               ("steps",) . showValue <$> (args !? #steps :: Maybe [Float]),
               ("offsets",) . showValue <$> (args !? #offsets :: Maybe [Float])]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_contrib_MultiBoxPrior" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_MultiBoxTarget" '(t, u) =
     '[ '("overlap_threshold", AttrOpt Float),
        '("ignore_label", AttrOpt Float),
        '("negative_mining_ratio", AttrOpt Float),
        '("negative_mining_thresh", AttrOpt Float),
        '("minimum_negative_samples", AttrOpt Int),
        '("variances", AttrOpt [Float]), '("anchor", AttrOpt (t u)),
        '("label", AttrOpt (t u)), '("cls_pred", AttrOpt (t u))]

__contrib_MultiBoxTarget ::
                         forall a t u .
                           (Tensor t, Fullfilled "__contrib_MultiBoxTarget" '(t, u) a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__contrib_MultiBoxTarget" '(t, u) a -> TensorApply (t u)
__contrib_MultiBoxTarget args
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
        tensorKeyArgs
          = catMaybes
              [("anchor",) . toRaw <$> (args !? #anchor :: Maybe (t u)),
               ("label",) . toRaw <$> (args !? #label :: Maybe (t u)),
               ("cls_pred",) . toRaw <$> (args !? #cls_pred :: Maybe (t u))]
      in
      applyRaw "_contrib_MultiBoxTarget" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_MultiProposal" '(t, u) =
     '[ '("rpn_pre_nms_top_n", AttrOpt Int),
        '("rpn_post_nms_top_n", AttrOpt Int),
        '("threshold", AttrOpt Float), '("rpn_min_size", AttrOpt Int),
        '("scales", AttrOpt [Float]), '("ratios", AttrOpt [Float]),
        '("feature_stride", AttrOpt Int), '("output_score", AttrOpt Bool),
        '("iou_loss", AttrOpt Bool), '("cls_prob", AttrOpt (t u)),
        '("bbox_pred", AttrOpt (t u)), '("im_info", AttrOpt (t u))]

__contrib_MultiProposal ::
                        forall a t u .
                          (Tensor t, Fullfilled "__contrib_MultiProposal" '(t, u) a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__contrib_MultiProposal" '(t, u) a -> TensorApply (t u)
__contrib_MultiProposal args
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
        tensorKeyArgs
          = catMaybes
              [("cls_prob",) . toRaw <$> (args !? #cls_prob :: Maybe (t u)),
               ("bbox_pred",) . toRaw <$> (args !? #bbox_pred :: Maybe (t u)),
               ("im_info",) . toRaw <$> (args !? #im_info :: Maybe (t u))]
      in
      applyRaw "_contrib_MultiProposal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_PSROIPooling" '(t, u) =
     '[ '("spatial_scale", AttrReq Float), '("output_dim", AttrReq Int),
        '("pooled_size", AttrReq Int), '("group_size", AttrOpt Int),
        '("data", AttrOpt (t u)), '("rois", AttrOpt (t u))]

__contrib_PSROIPooling ::
                       forall a t u .
                         (Tensor t, Fullfilled "__contrib_PSROIPooling" '(t, u) a,
                          HasCallStack, DType u) =>
                         ArgsHMap "__contrib_PSROIPooling" '(t, u) a -> TensorApply (t u)
__contrib_PSROIPooling args
  = let scalarArgs
          = catMaybes
              [("spatial_scale",) . showValue <$>
                 (args !? #spatial_scale :: Maybe Float),
               ("output_dim",) . showValue <$> (args !? #output_dim :: Maybe Int),
               ("pooled_size",) . showValue <$>
                 (args !? #pooled_size :: Maybe Int),
               ("group_size",) . showValue <$> (args !? #group_size :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("rois",) . toRaw <$> (args !? #rois :: Maybe (t u))]
      in
      applyRaw "_contrib_PSROIPooling" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_Proposal" '(t, u) =
     '[ '("rpn_pre_nms_top_n", AttrOpt Int),
        '("rpn_post_nms_top_n", AttrOpt Int),
        '("threshold", AttrOpt Float), '("rpn_min_size", AttrOpt Int),
        '("scales", AttrOpt [Float]), '("ratios", AttrOpt [Float]),
        '("feature_stride", AttrOpt Int), '("output_score", AttrOpt Bool),
        '("iou_loss", AttrOpt Bool), '("cls_prob", AttrOpt (t u)),
        '("bbox_pred", AttrOpt (t u)), '("im_info", AttrOpt (t u))]

__contrib_Proposal ::
                   forall a t u .
                     (Tensor t, Fullfilled "__contrib_Proposal" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__contrib_Proposal" '(t, u) a -> TensorApply (t u)
__contrib_Proposal args
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
        tensorKeyArgs
          = catMaybes
              [("cls_prob",) . toRaw <$> (args !? #cls_prob :: Maybe (t u)),
               ("bbox_pred",) . toRaw <$> (args !? #bbox_pred :: Maybe (t u)),
               ("im_info",) . toRaw <$> (args !? #im_info :: Maybe (t u))]
      in
      applyRaw "_contrib_Proposal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_ROIAlign" '(t, u) =
     '[ '("pooled_size", AttrReq [Int]),
        '("spatial_scale", AttrReq Float), '("sample_ratio", AttrOpt Int),
        '("position_sensitive", AttrOpt Bool), '("aligned", AttrOpt Bool),
        '("data", AttrOpt (t u)), '("rois", AttrOpt (t u))]

__contrib_ROIAlign ::
                   forall a t u .
                     (Tensor t, Fullfilled "__contrib_ROIAlign" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__contrib_ROIAlign" '(t, u) a -> TensorApply (t u)
__contrib_ROIAlign args
  = let scalarArgs
          = catMaybes
              [("pooled_size",) . showValue <$>
                 (args !? #pooled_size :: Maybe [Int]),
               ("spatial_scale",) . showValue <$>
                 (args !? #spatial_scale :: Maybe Float),
               ("sample_ratio",) . showValue <$>
                 (args !? #sample_ratio :: Maybe Int),
               ("position_sensitive",) . showValue <$>
                 (args !? #position_sensitive :: Maybe Bool),
               ("aligned",) . showValue <$> (args !? #aligned :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("rois",) . toRaw <$> (args !? #rois :: Maybe (t u))]
      in
      applyRaw "_contrib_ROIAlign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_RROIAlign" '(t, u) =
     '[ '("pooled_size", AttrReq [Int]),
        '("spatial_scale", AttrReq Float),
        '("sampling_ratio", AttrOpt Int), '("data", AttrOpt (t u)),
        '("rois", AttrOpt (t u))]

__contrib_RROIAlign ::
                    forall a t u .
                      (Tensor t, Fullfilled "__contrib_RROIAlign" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "__contrib_RROIAlign" '(t, u) a -> TensorApply (t u)
__contrib_RROIAlign args
  = let scalarArgs
          = catMaybes
              [("pooled_size",) . showValue <$>
                 (args !? #pooled_size :: Maybe [Int]),
               ("spatial_scale",) . showValue <$>
                 (args !? #spatial_scale :: Maybe Float),
               ("sampling_ratio",) . showValue <$>
                 (args !? #sampling_ratio :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("rois",) . toRaw <$> (args !? #rois :: Maybe (t u))]
      in
      applyRaw "_contrib_RROIAlign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_SparseEmbedding" '(t, u) =
     '[ '("input_dim", AttrReq Int), '("output_dim", AttrReq Int),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"])),
        '("sparse_grad", AttrOpt Bool), '("data", AttrOpt (t u)),
        '("weight", AttrOpt (t u))]

__contrib_SparseEmbedding ::
                          forall a t u .
                            (Tensor t, Fullfilled "__contrib_SparseEmbedding" '(t, u) a,
                             HasCallStack, DType u) =>
                            ArgsHMap "__contrib_SparseEmbedding" '(t, u) a -> TensorApply (t u)
__contrib_SparseEmbedding args
  = let scalarArgs
          = catMaybes
              [("input_dim",) . showValue <$> (args !? #input_dim :: Maybe Int),
               ("output_dim",) . showValue <$> (args !? #output_dim :: Maybe Int),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                           "int8", "uint8"])),
               ("sparse_grad",) . showValue <$>
                 (args !? #sparse_grad :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("weight",) . toRaw <$> (args !? #weight :: Maybe (t u))]
      in
      applyRaw "_contrib_SparseEmbedding" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_SyncBatchNorm" '(t, u) =
     '[ '("eps", AttrOpt Float), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool), '("ndev", AttrOpt Int),
        '("key", AttrReq Text), '("data", AttrOpt (t u)),
        '("gamma", AttrOpt (t u)), '("beta", AttrOpt (t u)),
        '("moving_mean", AttrOpt (t u)), '("moving_var", AttrOpt (t u))]

__contrib_SyncBatchNorm ::
                        forall a t u .
                          (Tensor t, Fullfilled "__contrib_SyncBatchNorm" '(t, u) a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__contrib_SyncBatchNorm" '(t, u) a -> TensorApply (t u)
__contrib_SyncBatchNorm args
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
               ("key",) . showValue <$> (args !? #key :: Maybe Text)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("gamma",) . toRaw <$> (args !? #gamma :: Maybe (t u)),
               ("beta",) . toRaw <$> (args !? #beta :: Maybe (t u)),
               ("moving_mean",) . toRaw <$> (args !? #moving_mean :: Maybe (t u)),
               ("moving_var",) . toRaw <$> (args !? #moving_var :: Maybe (t u))]
      in
      applyRaw "_contrib_SyncBatchNorm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_allclose" '(t, u) =
     '[ '("rtol", AttrOpt Float), '("atol", AttrOpt Float),
        '("equal_nan", AttrOpt Bool), '("a", AttrOpt (t u)),
        '("b", AttrOpt (t u))]

__contrib_allclose ::
                   forall a t u .
                     (Tensor t, Fullfilled "__contrib_allclose" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__contrib_allclose" '(t, u) a -> TensorApply (t u)
__contrib_allclose args
  = let scalarArgs
          = catMaybes
              [("rtol",) . showValue <$> (args !? #rtol :: Maybe Float),
               ("atol",) . showValue <$> (args !? #atol :: Maybe Float),
               ("equal_nan",) . showValue <$> (args !? #equal_nan :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> (args !? #a :: Maybe (t u)),
               ("b",) . toRaw <$> (args !? #b :: Maybe (t u))]
      in
      applyRaw "_contrib_allclose" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_arange_like" '(t, u) =
     '[ '("start", AttrOpt Double), '("step", AttrOpt Double),
        '("repeat", AttrOpt Int), '("ctx", AttrOpt Text),
        '("axis", AttrOpt (Maybe Int)), '("data", AttrOpt (t u))]

__contrib_arange_like ::
                      forall a t u .
                        (Tensor t, Fullfilled "__contrib_arange_like" '(t, u) a,
                         HasCallStack, DType u) =>
                        ArgsHMap "__contrib_arange_like" '(t, u) a -> TensorApply (t u)
__contrib_arange_like args
  = let scalarArgs
          = catMaybes
              [("start",) . showValue <$> (args !? #start :: Maybe Double),
               ("step",) . showValue <$> (args !? #step :: Maybe Double),
               ("repeat",) . showValue <$> (args !? #repeat :: Maybe Int),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_contrib_arange_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__contrib_backward_gradientmultiplier" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__contrib_backward_gradientmultiplier ::
                                      forall a t u .
                                        (Tensor t,
                                         Fullfilled "__contrib_backward_gradientmultiplier" '(t, u)
                                           a,
                                         HasCallStack, DType u) =>
                                        ArgsHMap "__contrib_backward_gradientmultiplier" '(t, u) a
                                          -> TensorApply (t u)
__contrib_backward_gradientmultiplier args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_contrib_backward_gradientmultiplier" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_backward_hawkesll" '() = '[]

__contrib_backward_hawkesll ::
                            forall a t u .
                              (Tensor t, Fullfilled "__contrib_backward_hawkesll" '() a,
                               HasCallStack, DType u) =>
                              ArgsHMap "__contrib_backward_hawkesll" '() a -> TensorApply (t u)
__contrib_backward_hawkesll args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_contrib_backward_hawkesll" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_backward_index_copy" '() =
     '[]

__contrib_backward_index_copy ::
                              forall a t u .
                                (Tensor t, Fullfilled "__contrib_backward_index_copy" '() a,
                                 HasCallStack, DType u) =>
                                ArgsHMap "__contrib_backward_index_copy" '() a -> TensorApply (t u)
__contrib_backward_index_copy args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_contrib_backward_index_copy" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_backward_quadratic" '() =
     '[]

__contrib_backward_quadratic ::
                             forall a t u .
                               (Tensor t, Fullfilled "__contrib_backward_quadratic" '() a,
                                HasCallStack, DType u) =>
                               ArgsHMap "__contrib_backward_quadratic" '() a -> TensorApply (t u)
__contrib_backward_quadratic args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_contrib_backward_quadratic" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_bipartite_matching" '(t, u)
     =
     '[ '("is_ascend", AttrOpt Bool), '("threshold", AttrReq Float),
        '("topk", AttrOpt Int), '("data", AttrOpt (t u))]

__contrib_bipartite_matching ::
                             forall a t u .
                               (Tensor t, Fullfilled "__contrib_bipartite_matching" '(t, u) a,
                                HasCallStack, DType u) =>
                               ArgsHMap "__contrib_bipartite_matching" '(t, u) a ->
                                 TensorApply (t u)
__contrib_bipartite_matching args
  = let scalarArgs
          = catMaybes
              [("is_ascend",) . showValue <$> (args !? #is_ascend :: Maybe Bool),
               ("threshold",) . showValue <$> (args !? #threshold :: Maybe Float),
               ("topk",) . showValue <$> (args !? #topk :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_contrib_bipartite_matching" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_boolean_mask" '(t, u) =
     '[ '("axis", AttrOpt Int), '("data", AttrOpt (t u)),
        '("index", AttrOpt (t u))]

__contrib_boolean_mask ::
                       forall a t u .
                         (Tensor t, Fullfilled "__contrib_boolean_mask" '(t, u) a,
                          HasCallStack, DType u) =>
                         ArgsHMap "__contrib_boolean_mask" '(t, u) a -> TensorApply (t u)
__contrib_boolean_mask args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("index",) . toRaw <$> (args !? #index :: Maybe (t u))]
      in
      applyRaw "_contrib_boolean_mask" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_box_decode" '(t, u) =
     '[ '("std0", AttrOpt Float), '("std1", AttrOpt Float),
        '("std2", AttrOpt Float), '("std3", AttrOpt Float),
        '("clip", AttrOpt Float),
        '("format", AttrOpt (EnumType '["center", "corner"])),
        '("data", AttrOpt (t u)), '("anchors", AttrOpt (t u))]

__contrib_box_decode ::
                     forall a t u .
                       (Tensor t, Fullfilled "__contrib_box_decode" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "__contrib_box_decode" '(t, u) a -> TensorApply (t u)
__contrib_box_decode args
  = let scalarArgs
          = catMaybes
              [("std0",) . showValue <$> (args !? #std0 :: Maybe Float),
               ("std1",) . showValue <$> (args !? #std1 :: Maybe Float),
               ("std2",) . showValue <$> (args !? #std2 :: Maybe Float),
               ("std3",) . showValue <$> (args !? #std3 :: Maybe Float),
               ("clip",) . showValue <$> (args !? #clip :: Maybe Float),
               ("format",) . showValue <$>
                 (args !? #format :: Maybe (EnumType '["center", "corner"]))]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("anchors",) . toRaw <$> (args !? #anchors :: Maybe (t u))]
      in
      applyRaw "_contrib_box_decode" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_box_encode" '(t, u) =
     '[ '("samples", AttrOpt (t u)), '("matches", AttrOpt (t u)),
        '("anchors", AttrOpt (t u)), '("refs", AttrOpt (t u)),
        '("means", AttrOpt (t u)), '("stds", AttrOpt (t u))]

__contrib_box_encode ::
                     forall a t u .
                       (Tensor t, Fullfilled "__contrib_box_encode" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "__contrib_box_encode" '(t, u) a -> TensorApply (t u)
__contrib_box_encode args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("samples",) . toRaw <$> (args !? #samples :: Maybe (t u)),
               ("matches",) . toRaw <$> (args !? #matches :: Maybe (t u)),
               ("anchors",) . toRaw <$> (args !? #anchors :: Maybe (t u)),
               ("refs",) . toRaw <$> (args !? #refs :: Maybe (t u)),
               ("means",) . toRaw <$> (args !? #means :: Maybe (t u)),
               ("stds",) . toRaw <$> (args !? #stds :: Maybe (t u))]
      in
      applyRaw "_contrib_box_encode" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_box_iou" '(t, u) =
     '[ '("format", AttrOpt (EnumType '["center", "corner"])),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__contrib_box_iou ::
                  forall a t u .
                    (Tensor t, Fullfilled "__contrib_box_iou" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__contrib_box_iou" '(t, u) a -> TensorApply (t u)
__contrib_box_iou args
  = let scalarArgs
          = catMaybes
              [("format",) . showValue <$>
                 (args !? #format :: Maybe (EnumType '["center", "corner"]))]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_contrib_box_iou" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_box_nms" '(t, u) =
     '[ '("overlap_thresh", AttrOpt Float),
        '("valid_thresh", AttrOpt Float), '("topk", AttrOpt Int),
        '("coord_start", AttrOpt Int), '("score_index", AttrOpt Int),
        '("id_index", AttrOpt Int), '("background_id", AttrOpt Int),
        '("force_suppress", AttrOpt Bool),
        '("in_format", AttrOpt (EnumType '["center", "corner"])),
        '("out_format", AttrOpt (EnumType '["center", "corner"])),
        '("data", AttrOpt (t u))]

__contrib_box_nms ::
                  forall a t u .
                    (Tensor t, Fullfilled "__contrib_box_nms" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__contrib_box_nms" '(t, u) a -> TensorApply (t u)
__contrib_box_nms args
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
               ("background_id",) . showValue <$>
                 (args !? #background_id :: Maybe Int),
               ("force_suppress",) . showValue <$>
                 (args !? #force_suppress :: Maybe Bool),
               ("in_format",) . showValue <$>
                 (args !? #in_format :: Maybe (EnumType '["center", "corner"])),
               ("out_format",) . showValue <$>
                 (args !? #out_format :: Maybe (EnumType '["center", "corner"]))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_contrib_box_nms" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_calibrate_entropy" '(t, u) =
     '[ '("num_quantized_bins", AttrOpt Int), '("hist", AttrOpt (t u)),
        '("hist_edges", AttrOpt (t u))]

__contrib_calibrate_entropy ::
                            forall a t u .
                              (Tensor t, Fullfilled "__contrib_calibrate_entropy" '(t, u) a,
                               HasCallStack, DType u) =>
                              ArgsHMap "__contrib_calibrate_entropy" '(t, u) a ->
                                TensorApply (t u)
__contrib_calibrate_entropy args
  = let scalarArgs
          = catMaybes
              [("num_quantized_bins",) . showValue <$>
                 (args !? #num_quantized_bins :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("hist",) . toRaw <$> (args !? #hist :: Maybe (t u)),
               ("hist_edges",) . toRaw <$> (args !? #hist_edges :: Maybe (t u))]
      in
      applyRaw "_contrib_calibrate_entropy" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_count_sketch" '(t, u) =
     '[ '("out_dim", AttrReq Int),
        '("processing_batch_size", AttrOpt Int), '("data", AttrOpt (t u)),
        '("h", AttrOpt (t u)), '("s", AttrOpt (t u))]

__contrib_count_sketch ::
                       forall a t u .
                         (Tensor t, Fullfilled "__contrib_count_sketch" '(t, u) a,
                          HasCallStack, DType u) =>
                         ArgsHMap "__contrib_count_sketch" '(t, u) a -> TensorApply (t u)
__contrib_count_sketch args
  = let scalarArgs
          = catMaybes
              [("out_dim",) . showValue <$> (args !? #out_dim :: Maybe Int),
               ("processing_batch_size",) . showValue <$>
                 (args !? #processing_batch_size :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("h",) . toRaw <$> (args !? #h :: Maybe (t u)),
               ("s",) . toRaw <$> (args !? #s :: Maybe (t u))]
      in
      applyRaw "_contrib_count_sketch" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_dequantize" '(t, u) =
     '[ '("out_type", AttrOpt (EnumType '["float32"])),
        '("data", AttrOpt (t u)), '("min_range", AttrOpt (t u)),
        '("max_range", AttrOpt (t u))]

__contrib_dequantize ::
                     forall a t u .
                       (Tensor t, Fullfilled "__contrib_dequantize" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "__contrib_dequantize" '(t, u) a -> TensorApply (t u)
__contrib_dequantize args
  = let scalarArgs
          = catMaybes
              [("out_type",) . showValue <$>
                 (args !? #out_type :: Maybe (EnumType '["float32"]))]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("min_range",) . toRaw <$> (args !? #min_range :: Maybe (t u)),
               ("max_range",) . toRaw <$> (args !? #max_range :: Maybe (t u))]
      in
      applyRaw "_contrib_dequantize" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_dgl_adjacency" '(t, u) =
     '[ '("data", AttrOpt (t u))]

__contrib_dgl_adjacency ::
                        forall a t u .
                          (Tensor t, Fullfilled "__contrib_dgl_adjacency" '(t, u) a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__contrib_dgl_adjacency" '(t, u) a -> TensorApply (t u)
__contrib_dgl_adjacency args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_contrib_dgl_adjacency" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_div_sqrt_dim" '(t, u) =
     '[ '("data", AttrOpt (t u))]

__contrib_div_sqrt_dim ::
                       forall a t u .
                         (Tensor t, Fullfilled "__contrib_div_sqrt_dim" '(t, u) a,
                          HasCallStack, DType u) =>
                         ArgsHMap "__contrib_div_sqrt_dim" '(t, u) a -> TensorApply (t u)
__contrib_div_sqrt_dim args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_contrib_div_sqrt_dim" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_edge_id" '(t, u) =
     '[ '("data", AttrOpt (t u)), '("u", AttrOpt (t u)),
        '("v", AttrOpt (t u))]

__contrib_edge_id ::
                  forall a t u .
                    (Tensor t, Fullfilled "__contrib_edge_id" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__contrib_edge_id" '(t, u) a -> TensorApply (t u)
__contrib_edge_id args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("u",) . toRaw <$> (args !? #u :: Maybe (t u)),
               ("v",) . toRaw <$> (args !? #v :: Maybe (t u))]
      in
      applyRaw "_contrib_edge_id" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_fft" '(t, u) =
     '[ '("compute_size", AttrOpt Int), '("data", AttrOpt (t u))]

__contrib_fft ::
              forall a t u .
                (Tensor t, Fullfilled "__contrib_fft" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__contrib_fft" '(t, u) a -> TensorApply (t u)
__contrib_fft args
  = let scalarArgs
          = catMaybes
              [("compute_size",) . showValue <$>
                 (args !? #compute_size :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_contrib_fft" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_getnnz" '(t, u) =
     '[ '("axis", AttrOpt (Maybe Int)), '("data", AttrOpt (t u))]

__contrib_getnnz ::
                 forall a t u .
                   (Tensor t, Fullfilled "__contrib_getnnz" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__contrib_getnnz" '(t, u) a -> TensorApply (t u)
__contrib_getnnz args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_contrib_getnnz" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_gradientmultiplier" '(t, u)
     =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__contrib_gradientmultiplier ::
                             forall a t u .
                               (Tensor t, Fullfilled "__contrib_gradientmultiplier" '(t, u) a,
                                HasCallStack, DType u) =>
                               ArgsHMap "__contrib_gradientmultiplier" '(t, u) a ->
                                 TensorApply (t u)
__contrib_gradientmultiplier args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_contrib_gradientmultiplier" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__contrib_group_adagrad_update" '(t, u) =
     '[ '("lr", AttrReq Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("weight", AttrOpt (t u)), '("grad", AttrOpt (t u)),
        '("history", AttrOpt (t u))]

__contrib_group_adagrad_update ::
                               forall a t u .
                                 (Tensor t, Fullfilled "__contrib_group_adagrad_update" '(t, u) a,
                                  HasCallStack, DType u) =>
                                 ArgsHMap "__contrib_group_adagrad_update" '(t, u) a ->
                                   TensorApply (t u)
__contrib_group_adagrad_update args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("epsilon",) . showValue <$> (args !? #epsilon :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("grad",) . toRaw <$> (args !? #grad :: Maybe (t u)),
               ("history",) . toRaw <$> (args !? #history :: Maybe (t u))]
      in
      applyRaw "_contrib_group_adagrad_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_hawkesll" '(t, u) =
     '[ '("lda", AttrOpt (t u)), '("alpha", AttrOpt (t u)),
        '("beta", AttrOpt (t u)), '("state", AttrOpt (t u)),
        '("lags", AttrOpt (t u)), '("marks", AttrOpt (t u)),
        '("valid_length", AttrOpt (t u)), '("max_time", AttrOpt (t u))]

__contrib_hawkesll ::
                   forall a t u .
                     (Tensor t, Fullfilled "__contrib_hawkesll" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__contrib_hawkesll" '(t, u) a -> TensorApply (t u)
__contrib_hawkesll args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lda",) . toRaw <$> (args !? #lda :: Maybe (t u)),
               ("alpha",) . toRaw <$> (args !? #alpha :: Maybe (t u)),
               ("beta",) . toRaw <$> (args !? #beta :: Maybe (t u)),
               ("state",) . toRaw <$> (args !? #state :: Maybe (t u)),
               ("lags",) . toRaw <$> (args !? #lags :: Maybe (t u)),
               ("marks",) . toRaw <$> (args !? #marks :: Maybe (t u)),
               ("valid_length",) . toRaw <$>
                 (args !? #valid_length :: Maybe (t u)),
               ("max_time",) . toRaw <$> (args !? #max_time :: Maybe (t u))]
      in
      applyRaw "_contrib_hawkesll" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_ifft" '(t, u) =
     '[ '("compute_size", AttrOpt Int), '("data", AttrOpt (t u))]

__contrib_ifft ::
               forall a t u .
                 (Tensor t, Fullfilled "__contrib_ifft" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__contrib_ifft" '(t, u) a -> TensorApply (t u)
__contrib_ifft args
  = let scalarArgs
          = catMaybes
              [("compute_size",) . showValue <$>
                 (args !? #compute_size :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_contrib_ifft" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_index_array" '(t, u) =
     '[ '("axes", AttrOpt (Maybe [Int])), '("data", AttrOpt (t u))]

__contrib_index_array ::
                      forall a t u .
                        (Tensor t, Fullfilled "__contrib_index_array" '(t, u) a,
                         HasCallStack, DType u) =>
                        ArgsHMap "__contrib_index_array" '(t, u) a -> TensorApply (t u)
__contrib_index_array args
  = let scalarArgs
          = catMaybes
              [("axes",) . showValue <$> (args !? #axes :: Maybe (Maybe [Int]))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_contrib_index_array" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_index_copy" '(t, u) =
     '[ '("old_tensor", AttrOpt (t u)),
        '("index_vector", AttrOpt (t u)), '("new_tensor", AttrOpt (t u))]

__contrib_index_copy ::
                     forall a t u .
                       (Tensor t, Fullfilled "__contrib_index_copy" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "__contrib_index_copy" '(t, u) a -> TensorApply (t u)
__contrib_index_copy args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("old_tensor",) . toRaw <$> (args !? #old_tensor :: Maybe (t u)),
               ("index_vector",) . toRaw <$>
                 (args !? #index_vector :: Maybe (t u)),
               ("new_tensor",) . toRaw <$> (args !? #new_tensor :: Maybe (t u))]
      in
      applyRaw "_contrib_index_copy" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__contrib_interleaved_matmul_encdec_qk" '(t, u) =
     '[ '("heads", AttrReq Int), '("queries", AttrOpt (t u)),
        '("keys_values", AttrOpt (t u))]

__contrib_interleaved_matmul_encdec_qk ::
                                       forall a t u .
                                         (Tensor t,
                                          Fullfilled "__contrib_interleaved_matmul_encdec_qk"
                                            '(t, u)
                                            a,
                                          HasCallStack, DType u) =>
                                         ArgsHMap "__contrib_interleaved_matmul_encdec_qk" '(t, u) a
                                           -> TensorApply (t u)
__contrib_interleaved_matmul_encdec_qk args
  = let scalarArgs
          = catMaybes
              [("heads",) . showValue <$> (args !? #heads :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("queries",) . toRaw <$> (args !? #queries :: Maybe (t u)),
               ("keys_values",) . toRaw <$> (args !? #keys_values :: Maybe (t u))]
      in
      applyRaw "_contrib_interleaved_matmul_encdec_qk" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__contrib_interleaved_matmul_encdec_valatt" '(t, u)
     =
     '[ '("heads", AttrReq Int), '("keys_values", AttrOpt (t u)),
        '("attention", AttrOpt (t u))]

__contrib_interleaved_matmul_encdec_valatt ::
                                           forall a t u .
                                             (Tensor t,
                                              Fullfilled
                                                "__contrib_interleaved_matmul_encdec_valatt"
                                                '(t, u)
                                                a,
                                              HasCallStack, DType u) =>
                                             ArgsHMap "__contrib_interleaved_matmul_encdec_valatt"
                                               '(t, u)
                                               a
                                               -> TensorApply (t u)
__contrib_interleaved_matmul_encdec_valatt args
  = let scalarArgs
          = catMaybes
              [("heads",) . showValue <$> (args !? #heads :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("keys_values",) . toRaw <$>
                 (args !? #keys_values :: Maybe (t u)),
               ("attention",) . toRaw <$> (args !? #attention :: Maybe (t u))]
      in
      applyRaw "_contrib_interleaved_matmul_encdec_valatt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__contrib_interleaved_matmul_selfatt_qk" '(t, u) =
     '[ '("heads", AttrReq Int),
        '("queries_keys_values", AttrOpt (t u))]

__contrib_interleaved_matmul_selfatt_qk ::
                                        forall a t u .
                                          (Tensor t,
                                           Fullfilled "__contrib_interleaved_matmul_selfatt_qk"
                                             '(t, u)
                                             a,
                                           HasCallStack, DType u) =>
                                          ArgsHMap "__contrib_interleaved_matmul_selfatt_qk" '(t, u)
                                            a
                                            -> TensorApply (t u)
__contrib_interleaved_matmul_selfatt_qk args
  = let scalarArgs
          = catMaybes
              [("heads",) . showValue <$> (args !? #heads :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("queries_keys_values",) . toRaw <$>
                 (args !? #queries_keys_values :: Maybe (t u))]
      in
      applyRaw "_contrib_interleaved_matmul_selfatt_qk" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__contrib_interleaved_matmul_selfatt_valatt" '(t, u)
     =
     '[ '("heads", AttrReq Int),
        '("queries_keys_values", AttrOpt (t u)),
        '("attention", AttrOpt (t u))]

__contrib_interleaved_matmul_selfatt_valatt ::
                                            forall a t u .
                                              (Tensor t,
                                               Fullfilled
                                                 "__contrib_interleaved_matmul_selfatt_valatt"
                                                 '(t, u)
                                                 a,
                                               HasCallStack, DType u) =>
                                              ArgsHMap "__contrib_interleaved_matmul_selfatt_valatt"
                                                '(t, u)
                                                a
                                                -> TensorApply (t u)
__contrib_interleaved_matmul_selfatt_valatt args
  = let scalarArgs
          = catMaybes
              [("heads",) . showValue <$> (args !? #heads :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("queries_keys_values",) . toRaw <$>
                 (args !? #queries_keys_values :: Maybe (t u)),
               ("attention",) . toRaw <$> (args !? #attention :: Maybe (t u))]
      in
      applyRaw "_contrib_interleaved_matmul_selfatt_valatt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__contrib_intgemm_fully_connected" '(t, u) =
     '[ '("num_hidden", AttrReq Int), '("no_bias", AttrOpt Bool),
        '("flatten", AttrOpt Bool),
        '("out_type", AttrOpt (EnumType '["float32", "int32"])),
        '("data", AttrOpt (t u)), '("weight", AttrOpt (t u)),
        '("scaling", AttrOpt (t u)), '("bias", AttrOpt (t u))]

__contrib_intgemm_fully_connected ::
                                  forall a t u .
                                    (Tensor t,
                                     Fullfilled "__contrib_intgemm_fully_connected" '(t, u) a,
                                     HasCallStack, DType u) =>
                                    ArgsHMap "__contrib_intgemm_fully_connected" '(t, u) a ->
                                      TensorApply (t u)
__contrib_intgemm_fully_connected args
  = let scalarArgs
          = catMaybes
              [("num_hidden",) . showValue <$>
                 (args !? #num_hidden :: Maybe Int),
               ("no_bias",) . showValue <$> (args !? #no_bias :: Maybe Bool),
               ("flatten",) . showValue <$> (args !? #flatten :: Maybe Bool),
               ("out_type",) . showValue <$>
                 (args !? #out_type :: Maybe (EnumType '["float32", "int32"]))]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("scaling",) . toRaw <$> (args !? #scaling :: Maybe (t u)),
               ("bias",) . toRaw <$> (args !? #bias :: Maybe (t u))]
      in
      applyRaw "_contrib_intgemm_fully_connected" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_intgemm_maxabsolute" '(t, u)
     = '[ '("data", AttrOpt (t u))]

__contrib_intgemm_maxabsolute ::
                              forall a t u .
                                (Tensor t, Fullfilled "__contrib_intgemm_maxabsolute" '(t, u) a,
                                 HasCallStack, DType u) =>
                                ArgsHMap "__contrib_intgemm_maxabsolute" '(t, u) a ->
                                  TensorApply (t u)
__contrib_intgemm_maxabsolute args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_contrib_intgemm_maxabsolute" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__contrib_intgemm_prepare_data" '(t, u) =
     '[ '("data", AttrOpt (t u)), '("maxabs", AttrOpt (t u))]

__contrib_intgemm_prepare_data ::
                               forall a t u .
                                 (Tensor t, Fullfilled "__contrib_intgemm_prepare_data" '(t, u) a,
                                  HasCallStack, DType u) =>
                                 ArgsHMap "__contrib_intgemm_prepare_data" '(t, u) a ->
                                   TensorApply (t u)
__contrib_intgemm_prepare_data args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("maxabs",) . toRaw <$> (args !? #maxabs :: Maybe (t u))]
      in
      applyRaw "_contrib_intgemm_prepare_data" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__contrib_intgemm_prepare_weight" '(t, u) =
     '[ '("already_quantized", AttrOpt Bool),
        '("weight", AttrOpt (t u)), '("maxabs", AttrOpt (t u))]

__contrib_intgemm_prepare_weight ::
                                 forall a t u .
                                   (Tensor t,
                                    Fullfilled "__contrib_intgemm_prepare_weight" '(t, u) a,
                                    HasCallStack, DType u) =>
                                   ArgsHMap "__contrib_intgemm_prepare_weight" '(t, u) a ->
                                     TensorApply (t u)
__contrib_intgemm_prepare_weight args
  = let scalarArgs
          = catMaybes
              [("already_quantized",) . showValue <$>
                 (args !? #already_quantized :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("maxabs",) . toRaw <$> (args !? #maxabs :: Maybe (t u))]
      in
      applyRaw "_contrib_intgemm_prepare_weight" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_intgemm_take_weight" '(t, u)
     = '[ '("weight", AttrOpt (t u)), '("indices", AttrOpt (t u))]

__contrib_intgemm_take_weight ::
                              forall a t u .
                                (Tensor t, Fullfilled "__contrib_intgemm_take_weight" '(t, u) a,
                                 HasCallStack, DType u) =>
                                ArgsHMap "__contrib_intgemm_take_weight" '(t, u) a ->
                                  TensorApply (t u)
__contrib_intgemm_take_weight args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("indices",) . toRaw <$> (args !? #indices :: Maybe (t u))]
      in
      applyRaw "_contrib_intgemm_take_weight" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_mrcnn_mask_target" '(t, u) =
     '[ '("num_rois", AttrReq Int), '("num_classes", AttrReq Int),
        '("mask_size", AttrReq [Int]), '("sample_ratio", AttrOpt Int),
        '("aligned", AttrOpt Bool), '("rois", AttrOpt (t u)),
        '("gt_masks", AttrOpt (t u)), '("matches", AttrOpt (t u)),
        '("cls_targets", AttrOpt (t u))]

__contrib_mrcnn_mask_target ::
                            forall a t u .
                              (Tensor t, Fullfilled "__contrib_mrcnn_mask_target" '(t, u) a,
                               HasCallStack, DType u) =>
                              ArgsHMap "__contrib_mrcnn_mask_target" '(t, u) a ->
                                TensorApply (t u)
__contrib_mrcnn_mask_target args
  = let scalarArgs
          = catMaybes
              [("num_rois",) . showValue <$> (args !? #num_rois :: Maybe Int),
               ("num_classes",) . showValue <$>
                 (args !? #num_classes :: Maybe Int),
               ("mask_size",) . showValue <$> (args !? #mask_size :: Maybe [Int]),
               ("sample_ratio",) . showValue <$>
                 (args !? #sample_ratio :: Maybe Int),
               ("aligned",) . showValue <$> (args !? #aligned :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("rois",) . toRaw <$> (args !? #rois :: Maybe (t u)),
               ("gt_masks",) . toRaw <$> (args !? #gt_masks :: Maybe (t u)),
               ("matches",) . toRaw <$> (args !? #matches :: Maybe (t u)),
               ("cls_targets",) . toRaw <$> (args !? #cls_targets :: Maybe (t u))]
      in
      applyRaw "_contrib_mrcnn_mask_target" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_quadratic" '(t, u) =
     '[ '("a", AttrOpt Float), '("b", AttrOpt Float),
        '("c", AttrOpt Float), '("data", AttrOpt (t u))]

__contrib_quadratic ::
                    forall a t u .
                      (Tensor t, Fullfilled "__contrib_quadratic" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "__contrib_quadratic" '(t, u) a -> TensorApply (t u)
__contrib_quadratic args
  = let scalarArgs
          = catMaybes
              [("a",) . showValue <$> (args !? #a :: Maybe Float),
               ("b",) . showValue <$> (args !? #b :: Maybe Float),
               ("c",) . showValue <$> (args !? #c :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_contrib_quadratic" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_quantize" '(t, u) =
     '[ '("out_type", AttrOpt (EnumType '["int8", "uint8"])),
        '("data", AttrOpt (t u)), '("min_range", AttrOpt (t u)),
        '("max_range", AttrOpt (t u))]

__contrib_quantize ::
                   forall a t u .
                     (Tensor t, Fullfilled "__contrib_quantize" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__contrib_quantize" '(t, u) a -> TensorApply (t u)
__contrib_quantize args
  = let scalarArgs
          = catMaybes
              [("out_type",) . showValue <$>
                 (args !? #out_type :: Maybe (EnumType '["int8", "uint8"]))]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("min_range",) . toRaw <$> (args !? #min_range :: Maybe (t u)),
               ("max_range",) . toRaw <$> (args !? #max_range :: Maybe (t u))]
      in
      applyRaw "_contrib_quantize" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_quantize_v2" '(t, u) =
     '[ '("out_type", AttrOpt (EnumType '["auto", "int8", "uint8"])),
        '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)),
        '("data", AttrOpt (t u))]

__contrib_quantize_v2 ::
                      forall a t u .
                        (Tensor t, Fullfilled "__contrib_quantize_v2" '(t, u) a,
                         HasCallStack, DType u) =>
                        ArgsHMap "__contrib_quantize_v2" '(t, u) a -> TensorApply (t u)
__contrib_quantize_v2 args
  = let scalarArgs
          = catMaybes
              [("out_type",) . showValue <$>
                 (args !? #out_type :: Maybe (EnumType '["auto", "int8", "uint8"])),
               ("min_calib_range",) . showValue <$>
                 (args !? #min_calib_range :: Maybe (Maybe Float)),
               ("max_calib_range",) . showValue <$>
                 (args !? #max_calib_range :: Maybe (Maybe Float))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_contrib_quantize_v2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_quantized_act" '(t, u) =
     '[ '("act_type",
          AttrReq
            (EnumType '["relu", "sigmoid", "softrelu", "softsign", "tanh"])),
        '("data", AttrOpt (t u)), '("min_data", AttrOpt (t u)),
        '("max_data", AttrOpt (t u))]

__contrib_quantized_act ::
                        forall a t u .
                          (Tensor t, Fullfilled "__contrib_quantized_act" '(t, u) a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__contrib_quantized_act" '(t, u) a -> TensorApply (t u)
__contrib_quantized_act args
  = let scalarArgs
          = catMaybes
              [("act_type",) . showValue <$>
                 (args !? #act_type ::
                    Maybe
                      (EnumType '["relu", "sigmoid", "softrelu", "softsign", "tanh"]))]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("min_data",) . toRaw <$> (args !? #min_data :: Maybe (t u)),
               ("max_data",) . toRaw <$> (args !? #max_data :: Maybe (t u))]
      in
      applyRaw "_contrib_quantized_act" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__contrib_quantized_batch_norm" '(t, u) =
     '[ '("eps", AttrOpt Double), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool), '("axis", AttrOpt Int),
        '("cudnn_off", AttrOpt Bool),
        '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)),
        '("data", AttrOpt (t u)), '("gamma", AttrOpt (t u)),
        '("beta", AttrOpt (t u)), '("moving_mean", AttrOpt (t u)),
        '("moving_var", AttrOpt (t u)), '("min_data", AttrOpt (t u)),
        '("max_data", AttrOpt (t u))]

__contrib_quantized_batch_norm ::
                               forall a t u .
                                 (Tensor t, Fullfilled "__contrib_quantized_batch_norm" '(t, u) a,
                                  HasCallStack, DType u) =>
                                 ArgsHMap "__contrib_quantized_batch_norm" '(t, u) a ->
                                   TensorApply (t u)
__contrib_quantized_batch_norm args
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
               ("cudnn_off",) . showValue <$> (args !? #cudnn_off :: Maybe Bool),
               ("min_calib_range",) . showValue <$>
                 (args !? #min_calib_range :: Maybe (Maybe Float)),
               ("max_calib_range",) . showValue <$>
                 (args !? #max_calib_range :: Maybe (Maybe Float))]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("gamma",) . toRaw <$> (args !? #gamma :: Maybe (t u)),
               ("beta",) . toRaw <$> (args !? #beta :: Maybe (t u)),
               ("moving_mean",) . toRaw <$> (args !? #moving_mean :: Maybe (t u)),
               ("moving_var",) . toRaw <$> (args !? #moving_var :: Maybe (t u)),
               ("min_data",) . toRaw <$> (args !? #min_data :: Maybe (t u)),
               ("max_data",) . toRaw <$> (args !? #max_data :: Maybe (t u))]
      in
      applyRaw "_contrib_quantized_batch_norm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_quantized_concat" '(t, u) =
     '[ '("num_args", AttrReq Int), '("dim", AttrOpt Int),
        '("data", AttrOpt [t u])]

__contrib_quantized_concat ::
                           forall a t u .
                             (Tensor t, Fullfilled "__contrib_quantized_concat" '(t, u) a,
                              HasCallStack, DType u) =>
                             ArgsHMap "__contrib_quantized_concat" '(t, u) a ->
                               TensorApply (t u)
__contrib_quantized_concat args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("dim",) . showValue <$> (args !? #dim :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in
      applyRaw "_contrib_quantized_concat" scalarArgs
        (Right tensorVarArgs)

type instance ParameterList "__contrib_quantized_conv" '(t, u) =
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
        '("data", AttrOpt (t u)), '("weight", AttrOpt (t u)),
        '("bias", AttrOpt (t u)), '("min_data", AttrOpt (t u)),
        '("max_data", AttrOpt (t u)), '("min_weight", AttrOpt (t u)),
        '("max_weight", AttrOpt (t u)), '("min_bias", AttrOpt (t u)),
        '("max_bias", AttrOpt (t u))]

__contrib_quantized_conv ::
                         forall a t u .
                           (Tensor t, Fullfilled "__contrib_quantized_conv" '(t, u) a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__contrib_quantized_conv" '(t, u) a -> TensorApply (t u)
__contrib_quantized_conv args
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
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("bias",) . toRaw <$> (args !? #bias :: Maybe (t u)),
               ("min_data",) . toRaw <$> (args !? #min_data :: Maybe (t u)),
               ("max_data",) . toRaw <$> (args !? #max_data :: Maybe (t u)),
               ("min_weight",) . toRaw <$> (args !? #min_weight :: Maybe (t u)),
               ("max_weight",) . toRaw <$> (args !? #max_weight :: Maybe (t u)),
               ("min_bias",) . toRaw <$> (args !? #min_bias :: Maybe (t u)),
               ("max_bias",) . toRaw <$> (args !? #max_bias :: Maybe (t u))]
      in
      applyRaw "_contrib_quantized_conv" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__contrib_quantized_elemwise_add" '(t, u) =
     '[ '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u)),
        '("lhs_min", AttrOpt (t u)), '("lhs_max", AttrOpt (t u)),
        '("rhs_min", AttrOpt (t u)), '("rhs_max", AttrOpt (t u))]

__contrib_quantized_elemwise_add ::
                                 forall a t u .
                                   (Tensor t,
                                    Fullfilled "__contrib_quantized_elemwise_add" '(t, u) a,
                                    HasCallStack, DType u) =>
                                   ArgsHMap "__contrib_quantized_elemwise_add" '(t, u) a ->
                                     TensorApply (t u)
__contrib_quantized_elemwise_add args
  = let scalarArgs
          = catMaybes
              [("min_calib_range",) . showValue <$>
                 (args !? #min_calib_range :: Maybe (Maybe Float)),
               ("max_calib_range",) . showValue <$>
                 (args !? #max_calib_range :: Maybe (Maybe Float))]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u)),
               ("lhs_min",) . toRaw <$> (args !? #lhs_min :: Maybe (t u)),
               ("lhs_max",) . toRaw <$> (args !? #lhs_max :: Maybe (t u)),
               ("rhs_min",) . toRaw <$> (args !? #rhs_min :: Maybe (t u)),
               ("rhs_max",) . toRaw <$> (args !? #rhs_max :: Maybe (t u))]
      in
      applyRaw "_contrib_quantized_elemwise_add" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__contrib_quantized_elemwise_mul" '(t, u) =
     '[ '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)),
        '("enable_float_output", AttrOpt Bool), '("lhs", AttrOpt (t u)),
        '("rhs", AttrOpt (t u)), '("lhs_min", AttrOpt (t u)),
        '("lhs_max", AttrOpt (t u)), '("rhs_min", AttrOpt (t u)),
        '("rhs_max", AttrOpt (t u))]

__contrib_quantized_elemwise_mul ::
                                 forall a t u .
                                   (Tensor t,
                                    Fullfilled "__contrib_quantized_elemwise_mul" '(t, u) a,
                                    HasCallStack, DType u) =>
                                   ArgsHMap "__contrib_quantized_elemwise_mul" '(t, u) a ->
                                     TensorApply (t u)
__contrib_quantized_elemwise_mul args
  = let scalarArgs
          = catMaybes
              [("min_calib_range",) . showValue <$>
                 (args !? #min_calib_range :: Maybe (Maybe Float)),
               ("max_calib_range",) . showValue <$>
                 (args !? #max_calib_range :: Maybe (Maybe Float)),
               ("enable_float_output",) . showValue <$>
                 (args !? #enable_float_output :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u)),
               ("lhs_min",) . toRaw <$> (args !? #lhs_min :: Maybe (t u)),
               ("lhs_max",) . toRaw <$> (args !? #lhs_max :: Maybe (t u)),
               ("rhs_min",) . toRaw <$> (args !? #rhs_min :: Maybe (t u)),
               ("rhs_max",) . toRaw <$> (args !? #rhs_max :: Maybe (t u))]
      in
      applyRaw "_contrib_quantized_elemwise_mul" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_quantized_embedding" '(t, u)
     =
     '[ '("input_dim", AttrReq Int), '("output_dim", AttrReq Int),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"])),
        '("sparse_grad", AttrOpt Bool), '("data", AttrOpt (t u)),
        '("weight", AttrOpt (t u)), '("min_weight", AttrOpt (t u)),
        '("max_weight", AttrOpt (t u))]

__contrib_quantized_embedding ::
                              forall a t u .
                                (Tensor t, Fullfilled "__contrib_quantized_embedding" '(t, u) a,
                                 HasCallStack, DType u) =>
                                ArgsHMap "__contrib_quantized_embedding" '(t, u) a ->
                                  TensorApply (t u)
__contrib_quantized_embedding args
  = let scalarArgs
          = catMaybes
              [("input_dim",) . showValue <$> (args !? #input_dim :: Maybe Int),
               ("output_dim",) . showValue <$> (args !? #output_dim :: Maybe Int),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                           "int8", "uint8"])),
               ("sparse_grad",) . showValue <$>
                 (args !? #sparse_grad :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("min_weight",) . toRaw <$> (args !? #min_weight :: Maybe (t u)),
               ("max_weight",) . toRaw <$> (args !? #max_weight :: Maybe (t u))]
      in
      applyRaw "_contrib_quantized_embedding" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_quantized_flatten" '(t, u) =
     '[ '("data", AttrOpt (t u)), '("min_data", AttrOpt (t u)),
        '("max_data", AttrOpt (t u))]

__contrib_quantized_flatten ::
                            forall a t u .
                              (Tensor t, Fullfilled "__contrib_quantized_flatten" '(t, u) a,
                               HasCallStack, DType u) =>
                              ArgsHMap "__contrib_quantized_flatten" '(t, u) a ->
                                TensorApply (t u)
__contrib_quantized_flatten args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("min_data",) . toRaw <$> (args !? #min_data :: Maybe (t u)),
               ("max_data",) . toRaw <$> (args !? #max_data :: Maybe (t u))]
      in
      applyRaw "_contrib_quantized_flatten" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__contrib_quantized_fully_connected" '(t, u) =
     '[ '("num_hidden", AttrReq Int), '("no_bias", AttrOpt Bool),
        '("flatten", AttrOpt Bool), '("data", AttrOpt (t u)),
        '("weight", AttrOpt (t u)), '("bias", AttrOpt (t u)),
        '("min_data", AttrOpt (t u)), '("max_data", AttrOpt (t u)),
        '("min_weight", AttrOpt (t u)), '("max_weight", AttrOpt (t u)),
        '("min_bias", AttrOpt (t u)), '("max_bias", AttrOpt (t u))]

__contrib_quantized_fully_connected ::
                                    forall a t u .
                                      (Tensor t,
                                       Fullfilled "__contrib_quantized_fully_connected" '(t, u) a,
                                       HasCallStack, DType u) =>
                                      ArgsHMap "__contrib_quantized_fully_connected" '(t, u) a ->
                                        TensorApply (t u)
__contrib_quantized_fully_connected args
  = let scalarArgs
          = catMaybes
              [("num_hidden",) . showValue <$>
                 (args !? #num_hidden :: Maybe Int),
               ("no_bias",) . showValue <$> (args !? #no_bias :: Maybe Bool),
               ("flatten",) . showValue <$> (args !? #flatten :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("bias",) . toRaw <$> (args !? #bias :: Maybe (t u)),
               ("min_data",) . toRaw <$> (args !? #min_data :: Maybe (t u)),
               ("max_data",) . toRaw <$> (args !? #max_data :: Maybe (t u)),
               ("min_weight",) . toRaw <$> (args !? #min_weight :: Maybe (t u)),
               ("max_weight",) . toRaw <$> (args !? #max_weight :: Maybe (t u)),
               ("min_bias",) . toRaw <$> (args !? #min_bias :: Maybe (t u)),
               ("max_bias",) . toRaw <$> (args !? #max_bias :: Maybe (t u))]
      in
      applyRaw "_contrib_quantized_fully_connected" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_quantized_pooling" '(t, u) =
     '[ '("kernel", AttrOpt [Int]),
        '("pool_type", AttrOpt (EnumType '["avg", "lp", "max", "sum"])),
        '("global_pool", AttrOpt Bool), '("cudnn_off", AttrOpt Bool),
        '("pooling_convention",
          AttrOpt (EnumType '["full", "same", "valid"])),
        '("stride", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("p_value", AttrOpt (Maybe Int)),
        '("count_include_pad", AttrOpt (Maybe Bool)),
        '("layout",
          AttrOpt
            (Maybe
               (EnumType '["NCDHW", "NCHW", "NCW", "NDHWC", "NHWC", "NWC"]))),
        '("data", AttrOpt (t u)), '("min_data", AttrOpt (t u)),
        '("max_data", AttrOpt (t u))]

__contrib_quantized_pooling ::
                            forall a t u .
                              (Tensor t, Fullfilled "__contrib_quantized_pooling" '(t, u) a,
                               HasCallStack, DType u) =>
                              ArgsHMap "__contrib_quantized_pooling" '(t, u) a ->
                                TensorApply (t u)
__contrib_quantized_pooling args
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
                    Maybe (EnumType '["full", "same", "valid"])),
               ("stride",) . showValue <$> (args !? #stride :: Maybe [Int]),
               ("pad",) . showValue <$> (args !? #pad :: Maybe [Int]),
               ("p_value",) . showValue <$>
                 (args !? #p_value :: Maybe (Maybe Int)),
               ("count_include_pad",) . showValue <$>
                 (args !? #count_include_pad :: Maybe (Maybe Bool)),
               ("layout",) . showValue <$>
                 (args !? #layout ::
                    Maybe
                      (Maybe
                         (EnumType '["NCDHW", "NCHW", "NCW", "NDHWC", "NHWC", "NWC"])))]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("min_data",) . toRaw <$> (args !? #min_data :: Maybe (t u)),
               ("max_data",) . toRaw <$> (args !? #max_data :: Maybe (t u))]
      in
      applyRaw "_contrib_quantized_pooling" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_requantize" '(t, u) =
     '[ '("out_type", AttrOpt (EnumType '["auto", "int8", "uint8"])),
        '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)),
        '("data", AttrOpt (t u)), '("min_range", AttrOpt (t u)),
        '("max_range", AttrOpt (t u))]

__contrib_requantize ::
                     forall a t u .
                       (Tensor t, Fullfilled "__contrib_requantize" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "__contrib_requantize" '(t, u) a -> TensorApply (t u)
__contrib_requantize args
  = let scalarArgs
          = catMaybes
              [("out_type",) . showValue <$>
                 (args !? #out_type :: Maybe (EnumType '["auto", "int8", "uint8"])),
               ("min_calib_range",) . showValue <$>
                 (args !? #min_calib_range :: Maybe (Maybe Float)),
               ("max_calib_range",) . showValue <$>
                 (args !? #max_calib_range :: Maybe (Maybe Float))]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("min_range",) . toRaw <$> (args !? #min_range :: Maybe (t u)),
               ("max_range",) . toRaw <$> (args !? #max_range :: Maybe (t u))]
      in
      applyRaw "_contrib_requantize" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_round_ste" '(t, u) =
     '[ '("data", AttrOpt (t u))]

__contrib_round_ste ::
                    forall a t u .
                      (Tensor t, Fullfilled "__contrib_round_ste" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "__contrib_round_ste" '(t, u) a -> TensorApply (t u)
__contrib_round_ste args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_contrib_round_ste" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__contrib_sign_ste" '(t, u) =
     '[ '("data", AttrOpt (t u))]

__contrib_sign_ste ::
                   forall a t u .
                     (Tensor t, Fullfilled "__contrib_sign_ste" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__contrib_sign_ste" '(t, u) a -> TensorApply (t u)
__contrib_sign_ste args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_contrib_sign_ste" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__copy" '(t, u) =
     '[ '("data", AttrOpt (t u))]

__copy ::
       forall a t u .
         (Tensor t, Fullfilled "__copy" '(t, u) a, HasCallStack, DType u) =>
         ArgsHMap "__copy" '(t, u) a -> TensorApply (t u)
__copy args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_copy" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__copyto" '(t, u) =
     '[ '("data", AttrOpt (t u))]

__copyto ::
         forall a t u .
           (Tensor t, Fullfilled "__copyto" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "__copyto" '(t, u) a -> TensorApply (t u)
__copyto args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_copyto" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__cvcopyMakeBorder" '(t, u) =
     '[ '("top", AttrReq Int), '("bot", AttrReq Int),
        '("left", AttrReq Int), '("right", AttrReq Int),
        '("type", AttrOpt Int), '("value", AttrOpt Double),
        '("values", AttrOpt [Double]), '("src", AttrOpt (t u))]

__cvcopyMakeBorder ::
                   forall a t u .
                     (Tensor t, Fullfilled "__cvcopyMakeBorder" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__cvcopyMakeBorder" '(t, u) a -> TensorApply (t u)
__cvcopyMakeBorder args
  = let scalarArgs
          = catMaybes
              [("top",) . showValue <$> (args !? #top :: Maybe Int),
               ("bot",) . showValue <$> (args !? #bot :: Maybe Int),
               ("left",) . showValue <$> (args !? #left :: Maybe Int),
               ("right",) . showValue <$> (args !? #right :: Maybe Int),
               ("type",) . showValue <$> (args !? #type :: Maybe Int),
               ("value",) . showValue <$> (args !? #value :: Maybe Double),
               ("values",) . showValue <$> (args !? #values :: Maybe [Double])]
        tensorKeyArgs
          = catMaybes [("src",) . toRaw <$> (args !? #src :: Maybe (t u))]
      in
      applyRaw "_cvcopyMakeBorder" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__cvimdecode" '(t, u) =
     '[ '("flag", AttrOpt Int), '("to_rgb", AttrOpt Bool),
        '("buf", AttrOpt (t u))]

__cvimdecode ::
             forall a t u .
               (Tensor t, Fullfilled "__cvimdecode" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__cvimdecode" '(t, u) a -> TensorApply (t u)
__cvimdecode args
  = let scalarArgs
          = catMaybes
              [("flag",) . showValue <$> (args !? #flag :: Maybe Int),
               ("to_rgb",) . showValue <$> (args !? #to_rgb :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("buf",) . toRaw <$> (args !? #buf :: Maybe (t u))]
      in
      applyRaw "_cvimdecode" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__cvimread" '() =
     '[ '("filename", AttrReq Text), '("flag", AttrOpt Int),
        '("to_rgb", AttrOpt Bool)]

__cvimread ::
           forall a t u .
             (Tensor t, Fullfilled "__cvimread" '() a, HasCallStack, DType u) =>
             ArgsHMap "__cvimread" '() a -> TensorApply (t u)
__cvimread args
  = let scalarArgs
          = catMaybes
              [("filename",) . showValue <$> (args !? #filename :: Maybe Text),
               ("flag",) . showValue <$> (args !? #flag :: Maybe Int),
               ("to_rgb",) . showValue <$> (args !? #to_rgb :: Maybe Bool)]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_cvimread" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__cvimresize" '(t, u) =
     '[ '("w", AttrReq Int), '("h", AttrReq Int),
        '("interp", AttrOpt Int), '("src", AttrOpt (t u))]

__cvimresize ::
             forall a t u .
               (Tensor t, Fullfilled "__cvimresize" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__cvimresize" '(t, u) a -> TensorApply (t u)
__cvimresize args
  = let scalarArgs
          = catMaybes
              [("w",) . showValue <$> (args !? #w :: Maybe Int),
               ("h",) . showValue <$> (args !? #h :: Maybe Int),
               ("interp",) . showValue <$> (args !? #interp :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("src",) . toRaw <$> (args !? #src :: Maybe (t u))]
      in
      applyRaw "_cvimresize" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__div_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__div_scalar ::
             forall a t u .
               (Tensor t, Fullfilled "__div_scalar" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__div_scalar" '(t, u) a -> TensorApply (t u)
__div_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_div_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__equal" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__equal ::
        forall a t u .
          (Tensor t, Fullfilled "__equal" '(t, u) a, HasCallStack,
           DType u) =>
          ArgsHMap "__equal" '(t, u) a -> TensorApply (t u)
__equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__equal_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__equal_scalar ::
               forall a t u .
                 (Tensor t, Fullfilled "__equal_scalar" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__equal_scalar" '(t, u) a -> TensorApply (t u)
__equal_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_equal_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__full" '() =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                 "int64", "int8", "uint8"])),
        '("value", AttrReq Double)]

__full ::
       forall a t u .
         (Tensor t, Fullfilled "__full" '() a, HasCallStack, DType u) =>
         ArgsHMap "__full" '() a -> TensorApply (t u)
__full args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                           "int64", "int8", "uint8"])),
               ("value",) . showValue <$> (args !? #value :: Maybe Double)]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_full" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__grad_add" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__grad_add ::
           forall a t u .
             (Tensor t, Fullfilled "__grad_add" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__grad_add" '(t, u) a -> TensorApply (t u)
__grad_add args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_grad_add" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__greater" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__greater ::
          forall a t u .
            (Tensor t, Fullfilled "__greater" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__greater" '(t, u) a -> TensorApply (t u)
__greater args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_greater" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__greater_equal" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__greater_equal ::
                forall a t u .
                  (Tensor t, Fullfilled "__greater_equal" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__greater_equal" '(t, u) a -> TensorApply (t u)
__greater_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_greater_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__greater_equal_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__greater_equal_scalar ::
                       forall a t u .
                         (Tensor t, Fullfilled "__greater_equal_scalar" '(t, u) a,
                          HasCallStack, DType u) =>
                         ArgsHMap "__greater_equal_scalar" '(t, u) a -> TensorApply (t u)
__greater_equal_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_greater_equal_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__greater_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__greater_scalar ::
                 forall a t u .
                   (Tensor t, Fullfilled "__greater_scalar" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__greater_scalar" '(t, u) a -> TensorApply (t u)
__greater_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_greater_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__histogram" '(t, u) =
     '[ '("bin_cnt", AttrOpt (Maybe Int)), '("range", AttrOpt Int),
        '("data", AttrOpt (t u)), '("bins", AttrOpt (t u))]

__histogram ::
            forall a t u .
              (Tensor t, Fullfilled "__histogram" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "__histogram" '(t, u) a -> TensorApply (t u)
__histogram args
  = let scalarArgs
          = catMaybes
              [("bin_cnt",) . showValue <$>
                 (args !? #bin_cnt :: Maybe (Maybe Int)),
               ("range",) . showValue <$> (args !? #range :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("bins",) . toRaw <$> (args !? #bins :: Maybe (t u))]
      in
      applyRaw "_histogram" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__hypot" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__hypot ::
        forall a t u .
          (Tensor t, Fullfilled "__hypot" '(t, u) a, HasCallStack,
           DType u) =>
          ArgsHMap "__hypot" '(t, u) a -> TensorApply (t u)
__hypot args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_hypot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__hypot_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__hypot_scalar ::
               forall a t u .
                 (Tensor t, Fullfilled "__hypot_scalar" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__hypot_scalar" '(t, u) a -> TensorApply (t u)
__hypot_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_hypot_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__identity_with_attr_like_rhs" '(t, u)
     = '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__identity_with_attr_like_rhs ::
                              forall a t u .
                                (Tensor t, Fullfilled "__identity_with_attr_like_rhs" '(t, u) a,
                                 HasCallStack, DType u) =>
                                ArgsHMap "__identity_with_attr_like_rhs" '(t, u) a ->
                                  TensorApply (t u)
__identity_with_attr_like_rhs args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_identity_with_attr_like_rhs" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__image_adjust_lighting" '(t, u) =
     '[ '("alpha", AttrReq [Float]), '("data", AttrOpt (t u))]

__image_adjust_lighting ::
                        forall a t u .
                          (Tensor t, Fullfilled "__image_adjust_lighting" '(t, u) a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__image_adjust_lighting" '(t, u) a -> TensorApply (t u)
__image_adjust_lighting args
  = let scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> (args !? #alpha :: Maybe [Float])]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_image_adjust_lighting" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__image_crop" '(t, u) =
     '[ '("x", AttrReq Int), '("y", AttrReq Int),
        '("width", AttrReq Int), '("height", AttrReq Int),
        '("data", AttrOpt (t u))]

__image_crop ::
             forall a t u .
               (Tensor t, Fullfilled "__image_crop" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__image_crop" '(t, u) a -> TensorApply (t u)
__image_crop args
  = let scalarArgs
          = catMaybes
              [("x",) . showValue <$> (args !? #x :: Maybe Int),
               ("y",) . showValue <$> (args !? #y :: Maybe Int),
               ("width",) . showValue <$> (args !? #width :: Maybe Int),
               ("height",) . showValue <$> (args !? #height :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_image_crop" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__image_flip_left_right" '(t, u) =
     '[ '("data", AttrOpt (t u))]

__image_flip_left_right ::
                        forall a t u .
                          (Tensor t, Fullfilled "__image_flip_left_right" '(t, u) a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__image_flip_left_right" '(t, u) a -> TensorApply (t u)
__image_flip_left_right args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_image_flip_left_right" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__image_flip_top_bottom" '(t, u) =
     '[ '("data", AttrOpt (t u))]

__image_flip_top_bottom ::
                        forall a t u .
                          (Tensor t, Fullfilled "__image_flip_top_bottom" '(t, u) a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__image_flip_top_bottom" '(t, u) a -> TensorApply (t u)
__image_flip_top_bottom args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_image_flip_top_bottom" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__image_normalize" '(t, u) =
     '[ '("mean", AttrOpt [Float]), '("std", AttrOpt [Float]),
        '("data", AttrOpt (t u))]

__image_normalize ::
                  forall a t u .
                    (Tensor t, Fullfilled "__image_normalize" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__image_normalize" '(t, u) a -> TensorApply (t u)
__image_normalize args
  = let scalarArgs
          = catMaybes
              [("mean",) . showValue <$> (args !? #mean :: Maybe [Float]),
               ("std",) . showValue <$> (args !? #std :: Maybe [Float])]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_image_normalize" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__image_random_brightness" '(t, u) =
     '[ '("min_factor", AttrReq Float), '("max_factor", AttrReq Float),
        '("data", AttrOpt (t u))]

__image_random_brightness ::
                          forall a t u .
                            (Tensor t, Fullfilled "__image_random_brightness" '(t, u) a,
                             HasCallStack, DType u) =>
                            ArgsHMap "__image_random_brightness" '(t, u) a -> TensorApply (t u)
__image_random_brightness args
  = let scalarArgs
          = catMaybes
              [("min_factor",) . showValue <$>
                 (args !? #min_factor :: Maybe Float),
               ("max_factor",) . showValue <$>
                 (args !? #max_factor :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_image_random_brightness" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__image_random_color_jitter" '(t, u) =
     '[ '("brightness", AttrReq Float), '("contrast", AttrReq Float),
        '("saturation", AttrReq Float), '("hue", AttrReq Float),
        '("data", AttrOpt (t u))]

__image_random_color_jitter ::
                            forall a t u .
                              (Tensor t, Fullfilled "__image_random_color_jitter" '(t, u) a,
                               HasCallStack, DType u) =>
                              ArgsHMap "__image_random_color_jitter" '(t, u) a ->
                                TensorApply (t u)
__image_random_color_jitter args
  = let scalarArgs
          = catMaybes
              [("brightness",) . showValue <$>
                 (args !? #brightness :: Maybe Float),
               ("contrast",) . showValue <$> (args !? #contrast :: Maybe Float),
               ("saturation",) . showValue <$>
                 (args !? #saturation :: Maybe Float),
               ("hue",) . showValue <$> (args !? #hue :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_image_random_color_jitter" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__image_random_contrast" '(t, u) =
     '[ '("min_factor", AttrReq Float), '("max_factor", AttrReq Float),
        '("data", AttrOpt (t u))]

__image_random_contrast ::
                        forall a t u .
                          (Tensor t, Fullfilled "__image_random_contrast" '(t, u) a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__image_random_contrast" '(t, u) a -> TensorApply (t u)
__image_random_contrast args
  = let scalarArgs
          = catMaybes
              [("min_factor",) . showValue <$>
                 (args !? #min_factor :: Maybe Float),
               ("max_factor",) . showValue <$>
                 (args !? #max_factor :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_image_random_contrast" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__image_random_flip_left_right" '(t, u) =
     '[ '("data", AttrOpt (t u))]

__image_random_flip_left_right ::
                               forall a t u .
                                 (Tensor t, Fullfilled "__image_random_flip_left_right" '(t, u) a,
                                  HasCallStack, DType u) =>
                                 ArgsHMap "__image_random_flip_left_right" '(t, u) a ->
                                   TensorApply (t u)
__image_random_flip_left_right args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_image_random_flip_left_right" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__image_random_flip_top_bottom" '(t, u) =
     '[ '("data", AttrOpt (t u))]

__image_random_flip_top_bottom ::
                               forall a t u .
                                 (Tensor t, Fullfilled "__image_random_flip_top_bottom" '(t, u) a,
                                  HasCallStack, DType u) =>
                                 ArgsHMap "__image_random_flip_top_bottom" '(t, u) a ->
                                   TensorApply (t u)
__image_random_flip_top_bottom args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_image_random_flip_top_bottom" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__image_random_hue" '(t, u) =
     '[ '("min_factor", AttrReq Float), '("max_factor", AttrReq Float),
        '("data", AttrOpt (t u))]

__image_random_hue ::
                   forall a t u .
                     (Tensor t, Fullfilled "__image_random_hue" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__image_random_hue" '(t, u) a -> TensorApply (t u)
__image_random_hue args
  = let scalarArgs
          = catMaybes
              [("min_factor",) . showValue <$>
                 (args !? #min_factor :: Maybe Float),
               ("max_factor",) . showValue <$>
                 (args !? #max_factor :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_image_random_hue" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__image_random_lighting" '(t, u) =
     '[ '("alpha_std", AttrOpt Float), '("data", AttrOpt (t u))]

__image_random_lighting ::
                        forall a t u .
                          (Tensor t, Fullfilled "__image_random_lighting" '(t, u) a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__image_random_lighting" '(t, u) a -> TensorApply (t u)
__image_random_lighting args
  = let scalarArgs
          = catMaybes
              [("alpha_std",) . showValue <$>
                 (args !? #alpha_std :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_image_random_lighting" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__image_random_saturation" '(t, u) =
     '[ '("min_factor", AttrReq Float), '("max_factor", AttrReq Float),
        '("data", AttrOpt (t u))]

__image_random_saturation ::
                          forall a t u .
                            (Tensor t, Fullfilled "__image_random_saturation" '(t, u) a,
                             HasCallStack, DType u) =>
                            ArgsHMap "__image_random_saturation" '(t, u) a -> TensorApply (t u)
__image_random_saturation args
  = let scalarArgs
          = catMaybes
              [("min_factor",) . showValue <$>
                 (args !? #min_factor :: Maybe Float),
               ("max_factor",) . showValue <$>
                 (args !? #max_factor :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_image_random_saturation" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__image_resize" '(t, u) =
     '[ '("size", AttrOpt [Int]), '("keep_ratio", AttrOpt Bool),
        '("interp", AttrOpt Int), '("data", AttrOpt (t u))]

__image_resize ::
               forall a t u .
                 (Tensor t, Fullfilled "__image_resize" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__image_resize" '(t, u) a -> TensorApply (t u)
__image_resize args
  = let scalarArgs
          = catMaybes
              [("size",) . showValue <$> (args !? #size :: Maybe [Int]),
               ("keep_ratio",) . showValue <$>
                 (args !? #keep_ratio :: Maybe Bool),
               ("interp",) . showValue <$> (args !? #interp :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_image_resize" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__image_to_tensor" '(t, u) =
     '[ '("data", AttrOpt (t u))]

__image_to_tensor ::
                  forall a t u .
                    (Tensor t, Fullfilled "__image_to_tensor" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__image_to_tensor" '(t, u) a -> TensorApply (t u)
__image_to_tensor args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_image_to_tensor" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__imdecode" '(t, u) =
     '[ '("index", AttrOpt Int), '("x0", AttrOpt Int),
        '("y0", AttrOpt Int), '("x1", AttrOpt Int), '("y1", AttrOpt Int),
        '("c", AttrOpt Int), '("size", AttrOpt Int),
        '("mean", AttrOpt (t u))]

__imdecode ::
           forall a t u .
             (Tensor t, Fullfilled "__imdecode" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__imdecode" '(t, u) a -> TensorApply (t u)
__imdecode args
  = let scalarArgs
          = catMaybes
              [("index",) . showValue <$> (args !? #index :: Maybe Int),
               ("x0",) . showValue <$> (args !? #x0 :: Maybe Int),
               ("y0",) . showValue <$> (args !? #y0 :: Maybe Int),
               ("x1",) . showValue <$> (args !? #x1 :: Maybe Int),
               ("y1",) . showValue <$> (args !? #y1 :: Maybe Int),
               ("c",) . showValue <$> (args !? #c :: Maybe Int),
               ("size",) . showValue <$> (args !? #size :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("mean",) . toRaw <$> (args !? #mean :: Maybe (t u))]
      in
      applyRaw "_imdecode" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__lesser" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__lesser ::
         forall a t u .
           (Tensor t, Fullfilled "__lesser" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "__lesser" '(t, u) a -> TensorApply (t u)
__lesser args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_lesser" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__lesser_equal" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__lesser_equal ::
               forall a t u .
                 (Tensor t, Fullfilled "__lesser_equal" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__lesser_equal" '(t, u) a -> TensorApply (t u)
__lesser_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_lesser_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__lesser_equal_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__lesser_equal_scalar ::
                      forall a t u .
                        (Tensor t, Fullfilled "__lesser_equal_scalar" '(t, u) a,
                         HasCallStack, DType u) =>
                        ArgsHMap "__lesser_equal_scalar" '(t, u) a -> TensorApply (t u)
__lesser_equal_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_lesser_equal_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__lesser_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__lesser_scalar ::
                forall a t u .
                  (Tensor t, Fullfilled "__lesser_scalar" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__lesser_scalar" '(t, u) a -> TensorApply (t u)
__lesser_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_lesser_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__linalg_det" '(t, u) =
     '[ '("a", AttrOpt (t u))]

__linalg_det ::
             forall a t u .
               (Tensor t, Fullfilled "__linalg_det" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__linalg_det" '(t, u) a -> TensorApply (t u)
__linalg_det args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_linalg_det" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__linalg_extractdiag" '(t, u) =
     '[ '("offset", AttrOpt Int), '("a", AttrOpt (t u))]

__linalg_extractdiag ::
                     forall a t u .
                       (Tensor t, Fullfilled "__linalg_extractdiag" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "__linalg_extractdiag" '(t, u) a -> TensorApply (t u)
__linalg_extractdiag args
  = let scalarArgs
          = catMaybes
              [("offset",) . showValue <$> (args !? #offset :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_linalg_extractdiag" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__linalg_extracttrian" '(t, u) =
     '[ '("offset", AttrOpt Int), '("lower", AttrOpt Bool),
        '("a", AttrOpt (t u))]

__linalg_extracttrian ::
                      forall a t u .
                        (Tensor t, Fullfilled "__linalg_extracttrian" '(t, u) a,
                         HasCallStack, DType u) =>
                        ArgsHMap "__linalg_extracttrian" '(t, u) a -> TensorApply (t u)
__linalg_extracttrian args
  = let scalarArgs
          = catMaybes
              [("offset",) . showValue <$> (args !? #offset :: Maybe Int),
               ("lower",) . showValue <$> (args !? #lower :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_linalg_extracttrian" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__linalg_gelqf" '(t, u) =
     '[ '("a", AttrOpt (t u))]

__linalg_gelqf ::
               forall a t u .
                 (Tensor t, Fullfilled "__linalg_gelqf" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__linalg_gelqf" '(t, u) a -> TensorApply (t u)
__linalg_gelqf args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_linalg_gelqf" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__linalg_gemm" '(t, u) =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("alpha", AttrOpt Double), '("beta", AttrOpt Double),
        '("axis", AttrOpt Int), '("a", AttrOpt (t u)),
        '("b", AttrOpt (t u)), '("c", AttrOpt (t u))]

__linalg_gemm ::
              forall a t u .
                (Tensor t, Fullfilled "__linalg_gemm" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__linalg_gemm" '(t, u) a -> TensorApply (t u)
__linalg_gemm args
  = let scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$>
                 (args !? #transpose_a :: Maybe Bool),
               ("transpose_b",) . showValue <$>
                 (args !? #transpose_b :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double),
               ("beta",) . showValue <$> (args !? #beta :: Maybe Double),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> (args !? #a :: Maybe (t u)),
               ("b",) . toRaw <$> (args !? #b :: Maybe (t u)),
               ("c",) . toRaw <$> (args !? #c :: Maybe (t u))]
      in
      applyRaw "_linalg_gemm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__linalg_gemm2" '(t, u) =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("alpha", AttrOpt Double), '("axis", AttrOpt Int),
        '("a", AttrOpt (t u)), '("b", AttrOpt (t u))]

__linalg_gemm2 ::
               forall a t u .
                 (Tensor t, Fullfilled "__linalg_gemm2" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__linalg_gemm2" '(t, u) a -> TensorApply (t u)
__linalg_gemm2 args
  = let scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$>
                 (args !? #transpose_a :: Maybe Bool),
               ("transpose_b",) . showValue <$>
                 (args !? #transpose_b :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> (args !? #a :: Maybe (t u)),
               ("b",) . toRaw <$> (args !? #b :: Maybe (t u))]
      in
      applyRaw "_linalg_gemm2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__linalg_inverse" '(t, u) =
     '[ '("a", AttrOpt (t u))]

__linalg_inverse ::
                 forall a t u .
                   (Tensor t, Fullfilled "__linalg_inverse" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__linalg_inverse" '(t, u) a -> TensorApply (t u)
__linalg_inverse args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_linalg_inverse" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__linalg_makediag" '(t, u) =
     '[ '("offset", AttrOpt Int), '("a", AttrOpt (t u))]

__linalg_makediag ::
                  forall a t u .
                    (Tensor t, Fullfilled "__linalg_makediag" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__linalg_makediag" '(t, u) a -> TensorApply (t u)
__linalg_makediag args
  = let scalarArgs
          = catMaybes
              [("offset",) . showValue <$> (args !? #offset :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_linalg_makediag" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__linalg_maketrian" '(t, u) =
     '[ '("offset", AttrOpt Int), '("lower", AttrOpt Bool),
        '("a", AttrOpt (t u))]

__linalg_maketrian ::
                   forall a t u .
                     (Tensor t, Fullfilled "__linalg_maketrian" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__linalg_maketrian" '(t, u) a -> TensorApply (t u)
__linalg_maketrian args
  = let scalarArgs
          = catMaybes
              [("offset",) . showValue <$> (args !? #offset :: Maybe Int),
               ("lower",) . showValue <$> (args !? #lower :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_linalg_maketrian" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__linalg_potrf" '(t, u) =
     '[ '("a", AttrOpt (t u))]

__linalg_potrf ::
               forall a t u .
                 (Tensor t, Fullfilled "__linalg_potrf" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__linalg_potrf" '(t, u) a -> TensorApply (t u)
__linalg_potrf args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_linalg_potrf" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__linalg_potri" '(t, u) =
     '[ '("a", AttrOpt (t u))]

__linalg_potri ::
               forall a t u .
                 (Tensor t, Fullfilled "__linalg_potri" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__linalg_potri" '(t, u) a -> TensorApply (t u)
__linalg_potri args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_linalg_potri" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__linalg_slogdet" '(t, u) =
     '[ '("a", AttrOpt (t u))]

__linalg_slogdet ::
                 forall a t u .
                   (Tensor t, Fullfilled "__linalg_slogdet" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__linalg_slogdet" '(t, u) a -> TensorApply (t u)
__linalg_slogdet args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_linalg_slogdet" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__linalg_sumlogdiag" '(t, u) =
     '[ '("a", AttrOpt (t u))]

__linalg_sumlogdiag ::
                    forall a t u .
                      (Tensor t, Fullfilled "__linalg_sumlogdiag" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "__linalg_sumlogdiag" '(t, u) a -> TensorApply (t u)
__linalg_sumlogdiag args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_linalg_sumlogdiag" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__linalg_syevd" '(t, u) =
     '[ '("a", AttrOpt (t u))]

__linalg_syevd ::
               forall a t u .
                 (Tensor t, Fullfilled "__linalg_syevd" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__linalg_syevd" '(t, u) a -> TensorApply (t u)
__linalg_syevd args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_linalg_syevd" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__linalg_syrk" '(t, u) =
     '[ '("transpose", AttrOpt Bool), '("alpha", AttrOpt Double),
        '("a", AttrOpt (t u))]

__linalg_syrk ::
              forall a t u .
                (Tensor t, Fullfilled "__linalg_syrk" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__linalg_syrk" '(t, u) a -> TensorApply (t u)
__linalg_syrk args
  = let scalarArgs
          = catMaybes
              [("transpose",) . showValue <$> (args !? #transpose :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double)]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_linalg_syrk" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__linalg_trmm" '(t, u) =
     '[ '("transpose", AttrOpt Bool), '("rightside", AttrOpt Bool),
        '("lower", AttrOpt Bool), '("alpha", AttrOpt Double),
        '("a", AttrOpt (t u)), '("b", AttrOpt (t u))]

__linalg_trmm ::
              forall a t u .
                (Tensor t, Fullfilled "__linalg_trmm" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__linalg_trmm" '(t, u) a -> TensorApply (t u)
__linalg_trmm args
  = let scalarArgs
          = catMaybes
              [("transpose",) . showValue <$> (args !? #transpose :: Maybe Bool),
               ("rightside",) . showValue <$> (args !? #rightside :: Maybe Bool),
               ("lower",) . showValue <$> (args !? #lower :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double)]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> (args !? #a :: Maybe (t u)),
               ("b",) . toRaw <$> (args !? #b :: Maybe (t u))]
      in
      applyRaw "_linalg_trmm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__linalg_trsm" '(t, u) =
     '[ '("transpose", AttrOpt Bool), '("rightside", AttrOpt Bool),
        '("lower", AttrOpt Bool), '("alpha", AttrOpt Double),
        '("a", AttrOpt (t u)), '("b", AttrOpt (t u))]

__linalg_trsm ::
              forall a t u .
                (Tensor t, Fullfilled "__linalg_trsm" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__linalg_trsm" '(t, u) a -> TensorApply (t u)
__linalg_trsm args
  = let scalarArgs
          = catMaybes
              [("transpose",) . showValue <$> (args !? #transpose :: Maybe Bool),
               ("rightside",) . showValue <$> (args !? #rightside :: Maybe Bool),
               ("lower",) . showValue <$> (args !? #lower :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double)]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> (args !? #a :: Maybe (t u)),
               ("b",) . toRaw <$> (args !? #b :: Maybe (t u))]
      in
      applyRaw "_linalg_trsm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__linspace" '() =
     '[ '("start", AttrReq Double), '("stop", AttrOpt (Maybe Double)),
        '("step", AttrOpt Double), '("repeat", AttrOpt Int),
        '("infer_range", AttrOpt Bool), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"]))]

__linspace ::
           forall a t u .
             (Tensor t, Fullfilled "__linspace" '() a, HasCallStack, DType u) =>
             ArgsHMap "__linspace" '() a -> TensorApply (t u)
__linspace args
  = let scalarArgs
          = catMaybes
              [("start",) . showValue <$> (args !? #start :: Maybe Double),
               ("stop",) . showValue <$> (args !? #stop :: Maybe (Maybe Double)),
               ("step",) . showValue <$> (args !? #step :: Maybe Double),
               ("repeat",) . showValue <$> (args !? #repeat :: Maybe Int),
               ("infer_range",) . showValue <$>
                 (args !? #infer_range :: Maybe Bool),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                           "int8", "uint8"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_linspace" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__logical_and" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__logical_and ::
              forall a t u .
                (Tensor t, Fullfilled "__logical_and" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__logical_and" '(t, u) a -> TensorApply (t u)
__logical_and args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_logical_and" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__logical_and_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__logical_and_scalar ::
                     forall a t u .
                       (Tensor t, Fullfilled "__logical_and_scalar" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "__logical_and_scalar" '(t, u) a -> TensorApply (t u)
__logical_and_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_logical_and_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__logical_or" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__logical_or ::
             forall a t u .
               (Tensor t, Fullfilled "__logical_or" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__logical_or" '(t, u) a -> TensorApply (t u)
__logical_or args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_logical_or" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__logical_or_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__logical_or_scalar ::
                    forall a t u .
                      (Tensor t, Fullfilled "__logical_or_scalar" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "__logical_or_scalar" '(t, u) a -> TensorApply (t u)
__logical_or_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_logical_or_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__logical_xor" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__logical_xor ::
              forall a t u .
                (Tensor t, Fullfilled "__logical_xor" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__logical_xor" '(t, u) a -> TensorApply (t u)
__logical_xor args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_logical_xor" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__logical_xor_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__logical_xor_scalar ::
                     forall a t u .
                       (Tensor t, Fullfilled "__logical_xor_scalar" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "__logical_xor_scalar" '(t, u) a -> TensorApply (t u)
__logical_xor_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_logical_xor_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__maximum" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__maximum ::
          forall a t u .
            (Tensor t, Fullfilled "__maximum" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__maximum" '(t, u) a -> TensorApply (t u)
__maximum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_maximum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__maximum_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__maximum_scalar ::
                 forall a t u .
                   (Tensor t, Fullfilled "__maximum_scalar" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__maximum_scalar" '(t, u) a -> TensorApply (t u)
__maximum_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_maximum_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__minimum" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__minimum ::
          forall a t u .
            (Tensor t, Fullfilled "__minimum" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__minimum" '(t, u) a -> TensorApply (t u)
__minimum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_minimum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__minimum_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__minimum_scalar ::
                 forall a t u .
                   (Tensor t, Fullfilled "__minimum_scalar" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__minimum_scalar" '(t, u) a -> TensorApply (t u)
__minimum_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_minimum_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__minus_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__minus_scalar ::
               forall a t u .
                 (Tensor t, Fullfilled "__minus_scalar" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__minus_scalar" '(t, u) a -> TensorApply (t u)
__minus_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_minus_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__mod" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__mod ::
      forall a t u .
        (Tensor t, Fullfilled "__mod" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "__mod" '(t, u) a -> TensorApply (t u)
__mod args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_mod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__mod_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__mod_scalar ::
             forall a t u .
               (Tensor t, Fullfilled "__mod_scalar" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__mod_scalar" '(t, u) a -> TensorApply (t u)
__mod_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_mod_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__mp_adamw_update" '(t, u) =
     '[ '("lr", AttrReq Float), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wd", AttrOpt Float), '("eta", AttrReq Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u)), '("mean", AttrOpt (t u)),
        '("var", AttrOpt (t u)), '("weight32", AttrOpt (t u)),
        '("rescale_grad", AttrOpt (t u))]

__mp_adamw_update ::
                  forall a t u .
                    (Tensor t, Fullfilled "__mp_adamw_update" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__mp_adamw_update" '(t, u) a -> TensorApply (t u)
__mp_adamw_update args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("beta1",) . showValue <$> (args !? #beta1 :: Maybe Float),
               ("beta2",) . showValue <$> (args !? #beta2 :: Maybe Float),
               ("epsilon",) . showValue <$> (args !? #epsilon :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("eta",) . showValue <$> (args !? #eta :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("grad",) . toRaw <$> (args !? #grad :: Maybe (t u)),
               ("mean",) . toRaw <$> (args !? #mean :: Maybe (t u)),
               ("var",) . toRaw <$> (args !? #var :: Maybe (t u)),
               ("weight32",) . toRaw <$> (args !? #weight32 :: Maybe (t u)),
               ("rescale_grad",) . toRaw <$>
                 (args !? #rescale_grad :: Maybe (t u))]
      in
      applyRaw "_mp_adamw_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__mul_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__mul_scalar ::
             forall a t u .
               (Tensor t, Fullfilled "__mul_scalar" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__mul_scalar" '(t, u) a -> TensorApply (t u)
__mul_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_mul_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__multi_adamw_update" '(t, u) =
     '[ '("lrs", AttrReq [Float]), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wds", AttrReq [Float]), '("etas", AttrReq [Float]),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("data", AttrOpt [t u])]

__multi_adamw_update ::
                     forall a t u .
                       (Tensor t, Fullfilled "__multi_adamw_update" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "__multi_adamw_update" '(t, u) a -> TensorApply (t u)
__multi_adamw_update args
  = let scalarArgs
          = catMaybes
              [("lrs",) . showValue <$> (args !? #lrs :: Maybe [Float]),
               ("beta1",) . showValue <$> (args !? #beta1 :: Maybe Float),
               ("beta2",) . showValue <$> (args !? #beta2 :: Maybe Float),
               ("epsilon",) . showValue <$> (args !? #epsilon :: Maybe Float),
               ("wds",) . showValue <$> (args !? #wds :: Maybe [Float]),
               ("etas",) . showValue <$> (args !? #etas :: Maybe [Float]),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("num_weights",) . showValue <$>
                 (args !? #num_weights :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "_multi_adamw_update" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__multi_lamb_update" '(t, u) =
     '[ '("learning_rates", AttrReq [Float]), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wds", AttrReq [Float]), '("rescale_grad", AttrOpt Float),
        '("lower_bound", AttrOpt Float), '("upper_bound", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("bias_correction", AttrOpt Bool), '("step_count", AttrReq [Int]),
        '("num_tensors", AttrOpt Int), '("data", AttrOpt [t u])]

__multi_lamb_update ::
                    forall a t u .
                      (Tensor t, Fullfilled "__multi_lamb_update" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "__multi_lamb_update" '(t, u) a -> TensorApply (t u)
__multi_lamb_update args
  = let scalarArgs
          = catMaybes
              [("learning_rates",) . showValue <$>
                 (args !? #learning_rates :: Maybe [Float]),
               ("beta1",) . showValue <$> (args !? #beta1 :: Maybe Float),
               ("beta2",) . showValue <$> (args !? #beta2 :: Maybe Float),
               ("epsilon",) . showValue <$> (args !? #epsilon :: Maybe Float),
               ("wds",) . showValue <$> (args !? #wds :: Maybe [Float]),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("lower_bound",) . showValue <$>
                 (args !? #lower_bound :: Maybe Float),
               ("upper_bound",) . showValue <$>
                 (args !? #upper_bound :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("bias_correction",) . showValue <$>
                 (args !? #bias_correction :: Maybe Bool),
               ("step_count",) . showValue <$>
                 (args !? #step_count :: Maybe [Int]),
               ("num_tensors",) . showValue <$>
                 (args !? #num_tensors :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "_multi_lamb_update" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__multi_mp_adamw_update" '(t, u) =
     '[ '("lrs", AttrReq [Float]), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wds", AttrReq [Float]), '("etas", AttrReq [Float]),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("data", AttrOpt [t u])]

__multi_mp_adamw_update ::
                        forall a t u .
                          (Tensor t, Fullfilled "__multi_mp_adamw_update" '(t, u) a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__multi_mp_adamw_update" '(t, u) a -> TensorApply (t u)
__multi_mp_adamw_update args
  = let scalarArgs
          = catMaybes
              [("lrs",) . showValue <$> (args !? #lrs :: Maybe [Float]),
               ("beta1",) . showValue <$> (args !? #beta1 :: Maybe Float),
               ("beta2",) . showValue <$> (args !? #beta2 :: Maybe Float),
               ("epsilon",) . showValue <$> (args !? #epsilon :: Maybe Float),
               ("wds",) . showValue <$> (args !? #wds :: Maybe [Float]),
               ("etas",) . showValue <$> (args !? #etas :: Maybe [Float]),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("num_weights",) . showValue <$>
                 (args !? #num_weights :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in
      applyRaw "_multi_mp_adamw_update" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__multi_mp_lamb_update" '(t, u) =
     '[ '("learning_rates", AttrReq [Float]), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wds", AttrReq [Float]), '("rescale_grad", AttrOpt Float),
        '("lower_bound", AttrOpt Float), '("upper_bound", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("bias_correction", AttrOpt Bool), '("step_count", AttrReq [Int]),
        '("num_tensors", AttrOpt Int), '("data", AttrOpt [t u])]

__multi_mp_lamb_update ::
                       forall a t u .
                         (Tensor t, Fullfilled "__multi_mp_lamb_update" '(t, u) a,
                          HasCallStack, DType u) =>
                         ArgsHMap "__multi_mp_lamb_update" '(t, u) a -> TensorApply (t u)
__multi_mp_lamb_update args
  = let scalarArgs
          = catMaybes
              [("learning_rates",) . showValue <$>
                 (args !? #learning_rates :: Maybe [Float]),
               ("beta1",) . showValue <$> (args !? #beta1 :: Maybe Float),
               ("beta2",) . showValue <$> (args !? #beta2 :: Maybe Float),
               ("epsilon",) . showValue <$> (args !? #epsilon :: Maybe Float),
               ("wds",) . showValue <$> (args !? #wds :: Maybe [Float]),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("lower_bound",) . showValue <$>
                 (args !? #lower_bound :: Maybe Float),
               ("upper_bound",) . showValue <$>
                 (args !? #upper_bound :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("bias_correction",) . showValue <$>
                 (args !? #bias_correction :: Maybe Bool),
               ("step_count",) . showValue <$>
                 (args !? #step_count :: Maybe [Int]),
               ("num_tensors",) . showValue <$>
                 (args !? #num_tensors :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in
      applyRaw "_multi_mp_lamb_update" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__not_equal" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__not_equal ::
            forall a t u .
              (Tensor t, Fullfilled "__not_equal" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "__not_equal" '(t, u) a -> TensorApply (t u)
__not_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_not_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__not_equal_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__not_equal_scalar ::
                   forall a t u .
                     (Tensor t, Fullfilled "__not_equal_scalar" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__not_equal_scalar" '(t, u) a -> TensorApply (t u)
__not_equal_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_not_equal_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__np_all" '(t, u) =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__np_all ::
         forall a t u .
           (Tensor t, Fullfilled "__np_all" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "__np_all" '(t, u) a -> TensorApply (t u)
__np_all args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_np_all" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__np_any" '(t, u) =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__np_any ::
         forall a t u .
           (Tensor t, Fullfilled "__np_any" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "__np_any" '(t, u) a -> TensorApply (t u)
__np_any args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_np_any" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__np_atleast_1d" '(t, u) =
     '[ '("num_args", AttrReq Int), '("arys", AttrOpt [t u])]

__np_atleast_1d ::
                forall a t u .
                  (Tensor t, Fullfilled "__np_atleast_1d" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__np_atleast_1d" '(t, u) a -> TensorApply (t u)
__np_atleast_1d args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #arys :: Maybe [RawTensor t])
      in applyRaw "_np_atleast_1d" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__np_atleast_2d" '(t, u) =
     '[ '("num_args", AttrReq Int), '("arys", AttrOpt [t u])]

__np_atleast_2d ::
                forall a t u .
                  (Tensor t, Fullfilled "__np_atleast_2d" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__np_atleast_2d" '(t, u) a -> TensorApply (t u)
__np_atleast_2d args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #arys :: Maybe [RawTensor t])
      in applyRaw "_np_atleast_2d" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__np_atleast_3d" '(t, u) =
     '[ '("num_args", AttrReq Int), '("arys", AttrOpt [t u])]

__np_atleast_3d ::
                forall a t u .
                  (Tensor t, Fullfilled "__np_atleast_3d" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__np_atleast_3d" '(t, u) a -> TensorApply (t u)
__np_atleast_3d args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #arys :: Maybe [RawTensor t])
      in applyRaw "_np_atleast_3d" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__np_copy" '(t, u) =
     '[ '("a", AttrOpt (t u))]

__np_copy ::
          forall a t u .
            (Tensor t, Fullfilled "__np_copy" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__np_copy" '(t, u) a -> TensorApply (t u)
__np_copy args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_np_copy" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__np_cumsum" '(t, u) =
     '[ '("axis", AttrOpt (Maybe Int)),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["float16", "float32", "float64", "int32", "int64", "int8"]))),
        '("a", AttrOpt (t u))]

__np_cumsum ::
            forall a t u .
              (Tensor t, Fullfilled "__np_cumsum" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "__np_cumsum" '(t, u) a -> TensorApply (t u)
__np_cumsum args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (Maybe
                         (EnumType
                            '["float16", "float32", "float64", "int32", "int64", "int8"])))]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_np_cumsum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__np_diag" '(t, u) =
     '[ '("k", AttrOpt Int), '("data", AttrOpt (t u))]

__np_diag ::
          forall a t u .
            (Tensor t, Fullfilled "__np_diag" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__np_diag" '(t, u) a -> TensorApply (t u)
__np_diag args
  = let scalarArgs
          = catMaybes [("k",) . showValue <$> (args !? #k :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_np_diag" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__np_diagflat" '(t, u) =
     '[ '("k", AttrOpt Int), '("data", AttrOpt (t u))]

__np_diagflat ::
              forall a t u .
                (Tensor t, Fullfilled "__np_diagflat" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__np_diagflat" '(t, u) a -> TensorApply (t u)
__np_diagflat args
  = let scalarArgs
          = catMaybes [("k",) . showValue <$> (args !? #k :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_np_diagflat" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__np_diagonal" '(t, u) =
     '[ '("offset", AttrOpt Int), '("axis1", AttrOpt Int),
        '("axis2", AttrOpt Int), '("data", AttrOpt (t u))]

__np_diagonal ::
              forall a t u .
                (Tensor t, Fullfilled "__np_diagonal" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__np_diagonal" '(t, u) a -> TensorApply (t u)
__np_diagonal args
  = let scalarArgs
          = catMaybes
              [("offset",) . showValue <$> (args !? #offset :: Maybe Int),
               ("axis1",) . showValue <$> (args !? #axis1 :: Maybe Int),
               ("axis2",) . showValue <$> (args !? #axis2 :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_np_diagonal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__np_dot" '(t, u) =
     '[ '("a", AttrOpt (t u)), '("b", AttrOpt (t u))]

__np_dot ::
         forall a t u .
           (Tensor t, Fullfilled "__np_dot" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "__np_dot" '(t, u) a -> TensorApply (t u)
__np_dot args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> (args !? #a :: Maybe (t u)),
               ("b",) . toRaw <$> (args !? #b :: Maybe (t u))]
      in
      applyRaw "_np_dot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__np_max" '(t, u) =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("initial", AttrOpt (Maybe Double)), '("a", AttrOpt (t u))]

__np_max ::
         forall a t u .
           (Tensor t, Fullfilled "__np_max" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "__np_max" '(t, u) a -> TensorApply (t u)
__np_max args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("initial",) . showValue <$>
                 (args !? #initial :: Maybe (Maybe Double))]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_np_max" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__np_min" '(t, u) =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("initial", AttrOpt (Maybe Double)), '("a", AttrOpt (t u))]

__np_min ::
         forall a t u .
           (Tensor t, Fullfilled "__np_min" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "__np_min" '(t, u) a -> TensorApply (t u)
__np_min args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("initial",) . showValue <$>
                 (args !? #initial :: Maybe (Maybe Double))]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_np_min" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__np_moveaxis" '(t, u) =
     '[ '("source", AttrReq [Int]), '("destination", AttrReq [Int]),
        '("a", AttrOpt (t u))]

__np_moveaxis ::
              forall a t u .
                (Tensor t, Fullfilled "__np_moveaxis" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__np_moveaxis" '(t, u) a -> TensorApply (t u)
__np_moveaxis args
  = let scalarArgs
          = catMaybes
              [("source",) . showValue <$> (args !? #source :: Maybe [Int]),
               ("destination",) . showValue <$>
                 (args !? #destination :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_np_moveaxis" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__np_prod" '(t, u) =
     '[ '("axis", AttrOpt (Maybe [Int])),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["bool", "float16", "float32", "float64", "int32", "int64",
                    "int8"]))),
        '("keepdims", AttrOpt Bool), '("initial", AttrOpt (Maybe Double)),
        '("a", AttrOpt (t u))]

__np_prod ::
          forall a t u .
            (Tensor t, Fullfilled "__np_prod" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__np_prod" '(t, u) a -> TensorApply (t u)
__np_prod args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (Maybe
                         (EnumType
                            '["bool", "float16", "float32", "float64", "int32", "int64",
                              "int8"]))),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("initial",) . showValue <$>
                 (args !? #initial :: Maybe (Maybe Double))]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_np_prod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__np_reshape" '(t, u) =
     '[ '("newshape", AttrReq [Int]), '("order", AttrOpt Text),
        '("a", AttrOpt (t u))]

__np_reshape ::
             forall a t u .
               (Tensor t, Fullfilled "__np_reshape" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__np_reshape" '(t, u) a -> TensorApply (t u)
__np_reshape args
  = let scalarArgs
          = catMaybes
              [("newshape",) . showValue <$> (args !? #newshape :: Maybe [Int]),
               ("order",) . showValue <$> (args !? #order :: Maybe Text)]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_np_reshape" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__np_roll" '(t, u) =
     '[ '("shift", AttrOpt (Maybe [Int])),
        '("axis", AttrOpt (Maybe [Int])), '("data", AttrOpt (t u))]

__np_roll ::
          forall a t u .
            (Tensor t, Fullfilled "__np_roll" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__np_roll" '(t, u) a -> TensorApply (t u)
__np_roll args
  = let scalarArgs
          = catMaybes
              [("shift",) . showValue <$>
                 (args !? #shift :: Maybe (Maybe [Int])),
               ("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int]))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_np_roll" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__np_squeeze" '(t, u) =
     '[ '("axis", AttrOpt (Maybe [Int])), '("a", AttrOpt (t u))]

__np_squeeze ::
             forall a t u .
               (Tensor t, Fullfilled "__np_squeeze" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__np_squeeze" '(t, u) a -> TensorApply (t u)
__np_squeeze args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int]))]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_np_squeeze" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__np_sum" '(t, u) =
     '[ '("axis", AttrOpt (Maybe [Int])),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["bool", "float16", "float32", "float64", "int32", "int64",
                    "int8"]))),
        '("keepdims", AttrOpt Bool), '("initial", AttrOpt (Maybe Double)),
        '("a", AttrOpt (t u))]

__np_sum ::
         forall a t u .
           (Tensor t, Fullfilled "__np_sum" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "__np_sum" '(t, u) a -> TensorApply (t u)
__np_sum args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (Maybe
                         (EnumType
                            '["bool", "float16", "float32", "float64", "int32", "int64",
                              "int8"]))),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("initial",) . showValue <$>
                 (args !? #initial :: Maybe (Maybe Double))]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_np_sum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__np_trace" '(t, u) =
     '[ '("offset", AttrOpt Int), '("axis1", AttrOpt Int),
        '("axis2", AttrOpt Int), '("data", AttrOpt (t u))]

__np_trace ::
           forall a t u .
             (Tensor t, Fullfilled "__np_trace" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__np_trace" '(t, u) a -> TensorApply (t u)
__np_trace args
  = let scalarArgs
          = catMaybes
              [("offset",) . showValue <$> (args !? #offset :: Maybe Int),
               ("axis1",) . showValue <$> (args !? #axis1 :: Maybe Int),
               ("axis2",) . showValue <$> (args !? #axis2 :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_np_trace" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__np_transpose" '(t, u) =
     '[ '("axes", AttrOpt [Int]), '("a", AttrOpt (t u))]

__np_transpose ::
               forall a t u .
                 (Tensor t, Fullfilled "__np_transpose" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__np_transpose" '(t, u) a -> TensorApply (t u)
__np_transpose args
  = let scalarArgs
          = catMaybes
              [("axes",) . showValue <$> (args !? #axes :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_np_transpose" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_absolute" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_absolute ::
               forall a t u .
                 (Tensor t, Fullfilled "__npi_absolute" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__npi_absolute" '(t, u) a -> TensorApply (t u)
__npi_absolute args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_absolute" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_add" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_add ::
          forall a t u .
            (Tensor t, Fullfilled "__npi_add" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__npi_add" '(t, u) a -> TensorApply (t u)
__npi_add args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_npi_add" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_add_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_add_scalar ::
                 forall a t u .
                   (Tensor t, Fullfilled "__npi_add_scalar" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__npi_add_scalar" '(t, u) a -> TensorApply (t u)
__npi_add_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_add_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_arange" '() =
     '[ '("start", AttrReq Double), '("stop", AttrOpt (Maybe Double)),
        '("step", AttrOpt Double), '("repeat", AttrOpt Int),
        '("infer_range", AttrOpt Bool), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"]))]

__npi_arange ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_arange" '() a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_arange" '() a -> TensorApply (t u)
__npi_arange args
  = let scalarArgs
          = catMaybes
              [("start",) . showValue <$> (args !? #start :: Maybe Double),
               ("stop",) . showValue <$> (args !? #stop :: Maybe (Maybe Double)),
               ("step",) . showValue <$> (args !? #step :: Maybe Double),
               ("repeat",) . showValue <$> (args !? #repeat :: Maybe Int),
               ("infer_range",) . showValue <$>
                 (args !? #infer_range :: Maybe Bool),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                           "int8", "uint8"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_arange" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_arccos" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_arccos ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_arccos" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_arccos" '(t, u) a -> TensorApply (t u)
__npi_arccos args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_arccos" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_arccosh" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_arccosh ::
              forall a t u .
                (Tensor t, Fullfilled "__npi_arccosh" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__npi_arccosh" '(t, u) a -> TensorApply (t u)
__npi_arccosh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_arccosh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_arcsin" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_arcsin ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_arcsin" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_arcsin" '(t, u) a -> TensorApply (t u)
__npi_arcsin args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_arcsin" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_arcsinh" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_arcsinh ::
              forall a t u .
                (Tensor t, Fullfilled "__npi_arcsinh" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__npi_arcsinh" '(t, u) a -> TensorApply (t u)
__npi_arcsinh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_arcsinh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_arctan" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_arctan ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_arctan" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_arctan" '(t, u) a -> TensorApply (t u)
__npi_arctan args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_arctan" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_arctan2" '(t, u) =
     '[ '("x1", AttrOpt (t u)), '("x2", AttrOpt (t u))]

__npi_arctan2 ::
              forall a t u .
                (Tensor t, Fullfilled "__npi_arctan2" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__npi_arctan2" '(t, u) a -> TensorApply (t u)
__npi_arctan2 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("x1",) . toRaw <$> (args !? #x1 :: Maybe (t u)),
               ("x2",) . toRaw <$> (args !? #x2 :: Maybe (t u))]
      in
      applyRaw "_npi_arctan2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_arctan2_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_arctan2_scalar ::
                     forall a t u .
                       (Tensor t, Fullfilled "__npi_arctan2_scalar" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "__npi_arctan2_scalar" '(t, u) a -> TensorApply (t u)
__npi_arctan2_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_arctan2_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_arctanh" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_arctanh ::
              forall a t u .
                (Tensor t, Fullfilled "__npi_arctanh" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__npi_arctanh" '(t, u) a -> TensorApply (t u)
__npi_arctanh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_arctanh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_argmax" '(t, u) =
     '[ '("axis", AttrOpt (Maybe Int)), '("keepdims", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_argmax ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_argmax" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_argmax" '(t, u) a -> TensorApply (t Int64)
__npi_argmax args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_argmax" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_argmin" '(t, u) =
     '[ '("axis", AttrOpt (Maybe Int)), '("keepdims", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_argmin ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_argmin" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_argmin" '(t, u) a -> TensorApply (t Int64)
__npi_argmin args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_argmin" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_around" '(t, u) =
     '[ '("decimals", AttrOpt Int), '("x", AttrOpt (t u))]

__npi_around ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_around" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_around" '(t, u) a -> TensorApply (t u)
__npi_around args
  = let scalarArgs
          = catMaybes
              [("decimals",) . showValue <$> (args !? #decimals :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_around" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_average" '(t, u) =
     '[ '("axis", AttrOpt (Maybe [Int])), '("returned", AttrOpt Bool),
        '("weighted", AttrOpt Bool), '("a", AttrOpt (t u)),
        '("weights", AttrOpt (t u))]

__npi_average ::
              forall a t u .
                (Tensor t, Fullfilled "__npi_average" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__npi_average" '(t, u) a -> TensorApply (t u)
__npi_average args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("returned",) . showValue <$> (args !? #returned :: Maybe Bool),
               ("weighted",) . showValue <$> (args !? #weighted :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> (args !? #a :: Maybe (t u)),
               ("weights",) . toRaw <$> (args !? #weights :: Maybe (t u))]
      in
      applyRaw "_npi_average" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_backward_ediff1d" '() = '[]

__npi_backward_ediff1d ::
                       forall a t u .
                         (Tensor t, Fullfilled "__npi_backward_ediff1d" '() a, HasCallStack,
                          DType u) =>
                         ArgsHMap "__npi_backward_ediff1d" '() a -> TensorApply (t u)
__npi_backward_ediff1d args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_backward_ediff1d" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_backward_nan_to_num" '() = '[]

__npi_backward_nan_to_num ::
                          forall a t u .
                            (Tensor t, Fullfilled "__npi_backward_nan_to_num" '() a,
                             HasCallStack, DType u) =>
                            ArgsHMap "__npi_backward_nan_to_num" '() a -> TensorApply (t u)
__npi_backward_nan_to_num args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_backward_nan_to_num" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_backward_polyval" '() = '[]

__npi_backward_polyval ::
                       forall a t u .
                         (Tensor t, Fullfilled "__npi_backward_polyval" '() a, HasCallStack,
                          DType u) =>
                         ArgsHMap "__npi_backward_polyval" '() a -> TensorApply (t u)
__npi_backward_polyval args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_backward_polyval" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_bernoulli" '(t, u) =
     '[ '("prob", AttrReq (Maybe Float)),
        '("logit", AttrReq (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bool", "float16", "float32", "float64", "int32", "uint8"])),
        '("is_logit", AttrReq Bool), '("input1", AttrOpt (t u))]

__npi_bernoulli ::
                forall a t u .
                  (Tensor t, Fullfilled "__npi_bernoulli" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__npi_bernoulli" '(t, u) a -> TensorApply (t u)
__npi_bernoulli args
  = let scalarArgs
          = catMaybes
              [("prob",) . showValue <$> (args !? #prob :: Maybe (Maybe Float)),
               ("logit",) . showValue <$> (args !? #logit :: Maybe (Maybe Float)),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int])),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bool", "float16", "float32", "float64", "int32", "uint8"])),
               ("is_logit",) . showValue <$> (args !? #is_logit :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> (args !? #input1 :: Maybe (t u))]
      in
      applyRaw "_npi_bernoulli" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_bincount" '(t, u) =
     '[ '("minlength", AttrOpt Int), '("has_weights", AttrOpt Bool),
        '("data", AttrOpt (t u)), '("weights", AttrOpt (t u))]

__npi_bincount ::
               forall a t u .
                 (Tensor t, Fullfilled "__npi_bincount" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__npi_bincount" '(t, u) a -> TensorApply (t u)
__npi_bincount args
  = let scalarArgs
          = catMaybes
              [("minlength",) . showValue <$> (args !? #minlength :: Maybe Int),
               ("has_weights",) . showValue <$>
                 (args !? #has_weights :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("weights",) . toRaw <$> (args !? #weights :: Maybe (t u))]
      in
      applyRaw "_npi_bincount" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_bitwise_and" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_bitwise_and ::
                  forall a t u .
                    (Tensor t, Fullfilled "__npi_bitwise_and" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__npi_bitwise_and" '(t, u) a -> TensorApply (t u)
__npi_bitwise_and args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_npi_bitwise_and" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_bitwise_and_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_bitwise_and_scalar ::
                         forall a t u .
                           (Tensor t, Fullfilled "__npi_bitwise_and_scalar" '(t, u) a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__npi_bitwise_and_scalar" '(t, u) a -> TensorApply (t u)
__npi_bitwise_and_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_bitwise_and_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_bitwise_not" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_bitwise_not ::
                  forall a t u .
                    (Tensor t, Fullfilled "__npi_bitwise_not" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__npi_bitwise_not" '(t, u) a -> TensorApply (t u)
__npi_bitwise_not args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_bitwise_not" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_bitwise_or" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_bitwise_or ::
                 forall a t u .
                   (Tensor t, Fullfilled "__npi_bitwise_or" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__npi_bitwise_or" '(t, u) a -> TensorApply (t u)
__npi_bitwise_or args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_npi_bitwise_or" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_bitwise_or_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_bitwise_or_scalar ::
                        forall a t u .
                          (Tensor t, Fullfilled "__npi_bitwise_or_scalar" '(t, u) a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__npi_bitwise_or_scalar" '(t, u) a -> TensorApply (t u)
__npi_bitwise_or_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_bitwise_or_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_bitwise_xor" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_bitwise_xor ::
                  forall a t u .
                    (Tensor t, Fullfilled "__npi_bitwise_xor" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__npi_bitwise_xor" '(t, u) a -> TensorApply (t u)
__npi_bitwise_xor args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_npi_bitwise_xor" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_bitwise_xor_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_bitwise_xor_scalar ::
                         forall a t u .
                           (Tensor t, Fullfilled "__npi_bitwise_xor_scalar" '(t, u) a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__npi_bitwise_xor_scalar" '(t, u) a -> TensorApply (t u)
__npi_bitwise_xor_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_bitwise_xor_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_blackman" '() =
     '[ '("m", AttrOpt Int), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"]))]

__npi_blackman ::
               forall a t u .
                 (Tensor t, Fullfilled "__npi_blackman" '() a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__npi_blackman" '() a -> TensorApply (t u)
__npi_blackman args
  = let scalarArgs
          = catMaybes
              [("m",) . showValue <$> (args !? #m :: Maybe Int),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                           "int8", "uint8"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_blackman" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__npi_boolean_mask_assign_scalar" '(t, u) =
     '[ '("value", AttrOpt Float), '("start_axis", AttrOpt Int),
        '("data", AttrOpt (t u)), '("mask", AttrOpt (t u))]

__npi_boolean_mask_assign_scalar ::
                                 forall a t u .
                                   (Tensor t,
                                    Fullfilled "__npi_boolean_mask_assign_scalar" '(t, u) a,
                                    HasCallStack, DType u) =>
                                   ArgsHMap "__npi_boolean_mask_assign_scalar" '(t, u) a ->
                                     TensorApply (t u)
__npi_boolean_mask_assign_scalar args
  = let scalarArgs
          = catMaybes
              [("value",) . showValue <$> (args !? #value :: Maybe Float),
               ("start_axis",) . showValue <$> (args !? #start_axis :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("mask",) . toRaw <$> (args !? #mask :: Maybe (t u))]
      in
      applyRaw "_npi_boolean_mask_assign_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__npi_boolean_mask_assign_tensor" '(t, u) =
     '[ '("start_axis", AttrOpt Int), '("data", AttrOpt (t u)),
        '("mask", AttrOpt (t u)), '("value", AttrOpt (t u))]

__npi_boolean_mask_assign_tensor ::
                                 forall a t u .
                                   (Tensor t,
                                    Fullfilled "__npi_boolean_mask_assign_tensor" '(t, u) a,
                                    HasCallStack, DType u) =>
                                   ArgsHMap "__npi_boolean_mask_assign_tensor" '(t, u) a ->
                                     TensorApply (t u)
__npi_boolean_mask_assign_tensor args
  = let scalarArgs
          = catMaybes
              [("start_axis",) . showValue <$>
                 (args !? #start_axis :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("mask",) . toRaw <$> (args !? #mask :: Maybe (t u)),
               ("value",) . toRaw <$> (args !? #value :: Maybe (t u))]
      in
      applyRaw "_npi_boolean_mask_assign_tensor" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_broadcast_to" '(t, u) =
     '[ '("shape", AttrOpt [Int]), '("array", AttrOpt (t u))]

__npi_broadcast_to ::
                   forall a t u .
                     (Tensor t, Fullfilled "__npi_broadcast_to" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__npi_broadcast_to" '(t, u) a -> TensorApply (t u)
__npi_broadcast_to args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes
              [("array",) . toRaw <$> (args !? #array :: Maybe (t u))]
      in
      applyRaw "_npi_broadcast_to" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_cbrt" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_cbrt ::
           forall a t u .
             (Tensor t, Fullfilled "__npi_cbrt" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__npi_cbrt" '(t, u) a -> TensorApply (t u)
__npi_cbrt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_cbrt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_ceil" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_ceil ::
           forall a t u .
             (Tensor t, Fullfilled "__npi_ceil" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__npi_ceil" '(t, u) a -> TensorApply (t u)
__npi_ceil args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_ceil" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_choice" '(t, u) =
     '[ '("a", AttrReq Int), '("size", AttrReq Int),
        '("ctx", AttrOpt Text), '("replace", AttrOpt Bool),
        '("weighted", AttrOpt Bool), '("input1", AttrOpt (t u)),
        '("input2", AttrOpt (t u))]

__npi_choice ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_choice" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_choice" '(t, u) a -> TensorApply (t u)
__npi_choice args
  = let scalarArgs
          = catMaybes
              [("a",) . showValue <$> (args !? #a :: Maybe Int),
               ("size",) . showValue <$> (args !? #size :: Maybe Int),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("replace",) . showValue <$> (args !? #replace :: Maybe Bool),
               ("weighted",) . showValue <$> (args !? #weighted :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> (args !? #input1 :: Maybe (t u)),
               ("input2",) . toRaw <$> (args !? #input2 :: Maybe (t u))]
      in
      applyRaw "_npi_choice" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_cholesky" '(t, u) =
     '[ '("a", AttrOpt (t u))]

__npi_cholesky ::
               forall a t u .
                 (Tensor t, Fullfilled "__npi_cholesky" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__npi_cholesky" '(t, u) a -> TensorApply (t u)
__npi_cholesky args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_npi_cholesky" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_column_stack" '(t, u) =
     '[ '("num_args", AttrReq Int), '("data", AttrOpt [t u])]

__npi_column_stack ::
                   forall a t u .
                     (Tensor t, Fullfilled "__npi_column_stack" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__npi_column_stack" '(t, u) a -> TensorApply (t u)
__npi_column_stack args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "_npi_column_stack" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__npi_concatenate" '(t, u) =
     '[ '("num_args", AttrReq Int), '("axis", AttrOpt Int),
        '("data", AttrOpt [t u])]

__npi_concatenate ::
                  forall a t u .
                    (Tensor t, Fullfilled "__npi_concatenate" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__npi_concatenate" '(t, u) a -> TensorApply (t u)
__npi_concatenate args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "_npi_concatenate" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__npi_copysign" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_copysign ::
               forall a t u .
                 (Tensor t, Fullfilled "__npi_copysign" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__npi_copysign" '(t, u) a -> TensorApply (t u)
__npi_copysign args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_npi_copysign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_copysign_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_copysign_scalar ::
                      forall a t u .
                        (Tensor t, Fullfilled "__npi_copysign_scalar" '(t, u) a,
                         HasCallStack, DType u) =>
                        ArgsHMap "__npi_copysign_scalar" '(t, u) a -> TensorApply (t u)
__npi_copysign_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_copysign_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_cos" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_cos ::
          forall a t u .
            (Tensor t, Fullfilled "__npi_cos" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__npi_cos" '(t, u) a -> TensorApply (t u)
__npi_cos args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_cos" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_cosh" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_cosh ::
           forall a t u .
             (Tensor t, Fullfilled "__npi_cosh" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__npi_cosh" '(t, u) a -> TensorApply (t u)
__npi_cosh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_cosh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_degrees" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_degrees ::
              forall a t u .
                (Tensor t, Fullfilled "__npi_degrees" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__npi_degrees" '(t, u) a -> TensorApply (t u)
__npi_degrees args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_degrees" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_delete" '(t, u) =
     '[ '("start", AttrOpt (Maybe Int)), '("stop", AttrOpt (Maybe Int)),
        '("step", AttrOpt (Maybe Int)), '("int_ind", AttrOpt (Maybe Int)),
        '("axis", AttrOpt (Maybe Int)), '("arr", AttrOpt (t u)),
        '("obj", AttrOpt (t u))]

__npi_delete ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_delete" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_delete" '(t, u) a -> TensorApply (t u)
__npi_delete args
  = let scalarArgs
          = catMaybes
              [("start",) . showValue <$> (args !? #start :: Maybe (Maybe Int)),
               ("stop",) . showValue <$> (args !? #stop :: Maybe (Maybe Int)),
               ("step",) . showValue <$> (args !? #step :: Maybe (Maybe Int)),
               ("int_ind",) . showValue <$>
                 (args !? #int_ind :: Maybe (Maybe Int)),
               ("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int))]
        tensorKeyArgs
          = catMaybes
              [("arr",) . toRaw <$> (args !? #arr :: Maybe (t u)),
               ("obj",) . toRaw <$> (args !? #obj :: Maybe (t u))]
      in
      applyRaw "_npi_delete" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_diag_indices_from" '(t, u) =
     '[ '("data", AttrOpt (t u))]

__npi_diag_indices_from ::
                        forall a t u .
                          (Tensor t, Fullfilled "__npi_diag_indices_from" '(t, u) a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__npi_diag_indices_from" '(t, u) a -> TensorApply (t u)
__npi_diag_indices_from args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_diag_indices_from" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_diff" '(t, u) =
     '[ '("n", AttrOpt Int), '("axis", AttrOpt Int),
        '("a", AttrOpt (t u))]

__npi_diff ::
           forall a t u .
             (Tensor t, Fullfilled "__npi_diff" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__npi_diff" '(t, u) a -> TensorApply (t u)
__npi_diff args
  = let scalarArgs
          = catMaybes
              [("n",) . showValue <$> (args !? #n :: Maybe Int),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_npi_diff" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_dsplit" '(t, u) =
     '[ '("indices", AttrReq [Int]), '("axis", AttrOpt Int),
        '("squeeze_axis", AttrOpt Bool), '("sections", AttrOpt Int),
        '("data", AttrOpt (t u))]

__npi_dsplit ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_dsplit" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_dsplit" '(t, u) a -> TensorApply (t u)
__npi_dsplit args
  = let scalarArgs
          = catMaybes
              [("indices",) . showValue <$> (args !? #indices :: Maybe [Int]),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("squeeze_axis",) . showValue <$>
                 (args !? #squeeze_axis :: Maybe Bool),
               ("sections",) . showValue <$> (args !? #sections :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_dsplit" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_dstack" '(t, u) =
     '[ '("num_args", AttrReq Int), '("dim", AttrOpt Int),
        '("data", AttrOpt [t u])]

__npi_dstack ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_dstack" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_dstack" '(t, u) a -> TensorApply (t u)
__npi_dstack args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("dim",) . showValue <$> (args !? #dim :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "_npi_dstack" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__npi_ediff1d" '(t, u) =
     '[ '("to_begin_arr_given", AttrOpt Bool),
        '("to_end_arr_given", AttrOpt Bool),
        '("to_begin_scalar", AttrOpt (Maybe Double)),
        '("to_end_scalar", AttrOpt (Maybe Double)),
        '("input1", AttrOpt (t u)), '("input2", AttrOpt (t u)),
        '("input3", AttrOpt (t u))]

__npi_ediff1d ::
              forall a t u .
                (Tensor t, Fullfilled "__npi_ediff1d" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__npi_ediff1d" '(t, u) a -> TensorApply (t u)
__npi_ediff1d args
  = let scalarArgs
          = catMaybes
              [("to_begin_arr_given",) . showValue <$>
                 (args !? #to_begin_arr_given :: Maybe Bool),
               ("to_end_arr_given",) . showValue <$>
                 (args !? #to_end_arr_given :: Maybe Bool),
               ("to_begin_scalar",) . showValue <$>
                 (args !? #to_begin_scalar :: Maybe (Maybe Double)),
               ("to_end_scalar",) . showValue <$>
                 (args !? #to_end_scalar :: Maybe (Maybe Double))]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> (args !? #input1 :: Maybe (t u)),
               ("input2",) . toRaw <$> (args !? #input2 :: Maybe (t u)),
               ("input3",) . toRaw <$> (args !? #input3 :: Maybe (t u))]
      in
      applyRaw "_npi_ediff1d" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_eig" '(t, u) =
     '[ '("a", AttrOpt (t u))]

__npi_eig ::
          forall a t u .
            (Tensor t, Fullfilled "__npi_eig" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__npi_eig" '(t, u) a -> TensorApply (t u)
__npi_eig args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_npi_eig" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_eigh" '(t, u) =
     '[ '("uPLO", AttrOpt Int), '("a", AttrOpt (t u))]

__npi_eigh ::
           forall a t u .
             (Tensor t, Fullfilled "__npi_eigh" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__npi_eigh" '(t, u) a -> TensorApply (t u)
__npi_eigh args
  = let scalarArgs
          = catMaybes
              [("uPLO",) . showValue <$> (args !? #uPLO :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_npi_eigh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_eigvals" '(t, u) =
     '[ '("a", AttrOpt (t u))]

__npi_eigvals ::
              forall a t u .
                (Tensor t, Fullfilled "__npi_eigvals" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__npi_eigvals" '(t, u) a -> TensorApply (t u)
__npi_eigvals args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_npi_eigvals" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_eigvalsh" '(t, u) =
     '[ '("uPLO", AttrOpt Int), '("a", AttrOpt (t u))]

__npi_eigvalsh ::
               forall a t u .
                 (Tensor t, Fullfilled "__npi_eigvalsh" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__npi_eigvalsh" '(t, u) a -> TensorApply (t u)
__npi_eigvalsh args
  = let scalarArgs
          = catMaybes
              [("uPLO",) . showValue <$> (args !? #uPLO :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_npi_eigvalsh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_einsum" '(t, u) =
     '[ '("num_args", AttrReq Int), '("subscripts", AttrOpt Text),
        '("optimize", AttrOpt Int), '("data", AttrOpt [t u])]

__npi_einsum ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_einsum" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_einsum" '(t, u) a -> TensorApply (t u)
__npi_einsum args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("subscripts",) . showValue <$>
                 (args !? #subscripts :: Maybe Text),
               ("optimize",) . showValue <$> (args !? #optimize :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "_npi_einsum" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__npi_equal" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_equal ::
            forall a t u .
              (Tensor t, Fullfilled "__npi_equal" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "__npi_equal" '(t, u) a -> TensorApply (t Bool)
__npi_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_npi_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_equal_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_equal_scalar ::
                   forall a t u .
                     (Tensor t, Fullfilled "__npi_equal_scalar" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__npi_equal_scalar" '(t, u) a -> TensorApply (t Bool)
__npi_equal_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_equal_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_exp" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_exp ::
          forall a t u .
            (Tensor t, Fullfilled "__npi_exp" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__npi_exp" '(t, u) a -> TensorApply (t u)
__npi_exp args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_exp" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_expm1" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_expm1 ::
            forall a t u .
              (Tensor t, Fullfilled "__npi_expm1" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "__npi_expm1" '(t, u) a -> TensorApply (t u)
__npi_expm1 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_expm1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_exponential" '(t, u) =
     '[ '("scale", AttrOpt (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text),
        '("input1", AttrOpt (t u))]

__npi_exponential ::
                  forall a t u .
                    (Tensor t, Fullfilled "__npi_exponential" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__npi_exponential" '(t, u) a -> TensorApply (t u)
__npi_exponential args
  = let scalarArgs
          = catMaybes
              [("scale",) . showValue <$>
                 (args !? #scale :: Maybe (Maybe Float)),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int])),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text)]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> (args !? #input1 :: Maybe (t u))]
      in
      applyRaw "_npi_exponential" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_fix" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_fix ::
          forall a t u .
            (Tensor t, Fullfilled "__npi_fix" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__npi_fix" '(t, u) a -> TensorApply (t u)
__npi_fix args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_fix" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_flip" '(t, u) =
     '[ '("axis", AttrReq [Int]), '("data", AttrOpt (t u))]

__npi_flip ::
           forall a t u .
             (Tensor t, Fullfilled "__npi_flip" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__npi_flip" '(t, u) a -> TensorApply (t u)
__npi_flip args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_flip" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_floor" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_floor ::
            forall a t u .
              (Tensor t, Fullfilled "__npi_floor" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "__npi_floor" '(t, u) a -> TensorApply (t u)
__npi_floor args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_floor" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_full_like" '(t, u) =
     '[ '("fill_value", AttrReq Double), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                    "int64", "int8", "uint8"]))),
        '("a", AttrOpt (t u))]

__npi_full_like ::
                forall a t u .
                  (Tensor t, Fullfilled "__npi_full_like" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__npi_full_like" '(t, u) a -> TensorApply (t u)
__npi_full_like args
  = let scalarArgs
          = catMaybes
              [("fill_value",) . showValue <$>
                 (args !? #fill_value :: Maybe Double),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (Maybe
                         (EnumType
                            '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                              "int64", "int8", "uint8"])))]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_npi_full_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_gamma" '(t, u) =
     '[ '("shape", AttrReq (Maybe Float)),
        '("scale", AttrReq (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text),
        '("dtype", AttrOpt (EnumType '["float16", "float32", "float64"])),
        '("input1", AttrOpt (t u)), '("input2", AttrOpt (t u))]

__npi_gamma ::
            forall a t u .
              (Tensor t, Fullfilled "__npi_gamma" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "__npi_gamma" '(t, u) a -> TensorApply (t u)
__npi_gamma args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$>
                 (args !? #shape :: Maybe (Maybe Float)),
               ("scale",) . showValue <$> (args !? #scale :: Maybe (Maybe Float)),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int])),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["float16", "float32", "float64"]))]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> (args !? #input1 :: Maybe (t u)),
               ("input2",) . toRaw <$> (args !? #input2 :: Maybe (t u))]
      in
      applyRaw "_npi_gamma" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_greater" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_greater ::
              forall a t u .
                (Tensor t, Fullfilled "__npi_greater" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__npi_greater" '(t, u) a -> TensorApply (t Bool)
__npi_greater args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_npi_greater" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_greater_equal" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_greater_equal ::
                    forall a t u .
                      (Tensor t, Fullfilled "__npi_greater_equal" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "__npi_greater_equal" '(t, u) a -> TensorApply (t Bool)
__npi_greater_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_npi_greater_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_greater_equal_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_greater_equal_scalar ::
                           forall a t u .
                             (Tensor t, Fullfilled "__npi_greater_equal_scalar" '(t, u) a,
                              HasCallStack, DType u) =>
                             ArgsHMap "__npi_greater_equal_scalar" '(t, u) a ->
                               TensorApply (t Bool)
__npi_greater_equal_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_greater_equal_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_greater_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_greater_scalar ::
                     forall a t u .
                       (Tensor t, Fullfilled "__npi_greater_scalar" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "__npi_greater_scalar" '(t, u) a -> TensorApply (t Bool)
__npi_greater_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_greater_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_gumbel" '(t, u) =
     '[ '("loc", AttrReq (Maybe Float)),
        '("scale", AttrReq (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text),
        '("input1", AttrOpt (t u)), '("input2", AttrOpt (t u))]

__npi_gumbel ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_gumbel" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_gumbel" '(t, u) a -> TensorApply (t u)
__npi_gumbel args
  = let scalarArgs
          = catMaybes
              [("loc",) . showValue <$> (args !? #loc :: Maybe (Maybe Float)),
               ("scale",) . showValue <$> (args !? #scale :: Maybe (Maybe Float)),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int])),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text)]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> (args !? #input1 :: Maybe (t u)),
               ("input2",) . toRaw <$> (args !? #input2 :: Maybe (t u))]
      in
      applyRaw "_npi_gumbel" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_hamming" '() =
     '[ '("m", AttrOpt Int), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"]))]

__npi_hamming ::
              forall a t u .
                (Tensor t, Fullfilled "__npi_hamming" '() a, HasCallStack,
                 DType u) =>
                ArgsHMap "__npi_hamming" '() a -> TensorApply (t u)
__npi_hamming args
  = let scalarArgs
          = catMaybes
              [("m",) . showValue <$> (args !? #m :: Maybe Int),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                           "int8", "uint8"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_hamming" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_hanning" '() =
     '[ '("m", AttrOpt Int), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"]))]

__npi_hanning ::
              forall a t u .
                (Tensor t, Fullfilled "__npi_hanning" '() a, HasCallStack,
                 DType u) =>
                ArgsHMap "__npi_hanning" '() a -> TensorApply (t u)
__npi_hanning args
  = let scalarArgs
          = catMaybes
              [("m",) . showValue <$> (args !? #m :: Maybe Int),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                           "int8", "uint8"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_hanning" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_hsplit" '(t, u) =
     '[ '("indices", AttrReq [Int]), '("axis", AttrOpt Int),
        '("squeeze_axis", AttrOpt Bool), '("sections", AttrOpt Int),
        '("data", AttrOpt (t u))]

__npi_hsplit ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_hsplit" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_hsplit" '(t, u) a -> TensorApply (t u)
__npi_hsplit args
  = let scalarArgs
          = catMaybes
              [("indices",) . showValue <$> (args !? #indices :: Maybe [Int]),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("squeeze_axis",) . showValue <$>
                 (args !? #squeeze_axis :: Maybe Bool),
               ("sections",) . showValue <$> (args !? #sections :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_hsplit" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_hsplit_backward" '() = '[]

__npi_hsplit_backward ::
                      forall a t u .
                        (Tensor t, Fullfilled "__npi_hsplit_backward" '() a, HasCallStack,
                         DType u) =>
                        ArgsHMap "__npi_hsplit_backward" '() a -> TensorApply (t u)
__npi_hsplit_backward args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_hsplit_backward" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_hstack" '(t, u) =
     '[ '("num_args", AttrReq Int), '("dim", AttrOpt Int),
        '("data", AttrOpt [t u])]

__npi_hstack ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_hstack" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_hstack" '(t, u) a -> TensorApply (t u)
__npi_hstack args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("dim",) . showValue <$> (args !? #dim :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "_npi_hstack" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__npi_hypot" '(t, u) =
     '[ '("x1", AttrOpt (t u)), '("x2", AttrOpt (t u))]

__npi_hypot ::
            forall a t u .
              (Tensor t, Fullfilled "__npi_hypot" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "__npi_hypot" '(t, u) a -> TensorApply (t u)
__npi_hypot args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("x1",) . toRaw <$> (args !? #x1 :: Maybe (t u)),
               ("x2",) . toRaw <$> (args !? #x2 :: Maybe (t u))]
      in
      applyRaw "_npi_hypot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_identity" '() =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                 "int64", "int8", "uint8"]))]

__npi_identity ::
               forall a t u .
                 (Tensor t, Fullfilled "__npi_identity" '() a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__npi_identity" '() a -> TensorApply (t u)
__npi_identity args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                           "int64", "int8", "uint8"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_identity" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_indices" '() =
     '[ '("dimensions", AttrReq [Int]),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"])),
        '("ctx", AttrOpt Text)]

__npi_indices ::
              forall a t u .
                (Tensor t, Fullfilled "__npi_indices" '() a, HasCallStack,
                 DType u) =>
                ArgsHMap "__npi_indices" '() a -> TensorApply (t u)
__npi_indices args
  = let scalarArgs
          = catMaybes
              [("dimensions",) . showValue <$>
                 (args !? #dimensions :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                           "int8", "uint8"])),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text)]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_indices" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_insert_scalar" '(t, u) =
     '[ '("val", AttrOpt (Maybe Double)),
        '("start", AttrOpt (Maybe Int)), '("stop", AttrOpt (Maybe Int)),
        '("step", AttrOpt (Maybe Int)), '("int_ind", AttrOpt (Maybe Int)),
        '("axis", AttrOpt (Maybe Int)), '("arr", AttrOpt (t u)),
        '("values", AttrOpt (t u))]

__npi_insert_scalar ::
                    forall a t u .
                      (Tensor t, Fullfilled "__npi_insert_scalar" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "__npi_insert_scalar" '(t, u) a -> TensorApply (t u)
__npi_insert_scalar args
  = let scalarArgs
          = catMaybes
              [("val",) . showValue <$> (args !? #val :: Maybe (Maybe Double)),
               ("start",) . showValue <$> (args !? #start :: Maybe (Maybe Int)),
               ("stop",) . showValue <$> (args !? #stop :: Maybe (Maybe Int)),
               ("step",) . showValue <$> (args !? #step :: Maybe (Maybe Int)),
               ("int_ind",) . showValue <$>
                 (args !? #int_ind :: Maybe (Maybe Int)),
               ("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int))]
        tensorKeyArgs
          = catMaybes
              [("arr",) . toRaw <$> (args !? #arr :: Maybe (t u)),
               ("values",) . toRaw <$> (args !? #values :: Maybe (t u))]
      in
      applyRaw "_npi_insert_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_insert_slice" '(t, u) =
     '[ '("val", AttrOpt (Maybe Double)),
        '("start", AttrOpt (Maybe Int)), '("stop", AttrOpt (Maybe Int)),
        '("step", AttrOpt (Maybe Int)), '("int_ind", AttrOpt (Maybe Int)),
        '("axis", AttrOpt (Maybe Int)), '("arr", AttrOpt (t u)),
        '("values", AttrOpt (t u))]

__npi_insert_slice ::
                   forall a t u .
                     (Tensor t, Fullfilled "__npi_insert_slice" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__npi_insert_slice" '(t, u) a -> TensorApply (t u)
__npi_insert_slice args
  = let scalarArgs
          = catMaybes
              [("val",) . showValue <$> (args !? #val :: Maybe (Maybe Double)),
               ("start",) . showValue <$> (args !? #start :: Maybe (Maybe Int)),
               ("stop",) . showValue <$> (args !? #stop :: Maybe (Maybe Int)),
               ("step",) . showValue <$> (args !? #step :: Maybe (Maybe Int)),
               ("int_ind",) . showValue <$>
                 (args !? #int_ind :: Maybe (Maybe Int)),
               ("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int))]
        tensorKeyArgs
          = catMaybes
              [("arr",) . toRaw <$> (args !? #arr :: Maybe (t u)),
               ("values",) . toRaw <$> (args !? #values :: Maybe (t u))]
      in
      applyRaw "_npi_insert_slice" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_insert_tensor" '(t, u) =
     '[ '("val", AttrOpt (Maybe Double)),
        '("start", AttrOpt (Maybe Int)), '("stop", AttrOpt (Maybe Int)),
        '("step", AttrOpt (Maybe Int)), '("int_ind", AttrOpt (Maybe Int)),
        '("axis", AttrOpt (Maybe Int)), '("arr", AttrOpt (t u)),
        '("values", AttrOpt (t u)), '("obj", AttrOpt (t u))]

__npi_insert_tensor ::
                    forall a t u .
                      (Tensor t, Fullfilled "__npi_insert_tensor" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "__npi_insert_tensor" '(t, u) a -> TensorApply (t u)
__npi_insert_tensor args
  = let scalarArgs
          = catMaybes
              [("val",) . showValue <$> (args !? #val :: Maybe (Maybe Double)),
               ("start",) . showValue <$> (args !? #start :: Maybe (Maybe Int)),
               ("stop",) . showValue <$> (args !? #stop :: Maybe (Maybe Int)),
               ("step",) . showValue <$> (args !? #step :: Maybe (Maybe Int)),
               ("int_ind",) . showValue <$>
                 (args !? #int_ind :: Maybe (Maybe Int)),
               ("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int))]
        tensorKeyArgs
          = catMaybes
              [("arr",) . toRaw <$> (args !? #arr :: Maybe (t u)),
               ("values",) . toRaw <$> (args !? #values :: Maybe (t u)),
               ("obj",) . toRaw <$> (args !? #obj :: Maybe (t u))]
      in
      applyRaw "_npi_insert_tensor" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_isfinite" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_isfinite ::
               forall a t u .
                 (Tensor t, Fullfilled "__npi_isfinite" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__npi_isfinite" '(t, u) a -> TensorApply (t u)
__npi_isfinite args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_isfinite" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_isinf" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_isinf ::
            forall a t u .
              (Tensor t, Fullfilled "__npi_isinf" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "__npi_isinf" '(t, u) a -> TensorApply (t u)
__npi_isinf args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_isinf" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_isnan" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_isnan ::
            forall a t u .
              (Tensor t, Fullfilled "__npi_isnan" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "__npi_isnan" '(t, u) a -> TensorApply (t u)
__npi_isnan args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_isnan" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_isneginf" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_isneginf ::
               forall a t u .
                 (Tensor t, Fullfilled "__npi_isneginf" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__npi_isneginf" '(t, u) a -> TensorApply (t u)
__npi_isneginf args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_isneginf" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_isposinf" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_isposinf ::
               forall a t u .
                 (Tensor t, Fullfilled "__npi_isposinf" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__npi_isposinf" '(t, u) a -> TensorApply (t u)
__npi_isposinf args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_isposinf" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_lcm" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_lcm ::
          forall a t u .
            (Tensor t, Fullfilled "__npi_lcm" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__npi_lcm" '(t, u) a -> TensorApply (t u)
__npi_lcm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_npi_lcm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_lcm_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_lcm_scalar ::
                 forall a t u .
                   (Tensor t, Fullfilled "__npi_lcm_scalar" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__npi_lcm_scalar" '(t, u) a -> TensorApply (t u)
__npi_lcm_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_lcm_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_ldexp" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_ldexp ::
            forall a t u .
              (Tensor t, Fullfilled "__npi_ldexp" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "__npi_ldexp" '(t, u) a -> TensorApply (t u)
__npi_ldexp args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_npi_ldexp" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_ldexp_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_ldexp_scalar ::
                   forall a t u .
                     (Tensor t, Fullfilled "__npi_ldexp_scalar" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__npi_ldexp_scalar" '(t, u) a -> TensorApply (t u)
__npi_ldexp_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_ldexp_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_less" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_less ::
           forall a t u .
             (Tensor t, Fullfilled "__npi_less" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__npi_less" '(t, u) a -> TensorApply (t Bool)
__npi_less args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_npi_less" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_less_equal" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_less_equal ::
                 forall a t u .
                   (Tensor t, Fullfilled "__npi_less_equal" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__npi_less_equal" '(t, u) a -> TensorApply (t Bool)
__npi_less_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_npi_less_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_less_equal_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_less_equal_scalar ::
                        forall a t u .
                          (Tensor t, Fullfilled "__npi_less_equal_scalar" '(t, u) a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__npi_less_equal_scalar" '(t, u) a ->
                            TensorApply (t Bool)
__npi_less_equal_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_less_equal_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_less_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_less_scalar ::
                  forall a t u .
                    (Tensor t, Fullfilled "__npi_less_scalar" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__npi_less_scalar" '(t, u) a -> TensorApply (t Bool)
__npi_less_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_less_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_log" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_log ::
          forall a t u .
            (Tensor t, Fullfilled "__npi_log" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__npi_log" '(t, u) a -> TensorApply (t u)
__npi_log args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_log" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_log10" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_log10 ::
            forall a t u .
              (Tensor t, Fullfilled "__npi_log10" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "__npi_log10" '(t, u) a -> TensorApply (t u)
__npi_log10 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_log10" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_log1p" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_log1p ::
            forall a t u .
              (Tensor t, Fullfilled "__npi_log1p" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "__npi_log1p" '(t, u) a -> TensorApply (t u)
__npi_log1p args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_log1p" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_log2" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_log2 ::
           forall a t u .
             (Tensor t, Fullfilled "__npi_log2" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__npi_log2" '(t, u) a -> TensorApply (t u)
__npi_log2 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_log2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_logical_not" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_logical_not ::
                  forall a t u .
                    (Tensor t, Fullfilled "__npi_logical_not" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__npi_logical_not" '(t, u) a -> TensorApply (t u)
__npi_logical_not args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_logical_not" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_logistic" '(t, u) =
     '[ '("loc", AttrReq (Maybe Float)),
        '("scale", AttrReq (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text),
        '("input1", AttrOpt (t u)), '("input2", AttrOpt (t u))]

__npi_logistic ::
               forall a t u .
                 (Tensor t, Fullfilled "__npi_logistic" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__npi_logistic" '(t, u) a -> TensorApply (t u)
__npi_logistic args
  = let scalarArgs
          = catMaybes
              [("loc",) . showValue <$> (args !? #loc :: Maybe (Maybe Float)),
               ("scale",) . showValue <$> (args !? #scale :: Maybe (Maybe Float)),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int])),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text)]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> (args !? #input1 :: Maybe (t u)),
               ("input2",) . toRaw <$> (args !? #input2 :: Maybe (t u))]
      in
      applyRaw "_npi_logistic" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_logspace" '() =
     '[ '("start", AttrReq Double), '("stop", AttrReq Double),
        '("num", AttrReq Int), '("endpoint", AttrOpt Bool),
        '("base", AttrOpt Double), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"]))]

__npi_logspace ::
               forall a t u .
                 (Tensor t, Fullfilled "__npi_logspace" '() a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__npi_logspace" '() a -> TensorApply (t u)
__npi_logspace args
  = let scalarArgs
          = catMaybes
              [("start",) . showValue <$> (args !? #start :: Maybe Double),
               ("stop",) . showValue <$> (args !? #stop :: Maybe Double),
               ("num",) . showValue <$> (args !? #num :: Maybe Int),
               ("endpoint",) . showValue <$> (args !? #endpoint :: Maybe Bool),
               ("base",) . showValue <$> (args !? #base :: Maybe Double),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                           "int8", "uint8"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_logspace" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_matmul" '(t, u) =
     '[ '("a", AttrOpt (t u)), '("b", AttrOpt (t u))]

__npi_matmul ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_matmul" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_matmul" '(t, u) a -> TensorApply (t u)
__npi_matmul args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> (args !? #a :: Maybe (t u)),
               ("b",) . toRaw <$> (args !? #b :: Maybe (t u))]
      in
      applyRaw "_npi_matmul" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_mean" '(t, u) =
     '[ '("axis", AttrOpt (Maybe [Int])),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["bool", "float16", "float32", "float64", "int32", "int64",
                    "int8"]))),
        '("keepdims", AttrOpt Bool), '("initial", AttrOpt (Maybe Double)),
        '("a", AttrOpt (t u))]

__npi_mean ::
           forall a t u .
             (Tensor t, Fullfilled "__npi_mean" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__npi_mean" '(t, u) a -> TensorApply (t u)
__npi_mean args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (Maybe
                         (EnumType
                            '["bool", "float16", "float32", "float64", "int32", "int64",
                              "int8"]))),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("initial",) . showValue <$>
                 (args !? #initial :: Maybe (Maybe Double))]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_npi_mean" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_mod" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_mod ::
          forall a t u .
            (Tensor t, Fullfilled "__npi_mod" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__npi_mod" '(t, u) a -> TensorApply (t u)
__npi_mod args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_npi_mod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_mod_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_mod_scalar ::
                 forall a t u .
                   (Tensor t, Fullfilled "__npi_mod_scalar" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__npi_mod_scalar" '(t, u) a -> TensorApply (t u)
__npi_mod_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_mod_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_multinomial" '(t, u) =
     '[ '("n", AttrReq Int), '("pvals", AttrOpt Int),
        '("size", AttrOpt (Maybe [Int])), '("a", AttrOpt (t u))]

__npi_multinomial ::
                  forall a t u .
                    (Tensor t, Fullfilled "__npi_multinomial" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__npi_multinomial" '(t, u) a -> TensorApply (t u)
__npi_multinomial args
  = let scalarArgs
          = catMaybes
              [("n",) . showValue <$> (args !? #n :: Maybe Int),
               ("pvals",) . showValue <$> (args !? #pvals :: Maybe Int),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int]))]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_npi_multinomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_multiply" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_multiply ::
               forall a t u .
                 (Tensor t, Fullfilled "__npi_multiply" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__npi_multiply" '(t, u) a -> TensorApply (t u)
__npi_multiply args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_npi_multiply" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_multiply_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_multiply_scalar ::
                      forall a t u .
                        (Tensor t, Fullfilled "__npi_multiply_scalar" '(t, u) a,
                         HasCallStack, DType u) =>
                        ArgsHMap "__npi_multiply_scalar" '(t, u) a -> TensorApply (t u)
__npi_multiply_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_multiply_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_nan_to_num" '(t, u) =
     '[ '("copy", AttrOpt Bool), '("nan", AttrOpt Double),
        '("posinf", AttrOpt (Maybe Double)),
        '("neginf", AttrOpt (Maybe Double)), '("data", AttrOpt (t u))]

__npi_nan_to_num ::
                 forall a t u .
                   (Tensor t, Fullfilled "__npi_nan_to_num" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__npi_nan_to_num" '(t, u) a -> TensorApply (t u)
__npi_nan_to_num args
  = let scalarArgs
          = catMaybes
              [("copy",) . showValue <$> (args !? #copy :: Maybe Bool),
               ("nan",) . showValue <$> (args !? #nan :: Maybe Double),
               ("posinf",) . showValue <$>
                 (args !? #posinf :: Maybe (Maybe Double)),
               ("neginf",) . showValue <$>
                 (args !? #neginf :: Maybe (Maybe Double))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_nan_to_num" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_negative" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_negative ::
               forall a t u .
                 (Tensor t, Fullfilled "__npi_negative" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__npi_negative" '(t, u) a -> TensorApply (t u)
__npi_negative args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_negative" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_norm" '(t, u) =
     '[ '("data", AttrOpt (t u))]

__npi_norm ::
           forall a t u .
             (Tensor t, Fullfilled "__npi_norm" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__npi_norm" '(t, u) a -> TensorApply (t u)
__npi_norm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_norm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_normal" '(t, u) =
     '[ '("loc", AttrReq (Maybe Float)),
        '("scale", AttrReq (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text),
        '("dtype", AttrOpt (EnumType '["float16", "float32", "float64"])),
        '("input1", AttrOpt (t u)), '("input2", AttrOpt (t u))]

__npi_normal ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_normal" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_normal" '(t, u) a -> TensorApply (t u)
__npi_normal args
  = let scalarArgs
          = catMaybes
              [("loc",) . showValue <$> (args !? #loc :: Maybe (Maybe Float)),
               ("scale",) . showValue <$> (args !? #scale :: Maybe (Maybe Float)),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int])),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["float16", "float32", "float64"]))]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> (args !? #input1 :: Maybe (t u)),
               ("input2",) . toRaw <$> (args !? #input2 :: Maybe (t u))]
      in
      applyRaw "_npi_normal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_normal_n" '(t, u) =
     '[ '("loc", AttrReq (Maybe Float)),
        '("scale", AttrReq (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text),
        '("dtype", AttrOpt (EnumType '["float16", "float32", "float64"])),
        '("input1", AttrOpt (t u)), '("input2", AttrOpt (t u))]

__npi_normal_n ::
               forall a t u .
                 (Tensor t, Fullfilled "__npi_normal_n" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__npi_normal_n" '(t, u) a -> TensorApply (t u)
__npi_normal_n args
  = let scalarArgs
          = catMaybes
              [("loc",) . showValue <$> (args !? #loc :: Maybe (Maybe Float)),
               ("scale",) . showValue <$> (args !? #scale :: Maybe (Maybe Float)),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int])),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["float16", "float32", "float64"]))]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> (args !? #input1 :: Maybe (t u)),
               ("input2",) . toRaw <$> (args !? #input2 :: Maybe (t u))]
      in
      applyRaw "_npi_normal_n" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_not_equal" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_not_equal ::
                forall a t u .
                  (Tensor t, Fullfilled "__npi_not_equal" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__npi_not_equal" '(t, u) a -> TensorApply (t Bool)
__npi_not_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_npi_not_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_not_equal_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_not_equal_scalar ::
                       forall a t u .
                         (Tensor t, Fullfilled "__npi_not_equal_scalar" '(t, u) a,
                          HasCallStack, DType u) =>
                         ArgsHMap "__npi_not_equal_scalar" '(t, u) a -> TensorApply (t Bool)
__npi_not_equal_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_not_equal_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_ones" '() =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                 "int64", "int8", "uint8"]))]

__npi_ones ::
           forall a t u .
             (Tensor t, Fullfilled "__npi_ones" '() a, HasCallStack, DType u) =>
             ArgsHMap "__npi_ones" '() a -> TensorApply (t u)
__npi_ones args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                           "int64", "int8", "uint8"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_ones" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_pareto" '(t, u) =
     '[ '("a", AttrOpt (Maybe Float)), '("size", AttrOpt (Maybe [Int])),
        '("ctx", AttrOpt Text), '("input1", AttrOpt (t u))]

__npi_pareto ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_pareto" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_pareto" '(t, u) a -> TensorApply (t u)
__npi_pareto args
  = let scalarArgs
          = catMaybes
              [("a",) . showValue <$> (args !? #a :: Maybe (Maybe Float)),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int])),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text)]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> (args !? #input1 :: Maybe (t u))]
      in
      applyRaw "_npi_pareto" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_percentile" '(t, u) =
     '[ '("axis", AttrOpt (Maybe [Int])),
        '("interpolation",
          AttrOpt
            (EnumType '["higher", "linear", "lower", "midpoint", "nearest"])),
        '("keepdims", AttrOpt Bool), '("q_scalar", AttrOpt (Maybe Double)),
        '("a", AttrOpt (t u)), '("q", AttrOpt (t u))]

__npi_percentile ::
                 forall a t u .
                   (Tensor t, Fullfilled "__npi_percentile" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__npi_percentile" '(t, u) a -> TensorApply (t u)
__npi_percentile args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("interpolation",) . showValue <$>
                 (args !? #interpolation ::
                    Maybe
                      (EnumType '["higher", "linear", "lower", "midpoint", "nearest"])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("q_scalar",) . showValue <$>
                 (args !? #q_scalar :: Maybe (Maybe Double))]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> (args !? #a :: Maybe (t u)),
               ("q",) . toRaw <$> (args !? #q :: Maybe (t u))]
      in
      applyRaw "_npi_percentile" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_pinv" '(t, u) =
     '[ '("hermitian", AttrOpt Bool), '("a", AttrOpt (t u)),
        '("rcond", AttrOpt (t u))]

__npi_pinv ::
           forall a t u .
             (Tensor t, Fullfilled "__npi_pinv" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__npi_pinv" '(t, u) a -> TensorApply (t u)
__npi_pinv args
  = let scalarArgs
          = catMaybes
              [("hermitian",) . showValue <$> (args !? #hermitian :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> (args !? #a :: Maybe (t u)),
               ("rcond",) . toRaw <$> (args !? #rcond :: Maybe (t u))]
      in
      applyRaw "_npi_pinv" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_pinv_scalar_rcond" '(t, u) =
     '[ '("rcond", AttrOpt Double), '("hermitian", AttrOpt Bool),
        '("a", AttrOpt (t u))]

__npi_pinv_scalar_rcond ::
                        forall a t u .
                          (Tensor t, Fullfilled "__npi_pinv_scalar_rcond" '(t, u) a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__npi_pinv_scalar_rcond" '(t, u) a -> TensorApply (t u)
__npi_pinv_scalar_rcond args
  = let scalarArgs
          = catMaybes
              [("rcond",) . showValue <$> (args !? #rcond :: Maybe Double),
               ("hermitian",) . showValue <$> (args !? #hermitian :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_npi_pinv_scalar_rcond" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_polyval" '(t, u) =
     '[ '("p", AttrOpt (t u)), '("x", AttrOpt (t u))]

__npi_polyval ::
              forall a t u .
                (Tensor t, Fullfilled "__npi_polyval" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__npi_polyval" '(t, u) a -> TensorApply (t u)
__npi_polyval args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("p",) . toRaw <$> (args !? #p :: Maybe (t u)),
               ("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_polyval" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_power" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_power ::
            forall a t u .
              (Tensor t, Fullfilled "__npi_power" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "__npi_power" '(t, u) a -> TensorApply (t u)
__npi_power args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_npi_power" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_power_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_power_scalar ::
                   forall a t u .
                     (Tensor t, Fullfilled "__npi_power_scalar" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__npi_power_scalar" '(t, u) a -> TensorApply (t u)
__npi_power_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_power_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_powerd" '(t, u) =
     '[ '("a", AttrOpt (Maybe Float)), '("size", AttrOpt (Maybe [Int])),
        '("input1", AttrOpt (t u))]

__npi_powerd ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_powerd" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_powerd" '(t, u) a -> TensorApply (t u)
__npi_powerd args
  = let scalarArgs
          = catMaybes
              [("a",) . showValue <$> (args !? #a :: Maybe (Maybe Float)),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int]))]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> (args !? #input1 :: Maybe (t u))]
      in
      applyRaw "_npi_powerd" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_radians" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_radians ::
              forall a t u .
                (Tensor t, Fullfilled "__npi_radians" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__npi_radians" '(t, u) a -> TensorApply (t u)
__npi_radians args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_radians" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_rarctan2_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_rarctan2_scalar ::
                      forall a t u .
                        (Tensor t, Fullfilled "__npi_rarctan2_scalar" '(t, u) a,
                         HasCallStack, DType u) =>
                        ArgsHMap "__npi_rarctan2_scalar" '(t, u) a -> TensorApply (t u)
__npi_rarctan2_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_rarctan2_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_rayleigh" '(t, u) =
     '[ '("scale", AttrOpt (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text),
        '("input1", AttrOpt (t u))]

__npi_rayleigh ::
               forall a t u .
                 (Tensor t, Fullfilled "__npi_rayleigh" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__npi_rayleigh" '(t, u) a -> TensorApply (t u)
__npi_rayleigh args
  = let scalarArgs
          = catMaybes
              [("scale",) . showValue <$>
                 (args !? #scale :: Maybe (Maybe Float)),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int])),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text)]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> (args !? #input1 :: Maybe (t u))]
      in
      applyRaw "_npi_rayleigh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_rcopysign_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_rcopysign_scalar ::
                       forall a t u .
                         (Tensor t, Fullfilled "__npi_rcopysign_scalar" '(t, u) a,
                          HasCallStack, DType u) =>
                         ArgsHMap "__npi_rcopysign_scalar" '(t, u) a -> TensorApply (t u)
__npi_rcopysign_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_rcopysign_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_reciprocal" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_reciprocal ::
                 forall a t u .
                   (Tensor t, Fullfilled "__npi_reciprocal" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__npi_reciprocal" '(t, u) a -> TensorApply (t u)
__npi_reciprocal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_reciprocal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_rint" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_rint ::
           forall a t u .
             (Tensor t, Fullfilled "__npi_rint" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__npi_rint" '(t, u) a -> TensorApply (t u)
__npi_rint args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_rint" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_rldexp_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_rldexp_scalar ::
                    forall a t u .
                      (Tensor t, Fullfilled "__npi_rldexp_scalar" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "__npi_rldexp_scalar" '(t, u) a -> TensorApply (t u)
__npi_rldexp_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_rldexp_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_rmod_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_rmod_scalar ::
                  forall a t u .
                    (Tensor t, Fullfilled "__npi_rmod_scalar" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__npi_rmod_scalar" '(t, u) a -> TensorApply (t u)
__npi_rmod_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_rmod_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_rot90" '(t, u) =
     '[ '("k", AttrOpt Int), '("axes", AttrOpt (Maybe [Int])),
        '("data", AttrOpt (t u))]

__npi_rot90 ::
            forall a t u .
              (Tensor t, Fullfilled "__npi_rot90" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "__npi_rot90" '(t, u) a -> TensorApply (t u)
__npi_rot90 args
  = let scalarArgs
          = catMaybes
              [("k",) . showValue <$> (args !? #k :: Maybe Int),
               ("axes",) . showValue <$> (args !? #axes :: Maybe (Maybe [Int]))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_rot90" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_rpower_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_rpower_scalar ::
                    forall a t u .
                      (Tensor t, Fullfilled "__npi_rpower_scalar" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "__npi_rpower_scalar" '(t, u) a -> TensorApply (t u)
__npi_rpower_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_rpower_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_rsubtract_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_rsubtract_scalar ::
                       forall a t u .
                         (Tensor t, Fullfilled "__npi_rsubtract_scalar" '(t, u) a,
                          HasCallStack, DType u) =>
                         ArgsHMap "__npi_rsubtract_scalar" '(t, u) a -> TensorApply (t u)
__npi_rsubtract_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_rsubtract_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_rtrue_divide_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_rtrue_divide_scalar ::
                          forall a t u .
                            (Tensor t, Fullfilled "__npi_rtrue_divide_scalar" '(t, u) a,
                             HasCallStack, DType u) =>
                            ArgsHMap "__npi_rtrue_divide_scalar" '(t, u) a -> TensorApply (t u)
__npi_rtrue_divide_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_rtrue_divide_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_share_memory" '(t, u) =
     '[ '("a", AttrOpt (t u)), '("b", AttrOpt (t u))]

__npi_share_memory ::
                   forall a t u .
                     (Tensor t, Fullfilled "__npi_share_memory" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__npi_share_memory" '(t, u) a -> TensorApply (t u)
__npi_share_memory args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> (args !? #a :: Maybe (t u)),
               ("b",) . toRaw <$> (args !? #b :: Maybe (t u))]
      in
      applyRaw "_npi_share_memory" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_sign" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_sign ::
           forall a t u .
             (Tensor t, Fullfilled "__npi_sign" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__npi_sign" '(t, u) a -> TensorApply (t u)
__npi_sign args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_sign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_sin" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_sin ::
          forall a t u .
            (Tensor t, Fullfilled "__npi_sin" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__npi_sin" '(t, u) a -> TensorApply (t u)
__npi_sin args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_sin" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_sinh" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_sinh ::
           forall a t u .
             (Tensor t, Fullfilled "__npi_sinh" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__npi_sinh" '(t, u) a -> TensorApply (t u)
__npi_sinh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_sinh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_solve" '(t, u) =
     '[ '("a", AttrOpt (t u)), '("b", AttrOpt (t u))]

__npi_solve ::
            forall a t u .
              (Tensor t, Fullfilled "__npi_solve" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "__npi_solve" '(t, u) a -> TensorApply (t u)
__npi_solve args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> (args !? #a :: Maybe (t u)),
               ("b",) . toRaw <$> (args !? #b :: Maybe (t u))]
      in
      applyRaw "_npi_solve" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_sqrt" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_sqrt ::
           forall a t u .
             (Tensor t, Fullfilled "__npi_sqrt" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__npi_sqrt" '(t, u) a -> TensorApply (t u)
__npi_sqrt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_sqrt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_square" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_square ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_square" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_square" '(t, u) a -> TensorApply (t u)
__npi_square args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_square" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_stack" '(t, u) =
     '[ '("axis", AttrOpt Int), '("num_args", AttrReq Int),
        '("data", AttrOpt [t u])]

__npi_stack ::
            forall a t u .
              (Tensor t, Fullfilled "__npi_stack" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "__npi_stack" '(t, u) a -> TensorApply (t u)
__npi_stack args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("num_args",) . showValue <$> (args !? #num_args :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "_npi_stack" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__npi_std" '(t, u) =
     '[ '("axis", AttrOpt (Maybe [Int])),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["float16", "float32", "float64", "int32", "int64", "int8"]))),
        '("ddof", AttrOpt Int), '("keepdims", AttrOpt Bool),
        '("a", AttrOpt (t u))]

__npi_std ::
          forall a t u .
            (Tensor t, Fullfilled "__npi_std" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__npi_std" '(t, u) a -> TensorApply (t u)
__npi_std args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (Maybe
                         (EnumType
                            '["float16", "float32", "float64", "int32", "int64", "int8"]))),
               ("ddof",) . showValue <$> (args !? #ddof :: Maybe Int),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_npi_std" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_subtract" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_subtract ::
               forall a t u .
                 (Tensor t, Fullfilled "__npi_subtract" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__npi_subtract" '(t, u) a -> TensorApply (t u)
__npi_subtract args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_npi_subtract" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_subtract_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_subtract_scalar ::
                      forall a t u .
                        (Tensor t, Fullfilled "__npi_subtract_scalar" '(t, u) a,
                         HasCallStack, DType u) =>
                        ArgsHMap "__npi_subtract_scalar" '(t, u) a -> TensorApply (t u)
__npi_subtract_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_subtract_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_svd" '(t, u) =
     '[ '("a", AttrOpt (t u))]

__npi_svd ::
          forall a t u .
            (Tensor t, Fullfilled "__npi_svd" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__npi_svd" '(t, u) a -> TensorApply (t u)
__npi_svd args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_npi_svd" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_tan" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_tan ::
          forall a t u .
            (Tensor t, Fullfilled "__npi_tan" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__npi_tan" '(t, u) a -> TensorApply (t u)
__npi_tan args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_tan" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_tanh" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_tanh ::
           forall a t u .
             (Tensor t, Fullfilled "__npi_tanh" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__npi_tanh" '(t, u) a -> TensorApply (t u)
__npi_tanh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_tanh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_tensordot" '(t, u) =
     '[ '("a_axes_summed", AttrReq [Int]),
        '("b_axes_summed", AttrReq [Int]), '("a", AttrOpt (t u)),
        '("b", AttrOpt (t u))]

__npi_tensordot ::
                forall a t u .
                  (Tensor t, Fullfilled "__npi_tensordot" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__npi_tensordot" '(t, u) a -> TensorApply (t u)
__npi_tensordot args
  = let scalarArgs
          = catMaybes
              [("a_axes_summed",) . showValue <$>
                 (args !? #a_axes_summed :: Maybe [Int]),
               ("b_axes_summed",) . showValue <$>
                 (args !? #b_axes_summed :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> (args !? #a :: Maybe (t u)),
               ("b",) . toRaw <$> (args !? #b :: Maybe (t u))]
      in
      applyRaw "_npi_tensordot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_tensordot_int_axes" '(t, u) =
     '[ '("axes", AttrReq Int), '("a", AttrOpt (t u)),
        '("b", AttrOpt (t u))]

__npi_tensordot_int_axes ::
                         forall a t u .
                           (Tensor t, Fullfilled "__npi_tensordot_int_axes" '(t, u) a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__npi_tensordot_int_axes" '(t, u) a -> TensorApply (t u)
__npi_tensordot_int_axes args
  = let scalarArgs
          = catMaybes
              [("axes",) . showValue <$> (args !? #axes :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> (args !? #a :: Maybe (t u)),
               ("b",) . toRaw <$> (args !? #b :: Maybe (t u))]
      in
      applyRaw "_npi_tensordot_int_axes" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_tensorinv" '(t, u) =
     '[ '("ind", AttrOpt Int), '("a", AttrOpt (t u))]

__npi_tensorinv ::
                forall a t u .
                  (Tensor t, Fullfilled "__npi_tensorinv" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__npi_tensorinv" '(t, u) a -> TensorApply (t u)
__npi_tensorinv args
  = let scalarArgs
          = catMaybes [("ind",) . showValue <$> (args !? #ind :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_npi_tensorinv" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_tensorsolve" '(t, u) =
     '[ '("a_axes", AttrOpt [Int]), '("a", AttrOpt (t u)),
        '("b", AttrOpt (t u))]

__npi_tensorsolve ::
                  forall a t u .
                    (Tensor t, Fullfilled "__npi_tensorsolve" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__npi_tensorsolve" '(t, u) a -> TensorApply (t u)
__npi_tensorsolve args
  = let scalarArgs
          = catMaybes
              [("a_axes",) . showValue <$> (args !? #a_axes :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> (args !? #a :: Maybe (t u)),
               ("b",) . toRaw <$> (args !? #b :: Maybe (t u))]
      in
      applyRaw "_npi_tensorsolve" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_tril" '(t, u) =
     '[ '("k", AttrOpt Int), '("data", AttrOpt (t u))]

__npi_tril ::
           forall a t u .
             (Tensor t, Fullfilled "__npi_tril" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__npi_tril" '(t, u) a -> TensorApply (t u)
__npi_tril args
  = let scalarArgs
          = catMaybes [("k",) . showValue <$> (args !? #k :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_tril" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_true_divide" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__npi_true_divide ::
                  forall a t u .
                    (Tensor t, Fullfilled "__npi_true_divide" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "__npi_true_divide" '(t, u) a -> TensorApply (t u)
__npi_true_divide args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_npi_true_divide" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_true_divide_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__npi_true_divide_scalar ::
                         forall a t u .
                           (Tensor t, Fullfilled "__npi_true_divide_scalar" '(t, u) a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__npi_true_divide_scalar" '(t, u) a -> TensorApply (t u)
__npi_true_divide_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_true_divide_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_trunc" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npi_trunc ::
            forall a t u .
              (Tensor t, Fullfilled "__npi_trunc" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "__npi_trunc" '(t, u) a -> TensorApply (t u)
__npi_trunc args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_trunc" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_uniform" '(t, u) =
     '[ '("low", AttrReq (Maybe Float)),
        '("high", AttrReq (Maybe Float)), '("size", AttrOpt (Maybe [Int])),
        '("ctx", AttrOpt Text),
        '("dtype", AttrOpt (EnumType '["float16", "float32", "float64"])),
        '("input1", AttrOpt (t u)), '("input2", AttrOpt (t u))]

__npi_uniform ::
              forall a t u .
                (Tensor t, Fullfilled "__npi_uniform" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__npi_uniform" '(t, u) a -> TensorApply (t u)
__npi_uniform args
  = let scalarArgs
          = catMaybes
              [("low",) . showValue <$> (args !? #low :: Maybe (Maybe Float)),
               ("high",) . showValue <$> (args !? #high :: Maybe (Maybe Float)),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int])),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["float16", "float32", "float64"]))]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> (args !? #input1 :: Maybe (t u)),
               ("input2",) . toRaw <$> (args !? #input2 :: Maybe (t u))]
      in
      applyRaw "_npi_uniform" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_uniform_n" '(t, u) =
     '[ '("low", AttrReq (Maybe Float)),
        '("high", AttrReq (Maybe Float)), '("size", AttrOpt (Maybe [Int])),
        '("ctx", AttrOpt Text),
        '("dtype", AttrOpt (EnumType '["float16", "float32", "float64"])),
        '("input1", AttrOpt (t u)), '("input2", AttrOpt (t u))]

__npi_uniform_n ::
                forall a t u .
                  (Tensor t, Fullfilled "__npi_uniform_n" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__npi_uniform_n" '(t, u) a -> TensorApply (t u)
__npi_uniform_n args
  = let scalarArgs
          = catMaybes
              [("low",) . showValue <$> (args !? #low :: Maybe (Maybe Float)),
               ("high",) . showValue <$> (args !? #high :: Maybe (Maybe Float)),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int])),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["float16", "float32", "float64"]))]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> (args !? #input1 :: Maybe (t u)),
               ("input2",) . toRaw <$> (args !? #input2 :: Maybe (t u))]
      in
      applyRaw "_npi_uniform_n" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_unique" '(t, u) =
     '[ '("return_index", AttrOpt Bool),
        '("return_inverse", AttrOpt Bool),
        '("return_counts", AttrOpt Bool), '("axis", AttrOpt (Maybe Int)),
        '("data", AttrOpt (t u))]

__npi_unique ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_unique" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_unique" '(t, u) a -> TensorApply (t u)
__npi_unique args
  = let scalarArgs
          = catMaybes
              [("return_index",) . showValue <$>
                 (args !? #return_index :: Maybe Bool),
               ("return_inverse",) . showValue <$>
                 (args !? #return_inverse :: Maybe Bool),
               ("return_counts",) . showValue <$>
                 (args !? #return_counts :: Maybe Bool),
               ("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npi_unique" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_var" '(t, u) =
     '[ '("axis", AttrOpt (Maybe [Int])),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["float16", "float32", "float64", "int32", "int64", "int8"]))),
        '("ddof", AttrOpt Int), '("keepdims", AttrOpt Bool),
        '("a", AttrOpt (t u))]

__npi_var ::
          forall a t u .
            (Tensor t, Fullfilled "__npi_var" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__npi_var" '(t, u) a -> TensorApply (t u)
__npi_var args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (Maybe
                         (EnumType
                            '["float16", "float32", "float64", "int32", "int64", "int8"]))),
               ("ddof",) . showValue <$> (args !? #ddof :: Maybe Int),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_npi_var" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_vstack" '(t, u) =
     '[ '("num_args", AttrReq Int), '("data", AttrOpt [t u])]

__npi_vstack ::
             forall a t u .
               (Tensor t, Fullfilled "__npi_vstack" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__npi_vstack" '(t, u) a -> TensorApply (t u)
__npi_vstack args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "_npi_vstack" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__npi_weibull" '(t, u) =
     '[ '("a", AttrOpt (Maybe Float)), '("size", AttrOpt (Maybe [Int])),
        '("ctx", AttrOpt Text), '("input1", AttrOpt (t u))]

__npi_weibull ::
              forall a t u .
                (Tensor t, Fullfilled "__npi_weibull" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__npi_weibull" '(t, u) a -> TensorApply (t u)
__npi_weibull args
  = let scalarArgs
          = catMaybes
              [("a",) . showValue <$> (args !? #a :: Maybe (Maybe Float)),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int])),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text)]
        tensorKeyArgs
          = catMaybes
              [("input1",) . toRaw <$> (args !? #input1 :: Maybe (t u))]
      in
      applyRaw "_npi_weibull" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_where" '(t, u) =
     '[ '("condition", AttrOpt (t Bool)), '("x", AttrOpt (t u)),
        '("y", AttrOpt (t u))]

__npi_where ::
            forall a t u .
              (Tensor t, Fullfilled "__npi_where" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "__npi_where" '(t, u) a -> TensorApply (t u)
__npi_where args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("condition",) . toRaw <$> (args !? #condition :: Maybe (t Bool)),
               ("x",) . toRaw <$> (args !? #x :: Maybe (t u)),
               ("y",) . toRaw <$> (args !? #y :: Maybe (t u))]
      in
      applyRaw "_npi_where" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_where_lscalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("condition", AttrOpt (t u)),
        '("x", AttrOpt (t u))]

__npi_where_lscalar ::
                    forall a t u .
                      (Tensor t, Fullfilled "__npi_where_lscalar" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "__npi_where_lscalar" '(t, u) a -> TensorApply (t u)
__npi_where_lscalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double)]
        tensorKeyArgs
          = catMaybes
              [("condition",) . toRaw <$> (args !? #condition :: Maybe (t u)),
               ("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npi_where_lscalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_where_rscalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("condition", AttrOpt (t u)),
        '("y", AttrOpt (t u))]

__npi_where_rscalar ::
                    forall a t u .
                      (Tensor t, Fullfilled "__npi_where_rscalar" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "__npi_where_rscalar" '(t, u) a -> TensorApply (t u)
__npi_where_rscalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double)]
        tensorKeyArgs
          = catMaybes
              [("condition",) . toRaw <$> (args !? #condition :: Maybe (t u)),
               ("y",) . toRaw <$> (args !? #y :: Maybe (t u))]
      in
      applyRaw "_npi_where_rscalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_where_scalar2" '(t, u) =
     '[ '("x", AttrOpt Double), '("y", AttrOpt Double),
        '("condition", AttrOpt (t u))]

__npi_where_scalar2 ::
                    forall a t u .
                      (Tensor t, Fullfilled "__npi_where_scalar2" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "__npi_where_scalar2" '(t, u) a -> TensorApply (t u)
__npi_where_scalar2 args
  = let scalarArgs
          = catMaybes
              [("x",) . showValue <$> (args !? #x :: Maybe Double),
               ("y",) . showValue <$> (args !? #y :: Maybe Double)]
        tensorKeyArgs
          = catMaybes
              [("condition",) . toRaw <$> (args !? #condition :: Maybe (t u))]
      in
      applyRaw "_npi_where_scalar2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npi_zeros" '() =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                 "int64", "int8", "uint8"]))]

__npi_zeros ::
            forall a t u .
              (Tensor t, Fullfilled "__npi_zeros" '() a, HasCallStack,
               DType u) =>
              ArgsHMap "__npi_zeros" '() a -> TensorApply (t u)
__npi_zeros args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                           "int64", "int8", "uint8"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_npi_zeros" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npx_constraint_check" '(t, u) =
     '[ '("msg", AttrOpt Text), '("input", AttrOpt (t u))]

__npx_constraint_check ::
                       forall a t u .
                         (Tensor t, Fullfilled "__npx_constraint_check" '(t, u) a,
                          HasCallStack, DType u) =>
                         ArgsHMap "__npx_constraint_check" '(t, u) a -> TensorApply (t u)
__npx_constraint_check args
  = let scalarArgs
          = catMaybes [("msg",) . showValue <$> (args !? #msg :: Maybe Text)]
        tensorKeyArgs
          = catMaybes
              [("input",) . toRaw <$> (args !? #input :: Maybe (t u))]
      in
      applyRaw "_npx_constraint_check" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npx_nonzero" '(t, u) =
     '[ '("x", AttrOpt (t u))]

__npx_nonzero ::
              forall a t u .
                (Tensor t, Fullfilled "__npx_nonzero" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__npx_nonzero" '(t, u) a -> TensorApply (t u)
__npx_nonzero args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("x",) . toRaw <$> (args !? #x :: Maybe (t u))]
      in
      applyRaw "_npx_nonzero" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npx_relu" '(t, u) =
     '[ '("data", AttrOpt (t u))]

__npx_relu ::
           forall a t u .
             (Tensor t, Fullfilled "__npx_relu" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__npx_relu" '(t, u) a -> TensorApply (t u)
__npx_relu args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npx_relu" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npx_reshape" '(t, u) =
     '[ '("newshape", AttrReq [Int]), '("reverse", AttrOpt Bool),
        '("order", AttrOpt Text), '("a", AttrOpt (t u))]

__npx_reshape ::
              forall a t u .
                (Tensor t, Fullfilled "__npx_reshape" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__npx_reshape" '(t, u) a -> TensorApply (t u)
__npx_reshape args
  = let scalarArgs
          = catMaybes
              [("newshape",) . showValue <$> (args !? #newshape :: Maybe [Int]),
               ("reverse",) . showValue <$> (args !? #reverse :: Maybe Bool),
               ("order",) . showValue <$> (args !? #order :: Maybe Text)]
        tensorKeyArgs
          = catMaybes [("a",) . toRaw <$> (args !? #a :: Maybe (t u))]
      in
      applyRaw "_npx_reshape" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__npx_sigmoid" '(t, u) =
     '[ '("data", AttrOpt (t u))]

__npx_sigmoid ::
              forall a t u .
                (Tensor t, Fullfilled "__npx_sigmoid" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__npx_sigmoid" '(t, u) a -> TensorApply (t u)
__npx_sigmoid args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_npx_sigmoid" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__onehot_encode" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__onehot_encode ::
                forall a t u .
                  (Tensor t, Fullfilled "__onehot_encode" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__onehot_encode" '(t, u) a -> TensorApply (t u)
__onehot_encode args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_onehot_encode" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__ones" '() =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                 "int64", "int8", "uint8"]))]

__ones ::
       forall a t u .
         (Tensor t, Fullfilled "__ones" '() a, HasCallStack, DType u) =>
         ArgsHMap "__ones" '() a -> TensorApply (t u)
__ones args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                           "int64", "int8", "uint8"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_ones" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__plus_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__plus_scalar ::
              forall a t u .
                (Tensor t, Fullfilled "__plus_scalar" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__plus_scalar" '(t, u) a -> TensorApply (t u)
__plus_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_plus_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__power" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__power ::
        forall a t u .
          (Tensor t, Fullfilled "__power" '(t, u) a, HasCallStack,
           DType u) =>
          ArgsHMap "__power" '(t, u) a -> TensorApply (t u)
__power args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_power" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__power_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__power_scalar ::
               forall a t u .
                 (Tensor t, Fullfilled "__power_scalar" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__power_scalar" '(t, u) a -> TensorApply (t u)
__power_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_power_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__random_exponential" '() =
     '[ '("lam", AttrOpt Float), '("shape", AttrOpt [Int]),
        '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

__random_exponential ::
                     forall a t u .
                       (Tensor t, Fullfilled "__random_exponential" '() a, HasCallStack,
                        DType u) =>
                       ArgsHMap "__random_exponential" '() a -> TensorApply (t u)
__random_exponential args
  = let scalarArgs
          = catMaybes
              [("lam",) . showValue <$> (args !? #lam :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_random_exponential" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__random_exponential_like" '(t, u) =
     '[ '("lam", AttrOpt Float), '("data", AttrOpt (t u))]

__random_exponential_like ::
                          forall a t u .
                            (Tensor t, Fullfilled "__random_exponential_like" '(t, u) a,
                             HasCallStack, DType u) =>
                            ArgsHMap "__random_exponential_like" '(t, u) a -> TensorApply (t u)
__random_exponential_like args
  = let scalarArgs
          = catMaybes
              [("lam",) . showValue <$> (args !? #lam :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_random_exponential_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__random_gamma" '() =
     '[ '("alpha", AttrOpt Float), '("beta", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

__random_gamma ::
               forall a t u .
                 (Tensor t, Fullfilled "__random_gamma" '() a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__random_gamma" '() a -> TensorApply (t u)
__random_gamma args
  = let scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> (args !? #alpha :: Maybe Float),
               ("beta",) . showValue <$> (args !? #beta :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_random_gamma" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__random_gamma_like" '(t, u) =
     '[ '("alpha", AttrOpt Float), '("beta", AttrOpt Float),
        '("data", AttrOpt (t u))]

__random_gamma_like ::
                    forall a t u .
                      (Tensor t, Fullfilled "__random_gamma_like" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "__random_gamma_like" '(t, u) a -> TensorApply (t u)
__random_gamma_like args
  = let scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> (args !? #alpha :: Maybe Float),
               ("beta",) . showValue <$> (args !? #beta :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_random_gamma_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__random_generalized_negative_binomial" '() =
     '[ '("mu", AttrOpt Float), '("alpha", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

__random_generalized_negative_binomial ::
                                       forall a t u .
                                         (Tensor t,
                                          Fullfilled "__random_generalized_negative_binomial" '() a,
                                          HasCallStack, DType u) =>
                                         ArgsHMap "__random_generalized_negative_binomial" '() a ->
                                           TensorApply (t u)
__random_generalized_negative_binomial args
  = let scalarArgs
          = catMaybes
              [("mu",) . showValue <$> (args !? #mu :: Maybe Float),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_random_generalized_negative_binomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__random_generalized_negative_binomial_like" '(t, u)
     =
     '[ '("mu", AttrOpt Float), '("alpha", AttrOpt Float),
        '("data", AttrOpt (t u))]

__random_generalized_negative_binomial_like ::
                                            forall a t u .
                                              (Tensor t,
                                               Fullfilled
                                                 "__random_generalized_negative_binomial_like"
                                                 '(t, u)
                                                 a,
                                               HasCallStack, DType u) =>
                                              ArgsHMap "__random_generalized_negative_binomial_like"
                                                '(t, u)
                                                a
                                                -> TensorApply (t u)
__random_generalized_negative_binomial_like args
  = let scalarArgs
          = catMaybes
              [("mu",) . showValue <$> (args !? #mu :: Maybe Float),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_random_generalized_negative_binomial_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__random_negative_binomial" '() =
     '[ '("k", AttrOpt Int), '("p", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

__random_negative_binomial ::
                           forall a t u .
                             (Tensor t, Fullfilled "__random_negative_binomial" '() a,
                              HasCallStack, DType u) =>
                             ArgsHMap "__random_negative_binomial" '() a -> TensorApply (t u)
__random_negative_binomial args
  = let scalarArgs
          = catMaybes
              [("k",) . showValue <$> (args !? #k :: Maybe Int),
               ("p",) . showValue <$> (args !? #p :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_random_negative_binomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__random_negative_binomial_like" '(t, u) =
     '[ '("k", AttrOpt Int), '("p", AttrOpt Float),
        '("data", AttrOpt (t u))]

__random_negative_binomial_like ::
                                forall a t u .
                                  (Tensor t, Fullfilled "__random_negative_binomial_like" '(t, u) a,
                                   HasCallStack, DType u) =>
                                  ArgsHMap "__random_negative_binomial_like" '(t, u) a ->
                                    TensorApply (t u)
__random_negative_binomial_like args
  = let scalarArgs
          = catMaybes
              [("k",) . showValue <$> (args !? #k :: Maybe Int),
               ("p",) . showValue <$> (args !? #p :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_random_negative_binomial_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__random_normal" '() =
     '[ '("loc", AttrOpt Float), '("scale", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

__random_normal ::
                forall a t u .
                  (Tensor t, Fullfilled "__random_normal" '() a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__random_normal" '() a -> TensorApply (t u)
__random_normal args
  = let scalarArgs
          = catMaybes
              [("loc",) . showValue <$> (args !? #loc :: Maybe Float),
               ("scale",) . showValue <$> (args !? #scale :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_random_normal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__random_normal_like" '(t, u) =
     '[ '("loc", AttrOpt Float), '("scale", AttrOpt Float),
        '("data", AttrOpt (t u))]

__random_normal_like ::
                     forall a t u .
                       (Tensor t, Fullfilled "__random_normal_like" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "__random_normal_like" '(t, u) a -> TensorApply (t u)
__random_normal_like args
  = let scalarArgs
          = catMaybes
              [("loc",) . showValue <$> (args !? #loc :: Maybe Float),
               ("scale",) . showValue <$> (args !? #scale :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_random_normal_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__random_pdf_dirichlet" '(t, u) =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt (t u)),
        '("alpha", AttrOpt (t u))]

__random_pdf_dirichlet ::
                       forall a t u .
                         (Tensor t, Fullfilled "__random_pdf_dirichlet" '(t, u) a,
                          HasCallStack, DType u) =>
                         ArgsHMap "__random_pdf_dirichlet" '(t, u) a -> TensorApply (t u)
__random_pdf_dirichlet args
  = let scalarArgs
          = catMaybes
              [("is_log",) . showValue <$> (args !? #is_log :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("sample",) . toRaw <$> (args !? #sample :: Maybe (t u)),
               ("alpha",) . toRaw <$> (args !? #alpha :: Maybe (t u))]
      in
      applyRaw "_random_pdf_dirichlet" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__random_pdf_exponential" '(t, u) =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt (t u)),
        '("lam", AttrOpt (t u))]

__random_pdf_exponential ::
                         forall a t u .
                           (Tensor t, Fullfilled "__random_pdf_exponential" '(t, u) a,
                            HasCallStack, DType u) =>
                           ArgsHMap "__random_pdf_exponential" '(t, u) a -> TensorApply (t u)
__random_pdf_exponential args
  = let scalarArgs
          = catMaybes
              [("is_log",) . showValue <$> (args !? #is_log :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("sample",) . toRaw <$> (args !? #sample :: Maybe (t u)),
               ("lam",) . toRaw <$> (args !? #lam :: Maybe (t u))]
      in
      applyRaw "_random_pdf_exponential" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__random_pdf_gamma" '(t, u) =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt (t u)),
        '("alpha", AttrOpt (t u)), '("beta", AttrOpt (t u))]

__random_pdf_gamma ::
                   forall a t u .
                     (Tensor t, Fullfilled "__random_pdf_gamma" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__random_pdf_gamma" '(t, u) a -> TensorApply (t u)
__random_pdf_gamma args
  = let scalarArgs
          = catMaybes
              [("is_log",) . showValue <$> (args !? #is_log :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("sample",) . toRaw <$> (args !? #sample :: Maybe (t u)),
               ("alpha",) . toRaw <$> (args !? #alpha :: Maybe (t u)),
               ("beta",) . toRaw <$> (args !? #beta :: Maybe (t u))]
      in
      applyRaw "_random_pdf_gamma" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__random_pdf_generalized_negative_binomial" '(t, u)
     =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt (t u)),
        '("mu", AttrOpt (t u)), '("alpha", AttrOpt (t u))]

__random_pdf_generalized_negative_binomial ::
                                           forall a t u .
                                             (Tensor t,
                                              Fullfilled
                                                "__random_pdf_generalized_negative_binomial"
                                                '(t, u)
                                                a,
                                              HasCallStack, DType u) =>
                                             ArgsHMap "__random_pdf_generalized_negative_binomial"
                                               '(t, u)
                                               a
                                               -> TensorApply (t u)
__random_pdf_generalized_negative_binomial args
  = let scalarArgs
          = catMaybes
              [("is_log",) . showValue <$> (args !? #is_log :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("sample",) . toRaw <$> (args !? #sample :: Maybe (t u)),
               ("mu",) . toRaw <$> (args !? #mu :: Maybe (t u)),
               ("alpha",) . toRaw <$> (args !? #alpha :: Maybe (t u))]
      in
      applyRaw "_random_pdf_generalized_negative_binomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__random_pdf_negative_binomial" '(t, u) =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt (t u)),
        '("k", AttrOpt (t u)), '("p", AttrOpt (t u))]

__random_pdf_negative_binomial ::
                               forall a t u .
                                 (Tensor t, Fullfilled "__random_pdf_negative_binomial" '(t, u) a,
                                  HasCallStack, DType u) =>
                                 ArgsHMap "__random_pdf_negative_binomial" '(t, u) a ->
                                   TensorApply (t u)
__random_pdf_negative_binomial args
  = let scalarArgs
          = catMaybes
              [("is_log",) . showValue <$> (args !? #is_log :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("sample",) . toRaw <$> (args !? #sample :: Maybe (t u)),
               ("k",) . toRaw <$> (args !? #k :: Maybe (t u)),
               ("p",) . toRaw <$> (args !? #p :: Maybe (t u))]
      in
      applyRaw "_random_pdf_negative_binomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__random_pdf_normal" '(t, u) =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt (t u)),
        '("mu", AttrOpt (t u)), '("sigma", AttrOpt (t u))]

__random_pdf_normal ::
                    forall a t u .
                      (Tensor t, Fullfilled "__random_pdf_normal" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "__random_pdf_normal" '(t, u) a -> TensorApply (t u)
__random_pdf_normal args
  = let scalarArgs
          = catMaybes
              [("is_log",) . showValue <$> (args !? #is_log :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("sample",) . toRaw <$> (args !? #sample :: Maybe (t u)),
               ("mu",) . toRaw <$> (args !? #mu :: Maybe (t u)),
               ("sigma",) . toRaw <$> (args !? #sigma :: Maybe (t u))]
      in
      applyRaw "_random_pdf_normal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__random_pdf_poisson" '(t, u) =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt (t u)),
        '("lam", AttrOpt (t u))]

__random_pdf_poisson ::
                     forall a t u .
                       (Tensor t, Fullfilled "__random_pdf_poisson" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "__random_pdf_poisson" '(t, u) a -> TensorApply (t u)
__random_pdf_poisson args
  = let scalarArgs
          = catMaybes
              [("is_log",) . showValue <$> (args !? #is_log :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("sample",) . toRaw <$> (args !? #sample :: Maybe (t u)),
               ("lam",) . toRaw <$> (args !? #lam :: Maybe (t u))]
      in
      applyRaw "_random_pdf_poisson" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__random_pdf_uniform" '(t, u) =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt (t u)),
        '("low", AttrOpt (t u)), '("high", AttrOpt (t u))]

__random_pdf_uniform ::
                     forall a t u .
                       (Tensor t, Fullfilled "__random_pdf_uniform" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "__random_pdf_uniform" '(t, u) a -> TensorApply (t u)
__random_pdf_uniform args
  = let scalarArgs
          = catMaybes
              [("is_log",) . showValue <$> (args !? #is_log :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("sample",) . toRaw <$> (args !? #sample :: Maybe (t u)),
               ("low",) . toRaw <$> (args !? #low :: Maybe (t u)),
               ("high",) . toRaw <$> (args !? #high :: Maybe (t u))]
      in
      applyRaw "_random_pdf_uniform" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__random_poisson" '() =
     '[ '("lam", AttrOpt Float), '("shape", AttrOpt [Int]),
        '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

__random_poisson ::
                 forall a t u .
                   (Tensor t, Fullfilled "__random_poisson" '() a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__random_poisson" '() a -> TensorApply (t u)
__random_poisson args
  = let scalarArgs
          = catMaybes
              [("lam",) . showValue <$> (args !? #lam :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_random_poisson" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__random_poisson_like" '(t, u) =
     '[ '("lam", AttrOpt Float), '("data", AttrOpt (t u))]

__random_poisson_like ::
                      forall a t u .
                        (Tensor t, Fullfilled "__random_poisson_like" '(t, u) a,
                         HasCallStack, DType u) =>
                        ArgsHMap "__random_poisson_like" '(t, u) a -> TensorApply (t u)
__random_poisson_like args
  = let scalarArgs
          = catMaybes
              [("lam",) . showValue <$> (args !? #lam :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_random_poisson_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__random_uniform" '() =
     '[ '("low", AttrOpt Float), '("high", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

__random_uniform ::
                 forall a t u .
                   (Tensor t, Fullfilled "__random_uniform" '() a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__random_uniform" '() a -> TensorApply (t u)
__random_uniform args
  = let scalarArgs
          = catMaybes
              [("low",) . showValue <$> (args !? #low :: Maybe Float),
               ("high",) . showValue <$> (args !? #high :: Maybe Float),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_random_uniform" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__random_uniform_like" '(t, u) =
     '[ '("low", AttrOpt Float), '("high", AttrOpt Float),
        '("data", AttrOpt (t u))]

__random_uniform_like ::
                      forall a t u .
                        (Tensor t, Fullfilled "__random_uniform_like" '(t, u) a,
                         HasCallStack, DType u) =>
                        ArgsHMap "__random_uniform_like" '(t, u) a -> TensorApply (t u)
__random_uniform_like args
  = let scalarArgs
          = catMaybes
              [("low",) . showValue <$> (args !? #low :: Maybe Float),
               ("high",) . showValue <$> (args !? #high :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_random_uniform_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__ravel_multi_index" '(t, u) =
     '[ '("shape", AttrOpt [Int]), '("data", AttrOpt (t u))]

__ravel_multi_index ::
                    forall a t u .
                      (Tensor t, Fullfilled "__ravel_multi_index" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "__ravel_multi_index" '(t, u) a -> TensorApply (t u)
__ravel_multi_index args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_ravel_multi_index" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__rdiv_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__rdiv_scalar ::
              forall a t u .
                (Tensor t, Fullfilled "__rdiv_scalar" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__rdiv_scalar" '(t, u) a -> TensorApply (t u)
__rdiv_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_rdiv_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__rminus_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__rminus_scalar ::
                forall a t u .
                  (Tensor t, Fullfilled "__rminus_scalar" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__rminus_scalar" '(t, u) a -> TensorApply (t u)
__rminus_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_rminus_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__rmod_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__rmod_scalar ::
              forall a t u .
                (Tensor t, Fullfilled "__rmod_scalar" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "__rmod_scalar" '(t, u) a -> TensorApply (t u)
__rmod_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_rmod_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__rnn_param_concat" '(t, u) =
     '[ '("num_args", AttrReq Int), '("dim", AttrOpt Int),
        '("data", AttrOpt [t u])]

__rnn_param_concat ::
                   forall a t u .
                     (Tensor t, Fullfilled "__rnn_param_concat" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "__rnn_param_concat" '(t, u) a -> TensorApply (t u)
__rnn_param_concat args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("dim",) . showValue <$> (args !? #dim :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "_rnn_param_concat" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__rpower_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__rpower_scalar ::
                forall a t u .
                  (Tensor t, Fullfilled "__rpower_scalar" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__rpower_scalar" '(t, u) a -> TensorApply (t u)
__rpower_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_rpower_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__sample_exponential" '(t, u) =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("lam", AttrOpt (t u))]

__sample_exponential ::
                     forall a t u .
                       (Tensor t, Fullfilled "__sample_exponential" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "__sample_exponential" '(t, u) a -> TensorApply (t u)
__sample_exponential args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorKeyArgs
          = catMaybes [("lam",) . toRaw <$> (args !? #lam :: Maybe (t u))]
      in
      applyRaw "_sample_exponential" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__sample_gamma" '(t, u) =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("alpha", AttrOpt (t u)), '("beta", AttrOpt (t u))]

__sample_gamma ::
               forall a t u .
                 (Tensor t, Fullfilled "__sample_gamma" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__sample_gamma" '(t, u) a -> TensorApply (t u)
__sample_gamma args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorKeyArgs
          = catMaybes
              [("alpha",) . toRaw <$> (args !? #alpha :: Maybe (t u)),
               ("beta",) . toRaw <$> (args !? #beta :: Maybe (t u))]
      in
      applyRaw "_sample_gamma" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "__sample_generalized_negative_binomial" '(t, u) =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("mu", AttrOpt (t u)), '("alpha", AttrOpt (t u))]

__sample_generalized_negative_binomial ::
                                       forall a t u .
                                         (Tensor t,
                                          Fullfilled "__sample_generalized_negative_binomial"
                                            '(t, u)
                                            a,
                                          HasCallStack, DType u) =>
                                         ArgsHMap "__sample_generalized_negative_binomial" '(t, u) a
                                           -> TensorApply (t u)
__sample_generalized_negative_binomial args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorKeyArgs
          = catMaybes
              [("mu",) . toRaw <$> (args !? #mu :: Maybe (t u)),
               ("alpha",) . toRaw <$> (args !? #alpha :: Maybe (t u))]
      in
      applyRaw "_sample_generalized_negative_binomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__sample_multinomial" '(t, u) =
     '[ '("shape", AttrOpt [Int]), '("get_prob", AttrOpt Bool),
        '("dtype",
          AttrOpt
            (EnumType '["float16", "float32", "float64", "int32", "uint8"])),
        '("data", AttrOpt (t u))]

__sample_multinomial ::
                     forall a t u .
                       (Tensor t, Fullfilled "__sample_multinomial" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "__sample_multinomial" '(t, u) a -> TensorApply (t u)
__sample_multinomial args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("get_prob",) . showValue <$> (args !? #get_prob :: Maybe Bool),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType '["float16", "float32", "float64", "int32", "uint8"]))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_sample_multinomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__sample_negative_binomial" '(t, u) =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("k", AttrOpt (t u)), '("p", AttrOpt (t u))]

__sample_negative_binomial ::
                           forall a t u .
                             (Tensor t, Fullfilled "__sample_negative_binomial" '(t, u) a,
                              HasCallStack, DType u) =>
                             ArgsHMap "__sample_negative_binomial" '(t, u) a ->
                               TensorApply (t u)
__sample_negative_binomial args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorKeyArgs
          = catMaybes
              [("k",) . toRaw <$> (args !? #k :: Maybe (t u)),
               ("p",) . toRaw <$> (args !? #p :: Maybe (t u))]
      in
      applyRaw "_sample_negative_binomial" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__sample_normal" '(t, u) =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("mu", AttrOpt (t u)), '("sigma", AttrOpt (t u))]

__sample_normal ::
                forall a t u .
                  (Tensor t, Fullfilled "__sample_normal" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__sample_normal" '(t, u) a -> TensorApply (t u)
__sample_normal args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorKeyArgs
          = catMaybes
              [("mu",) . toRaw <$> (args !? #mu :: Maybe (t u)),
               ("sigma",) . toRaw <$> (args !? #sigma :: Maybe (t u))]
      in
      applyRaw "_sample_normal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__sample_poisson" '(t, u) =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("lam", AttrOpt (t u))]

__sample_poisson ::
                 forall a t u .
                   (Tensor t, Fullfilled "__sample_poisson" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__sample_poisson" '(t, u) a -> TensorApply (t u)
__sample_poisson args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorKeyArgs
          = catMaybes [("lam",) . toRaw <$> (args !? #lam :: Maybe (t u))]
      in
      applyRaw "_sample_poisson" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__sample_uniform" '(t, u) =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("low", AttrOpt (t u)), '("high", AttrOpt (t u))]

__sample_uniform ::
                 forall a t u .
                   (Tensor t, Fullfilled "__sample_uniform" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__sample_uniform" '(t, u) a -> TensorApply (t u)
__sample_uniform args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorKeyArgs
          = catMaybes
              [("low",) . toRaw <$> (args !? #low :: Maybe (t u)),
               ("high",) . toRaw <$> (args !? #high :: Maybe (t u))]
      in
      applyRaw "_sample_uniform" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__sample_unique_zipfian" '() =
     '[ '("range_max", AttrReq Int), '("shape", AttrOpt [Int])]

__sample_unique_zipfian ::
                        forall a t u .
                          (Tensor t, Fullfilled "__sample_unique_zipfian" '() a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__sample_unique_zipfian" '() a -> TensorApply (t u)
__sample_unique_zipfian args
  = let scalarArgs
          = catMaybes
              [("range_max",) . showValue <$> (args !? #range_max :: Maybe Int),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_sample_unique_zipfian" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__scatter_elemwise_div" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

__scatter_elemwise_div ::
                       forall a t u .
                         (Tensor t, Fullfilled "__scatter_elemwise_div" '(t, u) a,
                          HasCallStack, DType u) =>
                         ArgsHMap "__scatter_elemwise_div" '(t, u) a -> TensorApply (t u)
__scatter_elemwise_div args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_scatter_elemwise_div" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__scatter_minus_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__scatter_minus_scalar ::
                       forall a t u .
                         (Tensor t, Fullfilled "__scatter_minus_scalar" '(t, u) a,
                          HasCallStack, DType u) =>
                         ArgsHMap "__scatter_minus_scalar" '(t, u) a -> TensorApply (t u)
__scatter_minus_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_scatter_minus_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__scatter_plus_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("is_int", AttrOpt Bool),
        '("data", AttrOpt (t u))]

__scatter_plus_scalar ::
                      forall a t u .
                        (Tensor t, Fullfilled "__scatter_plus_scalar" '(t, u) a,
                         HasCallStack, DType u) =>
                        ArgsHMap "__scatter_plus_scalar" '(t, u) a -> TensorApply (t u)
__scatter_plus_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("is_int",) . showValue <$> (args !? #is_int :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_scatter_plus_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__scatter_set_nd" '(t, u) =
     '[ '("shape", AttrReq [Int]), '("lhs", AttrOpt (t u)),
        '("rhs", AttrOpt (t u)), '("indices", AttrOpt (t u))]

__scatter_set_nd ::
                 forall a t u .
                   (Tensor t, Fullfilled "__scatter_set_nd" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__scatter_set_nd" '(t, u) a -> TensorApply (t u)
__scatter_set_nd args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u)),
               ("indices",) . toRaw <$> (args !? #indices :: Maybe (t u))]
      in
      applyRaw "_scatter_set_nd" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__set_value" '() =
     '[ '("src", AttrOpt Float)]

__set_value ::
            forall a t u .
              (Tensor t, Fullfilled "__set_value" '() a, HasCallStack,
               DType u) =>
              ArgsHMap "__set_value" '() a -> TensorApply (t u)
__set_value args
  = let scalarArgs
          = catMaybes
              [("src",) . showValue <$> (args !? #src :: Maybe Float)]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_set_value" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__sg_mkldnn_conv" '() = '[]

__sg_mkldnn_conv ::
                 forall a t u .
                   (Tensor t, Fullfilled "__sg_mkldnn_conv" '() a, HasCallStack,
                    DType u) =>
                   ArgsHMap "__sg_mkldnn_conv" '() a -> TensorApply (t u)
__sg_mkldnn_conv args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_sg_mkldnn_conv" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__sg_mkldnn_fully_connected" '() = '[]

__sg_mkldnn_fully_connected ::
                            forall a t u .
                              (Tensor t, Fullfilled "__sg_mkldnn_fully_connected" '() a,
                               HasCallStack, DType u) =>
                              ArgsHMap "__sg_mkldnn_fully_connected" '() a -> TensorApply (t u)
__sg_mkldnn_fully_connected args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_sg_mkldnn_fully_connected" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__shuffle" '(t, u) =
     '[ '("data", AttrOpt (t u))]

__shuffle ::
          forall a t u .
            (Tensor t, Fullfilled "__shuffle" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "__shuffle" '(t, u) a -> TensorApply (t u)
__shuffle args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_shuffle" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__slice_assign" '(t, u) =
     '[ '("begin", AttrReq [Int]), '("end", AttrReq [Int]),
        '("step", AttrOpt [Int]), '("lhs", AttrOpt (t u)),
        '("rhs", AttrOpt (t u))]

__slice_assign ::
               forall a t u .
                 (Tensor t, Fullfilled "__slice_assign" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "__slice_assign" '(t, u) a -> TensorApply (t u)
__slice_assign args
  = let scalarArgs
          = catMaybes
              [("begin",) . showValue <$> (args !? #begin :: Maybe [Int]),
               ("end",) . showValue <$> (args !? #end :: Maybe [Int]),
               ("step",) . showValue <$> (args !? #step :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "_slice_assign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__slice_assign_scalar" '(t, u) =
     '[ '("scalar", AttrOpt Double), '("begin", AttrReq [Int]),
        '("end", AttrReq [Int]), '("step", AttrOpt [Int]),
        '("data", AttrOpt (t u))]

__slice_assign_scalar ::
                      forall a t u .
                        (Tensor t, Fullfilled "__slice_assign_scalar" '(t, u) a,
                         HasCallStack, DType u) =>
                        ArgsHMap "__slice_assign_scalar" '(t, u) a -> TensorApply (t u)
__slice_assign_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("begin",) . showValue <$> (args !? #begin :: Maybe [Int]),
               ("end",) . showValue <$> (args !? #end :: Maybe [Int]),
               ("step",) . showValue <$> (args !? #step :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_slice_assign_scalar" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__sparse_adagrad_update" '(t, u) =
     '[ '("lr", AttrReq Float), '("epsilon", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u)), '("history", AttrOpt (t u))]

__sparse_adagrad_update ::
                        forall a t u .
                          (Tensor t, Fullfilled "__sparse_adagrad_update" '(t, u) a,
                           HasCallStack, DType u) =>
                          ArgsHMap "__sparse_adagrad_update" '(t, u) a -> TensorApply (t u)
__sparse_adagrad_update args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("epsilon",) . showValue <$> (args !? #epsilon :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("grad",) . toRaw <$> (args !? #grad :: Maybe (t u)),
               ("history",) . toRaw <$> (args !? #history :: Maybe (t u))]
      in
      applyRaw "_sparse_adagrad_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__sparse_retain" '(t, u) =
     '[ '("data", AttrOpt (t u)), '("indices", AttrOpt (t u))]

__sparse_retain ::
                forall a t u .
                  (Tensor t, Fullfilled "__sparse_retain" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__sparse_retain" '(t, u) a -> TensorApply (t u)
__sparse_retain args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("indices",) . toRaw <$> (args !? #indices :: Maybe (t u))]
      in
      applyRaw "_sparse_retain" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__split_v2" '(t, u) =
     '[ '("indices", AttrReq [Int]), '("axis", AttrOpt Int),
        '("squeeze_axis", AttrOpt Bool), '("sections", AttrOpt Int),
        '("data", AttrOpt (t u))]

__split_v2 ::
           forall a t u .
             (Tensor t, Fullfilled "__split_v2" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "__split_v2" '(t, u) a -> TensorApply (t u)
__split_v2 args
  = let scalarArgs
          = catMaybes
              [("indices",) . showValue <$> (args !? #indices :: Maybe [Int]),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("squeeze_axis",) . showValue <$>
                 (args !? #squeeze_axis :: Maybe Bool),
               ("sections",) . showValue <$> (args !? #sections :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_split_v2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__split_v2_backward" '() = '[]

__split_v2_backward ::
                    forall a t u .
                      (Tensor t, Fullfilled "__split_v2_backward" '() a, HasCallStack,
                       DType u) =>
                      ArgsHMap "__split_v2_backward" '() a -> TensorApply (t u)
__split_v2_backward args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_split_v2_backward" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__square_sum" '(t, u) =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt (t u))]

__square_sum ::
             forall a t u .
               (Tensor t, Fullfilled "__square_sum" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "__square_sum" '(t, u) a -> TensorApply (t u)
__square_sum args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_square_sum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__unravel_index" '(t, u) =
     '[ '("shape", AttrOpt [Int]), '("data", AttrOpt (t u))]

__unravel_index ::
                forall a t u .
                  (Tensor t, Fullfilled "__unravel_index" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "__unravel_index" '(t, u) a -> TensorApply (t u)
__unravel_index args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "_unravel_index" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__zeros" '() =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                 "int64", "int8", "uint8"]))]

__zeros ::
        forall a t u .
          (Tensor t, Fullfilled "__zeros" '() a, HasCallStack, DType u) =>
          ArgsHMap "__zeros" '() a -> TensorApply (t u)
__zeros args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bfloat16", "bool", "float16", "float32", "float64", "int32",
                           "int64", "int8", "uint8"]))]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_zeros" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "__zeros_without_dtype" '() =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype", AttrOpt Int)]

__zeros_without_dtype ::
                      forall a t u .
                        (Tensor t, Fullfilled "__zeros_without_dtype" '() a, HasCallStack,
                         DType u) =>
                        ArgsHMap "__zeros_without_dtype" '() a -> TensorApply (t u)
__zeros_without_dtype args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$> (args !? #dtype :: Maybe Int)]
        tensorKeyArgs = catMaybes []
      in
      applyRaw "_zeros_without_dtype" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_abs" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_abs ::
     forall a t u .
       (Tensor t, Fullfilled "_abs" '(t, u) a, HasCallStack, DType u) =>
       ArgsHMap "_abs" '(t, u) a -> TensorApply (t u)
_abs args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "abs" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_adam_update" '(t, u) =
     '[ '("lr", AttrReq Float), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("lazy_update", AttrOpt Bool),
        '("weight", AttrOpt (t u)), '("grad", AttrOpt (t u)),
        '("mean", AttrOpt (t u)), '("var", AttrOpt (t u))]

_adam_update ::
             forall a t u .
               (Tensor t, Fullfilled "_adam_update" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "_adam_update" '(t, u) a -> TensorApply (t u)
_adam_update args
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
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("grad",) . toRaw <$> (args !? #grad :: Maybe (t u)),
               ("mean",) . toRaw <$> (args !? #mean :: Maybe (t u)),
               ("var",) . toRaw <$> (args !? #var :: Maybe (t u))]
      in
      applyRaw "adam_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_add_n" '(t, u) =
     '[ '("args", AttrOpt [t u])]

_add_n ::
       forall a t u .
         (Tensor t, Fullfilled "_add_n" '(t, u) a, HasCallStack, DType u) =>
         ArgsHMap "_add_n" '(t, u) a -> TensorApply (t u)
_add_n args
  = let scalarArgs
          = catMaybes [Just ("num_args", showValue (length tensorVarArgs))]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #args :: Maybe [RawTensor t])
      in applyRaw "add_n" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_all_finite" '(t, u) =
     '[ '("init_output", AttrOpt Bool), '("data", AttrOpt (t u))]

_all_finite ::
            forall a t u .
              (Tensor t, Fullfilled "_all_finite" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "_all_finite" '(t, u) a -> TensorApply (t u)
_all_finite args
  = let scalarArgs
          = catMaybes
              [("init_output",) . showValue <$>
                 (args !? #init_output :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "all_finite" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_amp_cast" '(t, u) =
     '[ '("dtype",
          AttrReq
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"])),
        '("data", AttrOpt (t u))]

_amp_cast ::
          forall a t u .
            (Tensor t, Fullfilled "_amp_cast" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "_amp_cast" '(t, u) a -> TensorApply (t u)
_amp_cast args
  = let scalarArgs
          = catMaybes
              [("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                           "int8", "uint8"]))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "amp_cast" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_amp_multicast" '(t, u) =
     '[ '("num_outputs", AttrReq Int), '("cast_narrow", AttrOpt Bool),
        '("data", AttrOpt [t u])]

_amp_multicast ::
               forall a t u .
                 (Tensor t, Fullfilled "_amp_multicast" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "_amp_multicast" '(t, u) a -> TensorApply (t u)
_amp_multicast args
  = let scalarArgs
          = catMaybes
              [("num_outputs",) . showValue <$>
                 (args !? #num_outputs :: Maybe Int),
               ("cast_narrow",) . showValue <$>
                 (args !? #cast_narrow :: Maybe Bool)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "amp_multicast" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_arccos" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_arccos ::
        forall a t u .
          (Tensor t, Fullfilled "_arccos" '(t, u) a, HasCallStack,
           DType u) =>
          ArgsHMap "_arccos" '(t, u) a -> TensorApply (t u)
_arccos args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "arccos" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_arccosh" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_arccosh ::
         forall a t u .
           (Tensor t, Fullfilled "_arccosh" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "_arccosh" '(t, u) a -> TensorApply (t u)
_arccosh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "arccosh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_arcsin" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_arcsin ::
        forall a t u .
          (Tensor t, Fullfilled "_arcsin" '(t, u) a, HasCallStack,
           DType u) =>
          ArgsHMap "_arcsin" '(t, u) a -> TensorApply (t u)
_arcsin args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "arcsin" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_arcsinh" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_arcsinh ::
         forall a t u .
           (Tensor t, Fullfilled "_arcsinh" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "_arcsinh" '(t, u) a -> TensorApply (t u)
_arcsinh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "arcsinh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_arctan" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_arctan ::
        forall a t u .
          (Tensor t, Fullfilled "_arctan" '(t, u) a, HasCallStack,
           DType u) =>
          ArgsHMap "_arctan" '(t, u) a -> TensorApply (t u)
_arctan args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "arctan" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_arctanh" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_arctanh ::
         forall a t u .
           (Tensor t, Fullfilled "_arctanh" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "_arctanh" '(t, u) a -> TensorApply (t u)
_arctanh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "arctanh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_argmax" '(t, u) =
     '[ '("axis", AttrOpt (Maybe Int)), '("keepdims", AttrOpt Bool),
        '("data", AttrOpt (t u))]

_argmax ::
        forall a t u .
          (Tensor t, Fullfilled "_argmax" '(t, u) a, HasCallStack,
           DType u) =>
          ArgsHMap "_argmax" '(t, u) a -> TensorApply (t u)
_argmax args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "argmax" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_argmax_channel" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_argmax_channel ::
                forall a t u .
                  (Tensor t, Fullfilled "_argmax_channel" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "_argmax_channel" '(t, u) a -> TensorApply (t u)
_argmax_channel args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "argmax_channel" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_argmin" '(t, u) =
     '[ '("axis", AttrOpt (Maybe Int)), '("keepdims", AttrOpt Bool),
        '("data", AttrOpt (t u))]

_argmin ::
        forall a t u .
          (Tensor t, Fullfilled "_argmin" '(t, u) a, HasCallStack,
           DType u) =>
          ArgsHMap "_argmin" '(t, u) a -> TensorApply (t u)
_argmin args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "argmin" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_argsort" '(t, u) =
     '[ '("axis", AttrOpt (Maybe Int)), '("is_ascend", AttrOpt Bool),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "uint8"])),
        '("data", AttrOpt (t u))]

_argsort ::
         forall a t u .
           (Tensor t, Fullfilled "_argsort" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "_argsort" '(t, u) a -> TensorApply (t u)
_argsort args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("is_ascend",) . showValue <$> (args !? #is_ascend :: Maybe Bool),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["float16", "float32", "float64", "int32", "int64", "uint8"]))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "argsort" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_batch_dot" '(t, u) =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("forward_stype",
          AttrOpt (Maybe (EnumType '["csr", "default", "row_sparse"]))),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_batch_dot ::
           forall a t u .
             (Tensor t, Fullfilled "_batch_dot" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "_batch_dot" '(t, u) a -> TensorApply (t u)
_batch_dot args
  = let scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$>
                 (args !? #transpose_a :: Maybe Bool),
               ("transpose_b",) . showValue <$>
                 (args !? #transpose_b :: Maybe Bool),
               ("forward_stype",) . showValue <$>
                 (args !? #forward_stype ::
                    Maybe (Maybe (EnumType '["csr", "default", "row_sparse"])))]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "batch_dot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_batch_take" '(t, u) =
     '[ '("a", AttrOpt (t u)), '("indices", AttrOpt (t u))]

_batch_take ::
            forall a t u .
              (Tensor t, Fullfilled "_batch_take" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "_batch_take" '(t, u) a -> TensorApply (t u)
_batch_take args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> (args !? #a :: Maybe (t u)),
               ("indices",) . toRaw <$> (args !? #indices :: Maybe (t u))]
      in
      applyRaw "batch_take" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_broadcast_add" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_add ::
               forall a t u .
                 (Tensor t, Fullfilled "_broadcast_add" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "_broadcast_add" '(t, u) a -> TensorApply (t u)
_broadcast_add args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "broadcast_add" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_broadcast_axis" '(t, u) =
     '[ '("axis", AttrOpt [Int]), '("size", AttrOpt [Int]),
        '("data", AttrOpt (t u))]

_broadcast_axis ::
                forall a t u .
                  (Tensor t, Fullfilled "_broadcast_axis" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "_broadcast_axis" '(t, u) a -> TensorApply (t u)
_broadcast_axis args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int]),
               ("size",) . showValue <$> (args !? #size :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "broadcast_axis" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_broadcast_div" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_div ::
               forall a t u .
                 (Tensor t, Fullfilled "_broadcast_div" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "_broadcast_div" '(t, u) a -> TensorApply (t u)
_broadcast_div args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "broadcast_div" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_broadcast_equal" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_equal ::
                 forall a t u .
                   (Tensor t, Fullfilled "_broadcast_equal" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "_broadcast_equal" '(t, u) a -> TensorApply (t u)
_broadcast_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "broadcast_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_broadcast_greater" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_greater ::
                   forall a t u .
                     (Tensor t, Fullfilled "_broadcast_greater" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "_broadcast_greater" '(t, u) a -> TensorApply (t u)
_broadcast_greater args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "broadcast_greater" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_broadcast_greater_equal" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_greater_equal ::
                         forall a t u .
                           (Tensor t, Fullfilled "_broadcast_greater_equal" '(t, u) a,
                            HasCallStack, DType u) =>
                           ArgsHMap "_broadcast_greater_equal" '(t, u) a -> TensorApply (t u)
_broadcast_greater_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "broadcast_greater_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_broadcast_hypot" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_hypot ::
                 forall a t u .
                   (Tensor t, Fullfilled "_broadcast_hypot" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "_broadcast_hypot" '(t, u) a -> TensorApply (t u)
_broadcast_hypot args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "broadcast_hypot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_broadcast_lesser" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_lesser ::
                  forall a t u .
                    (Tensor t, Fullfilled "_broadcast_lesser" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "_broadcast_lesser" '(t, u) a -> TensorApply (t u)
_broadcast_lesser args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "broadcast_lesser" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_broadcast_lesser_equal" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_lesser_equal ::
                        forall a t u .
                          (Tensor t, Fullfilled "_broadcast_lesser_equal" '(t, u) a,
                           HasCallStack, DType u) =>
                          ArgsHMap "_broadcast_lesser_equal" '(t, u) a -> TensorApply (t u)
_broadcast_lesser_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "broadcast_lesser_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_broadcast_like" '(t, u) =
     '[ '("lhs_axes", AttrOpt (Maybe [Int])),
        '("rhs_axes", AttrOpt (Maybe [Int])), '("lhs", AttrOpt (t u)),
        '("rhs", AttrOpt (t u))]

_broadcast_like ::
                forall a t u .
                  (Tensor t, Fullfilled "_broadcast_like" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "_broadcast_like" '(t, u) a -> TensorApply (t u)
_broadcast_like args
  = let scalarArgs
          = catMaybes
              [("lhs_axes",) . showValue <$>
                 (args !? #lhs_axes :: Maybe (Maybe [Int])),
               ("rhs_axes",) . showValue <$>
                 (args !? #rhs_axes :: Maybe (Maybe [Int]))]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "broadcast_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_broadcast_logical_and" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_logical_and ::
                       forall a t u .
                         (Tensor t, Fullfilled "_broadcast_logical_and" '(t, u) a,
                          HasCallStack, DType u) =>
                         ArgsHMap "_broadcast_logical_and" '(t, u) a -> TensorApply (t u)
_broadcast_logical_and args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "broadcast_logical_and" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_broadcast_logical_or" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_logical_or ::
                      forall a t u .
                        (Tensor t, Fullfilled "_broadcast_logical_or" '(t, u) a,
                         HasCallStack, DType u) =>
                        ArgsHMap "_broadcast_logical_or" '(t, u) a -> TensorApply (t u)
_broadcast_logical_or args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "broadcast_logical_or" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_broadcast_logical_xor" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_logical_xor ::
                       forall a t u .
                         (Tensor t, Fullfilled "_broadcast_logical_xor" '(t, u) a,
                          HasCallStack, DType u) =>
                         ArgsHMap "_broadcast_logical_xor" '(t, u) a -> TensorApply (t u)
_broadcast_logical_xor args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "broadcast_logical_xor" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_broadcast_maximum" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_maximum ::
                   forall a t u .
                     (Tensor t, Fullfilled "_broadcast_maximum" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "_broadcast_maximum" '(t, u) a -> TensorApply (t u)
_broadcast_maximum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "broadcast_maximum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_broadcast_minimum" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_minimum ::
                   forall a t u .
                     (Tensor t, Fullfilled "_broadcast_minimum" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "_broadcast_minimum" '(t, u) a -> TensorApply (t u)
_broadcast_minimum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "broadcast_minimum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_broadcast_mod" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_mod ::
               forall a t u .
                 (Tensor t, Fullfilled "_broadcast_mod" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "_broadcast_mod" '(t, u) a -> TensorApply (t u)
_broadcast_mod args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "broadcast_mod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_broadcast_mul" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_mul ::
               forall a t u .
                 (Tensor t, Fullfilled "_broadcast_mul" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "_broadcast_mul" '(t, u) a -> TensorApply (t u)
_broadcast_mul args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "broadcast_mul" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_broadcast_not_equal" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_not_equal ::
                     forall a t u .
                       (Tensor t, Fullfilled "_broadcast_not_equal" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "_broadcast_not_equal" '(t, u) a -> TensorApply (t u)
_broadcast_not_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "broadcast_not_equal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_broadcast_power" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_power ::
                 forall a t u .
                   (Tensor t, Fullfilled "_broadcast_power" '(t, u) a, HasCallStack,
                    DType u) =>
                   ArgsHMap "_broadcast_power" '(t, u) a -> TensorApply (t u)
_broadcast_power args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "broadcast_power" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_broadcast_sub" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_broadcast_sub ::
               forall a t u .
                 (Tensor t, Fullfilled "_broadcast_sub" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "_broadcast_sub" '(t, u) a -> TensorApply (t u)
_broadcast_sub args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "broadcast_sub" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_broadcast_to" '(t, u) =
     '[ '("shape", AttrOpt [Int]), '("data", AttrOpt (t u))]

_broadcast_to ::
              forall a t u .
                (Tensor t, Fullfilled "_broadcast_to" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "_broadcast_to" '(t, u) a -> TensorApply (t u)
_broadcast_to args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "broadcast_to" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_cast_storage" '(t, u) =
     '[ '("stype",
          AttrReq (EnumType '["csr", "default", "row_sparse"])),
        '("data", AttrOpt (t u))]

_cast_storage ::
              forall a t u .
                (Tensor t, Fullfilled "_cast_storage" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "_cast_storage" '(t, u) a -> TensorApply (t u)
_cast_storage args
  = let scalarArgs
          = catMaybes
              [("stype",) . showValue <$>
                 (args !? #stype ::
                    Maybe (EnumType '["csr", "default", "row_sparse"]))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "cast_storage" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_cbrt" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_cbrt ::
      forall a t u .
        (Tensor t, Fullfilled "_cbrt" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "_cbrt" '(t, u) a -> TensorApply (t u)
_cbrt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "cbrt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_ceil" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_ceil ::
      forall a t u .
        (Tensor t, Fullfilled "_ceil" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "_ceil" '(t, u) a -> TensorApply (t u)
_ceil args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "ceil" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_clip" '(t, u) =
     '[ '("a_min", AttrReq Float), '("a_max", AttrReq Float),
        '("data", AttrOpt (t u))]

_clip ::
      forall a t u .
        (Tensor t, Fullfilled "_clip" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "_clip" '(t, u) a -> TensorApply (t u)
_clip args
  = let scalarArgs
          = catMaybes
              [("a_min",) . showValue <$> (args !? #a_min :: Maybe Float),
               ("a_max",) . showValue <$> (args !? #a_max :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "clip" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_col2im" '(t, u) =
     '[ '("output_size", AttrReq [Int]), '("kernel", AttrReq [Int]),
        '("stride", AttrOpt [Int]), '("dilate", AttrOpt [Int]),
        '("pad", AttrOpt [Int]), '("data", AttrOpt (t u))]

_col2im ::
        forall a t u .
          (Tensor t, Fullfilled "_col2im" '(t, u) a, HasCallStack,
           DType u) =>
          ArgsHMap "_col2im" '(t, u) a -> TensorApply (t u)
_col2im args
  = let scalarArgs
          = catMaybes
              [("output_size",) . showValue <$>
                 (args !? #output_size :: Maybe [Int]),
               ("kernel",) . showValue <$> (args !? #kernel :: Maybe [Int]),
               ("stride",) . showValue <$> (args !? #stride :: Maybe [Int]),
               ("dilate",) . showValue <$> (args !? #dilate :: Maybe [Int]),
               ("pad",) . showValue <$> (args !? #pad :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "col2im" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_cos" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_cos ::
     forall a t u .
       (Tensor t, Fullfilled "_cos" '(t, u) a, HasCallStack, DType u) =>
       ArgsHMap "_cos" '(t, u) a -> TensorApply (t u)
_cos args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "cos" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_cosh" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_cosh ::
      forall a t u .
        (Tensor t, Fullfilled "_cosh" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "_cosh" '(t, u) a -> TensorApply (t u)
_cosh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "cosh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_degrees" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_degrees ::
         forall a t u .
           (Tensor t, Fullfilled "_degrees" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "_degrees" '(t, u) a -> TensorApply (t u)
_degrees args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "degrees" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_depth_to_space" '(t, u) =
     '[ '("block_size", AttrReq Int), '("data", AttrOpt (t u))]

_depth_to_space ::
                forall a t u .
                  (Tensor t, Fullfilled "_depth_to_space" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "_depth_to_space" '(t, u) a -> TensorApply (t u)
_depth_to_space args
  = let scalarArgs
          = catMaybes
              [("block_size",) . showValue <$>
                 (args !? #block_size :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "depth_to_space" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_diag" '(t, u) =
     '[ '("k", AttrOpt Int), '("axis1", AttrOpt Int),
        '("axis2", AttrOpt Int), '("data", AttrOpt (t u))]

_diag ::
      forall a t u .
        (Tensor t, Fullfilled "_diag" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "_diag" '(t, u) a -> TensorApply (t u)
_diag args
  = let scalarArgs
          = catMaybes
              [("k",) . showValue <$> (args !? #k :: Maybe Int),
               ("axis1",) . showValue <$> (args !? #axis1 :: Maybe Int),
               ("axis2",) . showValue <$> (args !? #axis2 :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "diag" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_dot" '(t, u) =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("forward_stype",
          AttrOpt (Maybe (EnumType '["csr", "default", "row_sparse"]))),
        '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_dot ::
     forall a t u .
       (Tensor t, Fullfilled "_dot" '(t, u) a, HasCallStack, DType u) =>
       ArgsHMap "_dot" '(t, u) a -> TensorApply (t u)
_dot args
  = let scalarArgs
          = catMaybes
              [("transpose_a",) . showValue <$>
                 (args !? #transpose_a :: Maybe Bool),
               ("transpose_b",) . showValue <$>
                 (args !? #transpose_b :: Maybe Bool),
               ("forward_stype",) . showValue <$>
                 (args !? #forward_stype ::
                    Maybe (Maybe (EnumType '["csr", "default", "row_sparse"])))]
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "dot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_elemwise_add" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_elemwise_add ::
              forall a t u .
                (Tensor t, Fullfilled "_elemwise_add" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "_elemwise_add" '(t, u) a -> TensorApply (t u)
_elemwise_add args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "elemwise_add" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_elemwise_div" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_elemwise_div ::
              forall a t u .
                (Tensor t, Fullfilled "_elemwise_div" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "_elemwise_div" '(t, u) a -> TensorApply (t u)
_elemwise_div args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "elemwise_div" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_elemwise_mul" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_elemwise_mul ::
              forall a t u .
                (Tensor t, Fullfilled "_elemwise_mul" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "_elemwise_mul" '(t, u) a -> TensorApply (t u)
_elemwise_mul args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "elemwise_mul" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_elemwise_sub" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("rhs", AttrOpt (t u))]

_elemwise_sub ::
              forall a t u .
                (Tensor t, Fullfilled "_elemwise_sub" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "_elemwise_sub" '(t, u) a -> TensorApply (t u)
_elemwise_sub args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "elemwise_sub" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_erf" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_erf ::
     forall a t u .
       (Tensor t, Fullfilled "_erf" '(t, u) a, HasCallStack, DType u) =>
       ArgsHMap "_erf" '(t, u) a -> TensorApply (t u)
_erf args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "erf" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_erfinv" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_erfinv ::
        forall a t u .
          (Tensor t, Fullfilled "_erfinv" '(t, u) a, HasCallStack,
           DType u) =>
          ArgsHMap "_erfinv" '(t, u) a -> TensorApply (t u)
_erfinv args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "erfinv" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_exp" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_exp ::
     forall a t u .
       (Tensor t, Fullfilled "_exp" '(t, u) a, HasCallStack, DType u) =>
       ArgsHMap "_exp" '(t, u) a -> TensorApply (t u)
_exp args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "exp" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_expand_dims" '(t, u) =
     '[ '("axis", AttrReq Int), '("data", AttrOpt (t u))]

_expand_dims ::
             forall a t u .
               (Tensor t, Fullfilled "_expand_dims" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "_expand_dims" '(t, u) a -> TensorApply (t u)
_expand_dims args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "expand_dims" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_expm1" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_expm1 ::
       forall a t u .
         (Tensor t, Fullfilled "_expm1" '(t, u) a, HasCallStack, DType u) =>
         ArgsHMap "_expm1" '(t, u) a -> TensorApply (t u)
_expm1 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "expm1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_fill_element_0index" '(t, u) =
     '[ '("lhs", AttrOpt (t u)), '("mhs", AttrOpt (t u)),
        '("rhs", AttrOpt (t u))]

_fill_element_0index ::
                     forall a t u .
                       (Tensor t, Fullfilled "_fill_element_0index" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "_fill_element_0index" '(t, u) a -> TensorApply (t u)
_fill_element_0index args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("mhs",) . toRaw <$> (args !? #mhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "fill_element_0index" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_fix" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_fix ::
     forall a t u .
       (Tensor t, Fullfilled "_fix" '(t, u) a, HasCallStack, DType u) =>
       ArgsHMap "_fix" '(t, u) a -> TensorApply (t u)
_fix args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "fix" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_floor" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_floor ::
       forall a t u .
         (Tensor t, Fullfilled "_floor" '(t, u) a, HasCallStack, DType u) =>
         ArgsHMap "_floor" '(t, u) a -> TensorApply (t u)
_floor args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "floor" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_ftml_update" '(t, u) =
     '[ '("lr", AttrReq Float), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Double),
        '("t", AttrReq Int), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float), '("clip_grad", AttrOpt Float),
        '("weight", AttrOpt (t u)), '("grad", AttrOpt (t u)),
        '("d", AttrOpt (t u)), '("v", AttrOpt (t u)),
        '("z", AttrOpt (t u))]

_ftml_update ::
             forall a t u .
               (Tensor t, Fullfilled "_ftml_update" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "_ftml_update" '(t, u) a -> TensorApply (t u)
_ftml_update args
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
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("grad",) . toRaw <$> (args !? #grad :: Maybe (t u)),
               ("d",) . toRaw <$> (args !? #d :: Maybe (t u)),
               ("v",) . toRaw <$> (args !? #v :: Maybe (t u)),
               ("z",) . toRaw <$> (args !? #z :: Maybe (t u))]
      in
      applyRaw "ftml_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_ftrl_update" '(t, u) =
     '[ '("lr", AttrReq Float), '("lamda1", AttrOpt Float),
        '("beta", AttrOpt Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u)), '("z", AttrOpt (t u)),
        '("n", AttrOpt (t u))]

_ftrl_update ::
             forall a t u .
               (Tensor t, Fullfilled "_ftrl_update" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "_ftrl_update" '(t, u) a -> TensorApply (t u)
_ftrl_update args
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
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("grad",) . toRaw <$> (args !? #grad :: Maybe (t u)),
               ("z",) . toRaw <$> (args !? #z :: Maybe (t u)),
               ("n",) . toRaw <$> (args !? #n :: Maybe (t u))]
      in
      applyRaw "ftrl_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_gamma" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_gamma ::
       forall a t u .
         (Tensor t, Fullfilled "_gamma" '(t, u) a, HasCallStack, DType u) =>
         ArgsHMap "_gamma" '(t, u) a -> TensorApply (t u)
_gamma args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "gamma" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_gammaln" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_gammaln ::
         forall a t u .
           (Tensor t, Fullfilled "_gammaln" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "_gammaln" '(t, u) a -> TensorApply (t u)
_gammaln args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "gammaln" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_gather_nd" '(t, u) =
     '[ '("data", AttrOpt (t u)), '("indices", AttrOpt (t u))]

_gather_nd ::
           forall a t u .
             (Tensor t, Fullfilled "_gather_nd" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "_gather_nd" '(t, u) a -> TensorApply (t u)
_gather_nd args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("indices",) . toRaw <$> (args !? #indices :: Maybe (t u))]
      in
      applyRaw "gather_nd" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_hard_sigmoid" '(t, u) =
     '[ '("alpha", AttrOpt Float), '("beta", AttrOpt Float),
        '("data", AttrOpt (t u))]

_hard_sigmoid ::
              forall a t u .
                (Tensor t, Fullfilled "_hard_sigmoid" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "_hard_sigmoid" '(t, u) a -> TensorApply (t u)
_hard_sigmoid args
  = let scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> (args !? #alpha :: Maybe Float),
               ("beta",) . showValue <$> (args !? #beta :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "hard_sigmoid" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_im2col" '(t, u) =
     '[ '("kernel", AttrReq [Int]), '("stride", AttrOpt [Int]),
        '("dilate", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("data", AttrOpt (t u))]

_im2col ::
        forall a t u .
          (Tensor t, Fullfilled "_im2col" '(t, u) a, HasCallStack,
           DType u) =>
          ArgsHMap "_im2col" '(t, u) a -> TensorApply (t u)
_im2col args
  = let scalarArgs
          = catMaybes
              [("kernel",) . showValue <$> (args !? #kernel :: Maybe [Int]),
               ("stride",) . showValue <$> (args !? #stride :: Maybe [Int]),
               ("dilate",) . showValue <$> (args !? #dilate :: Maybe [Int]),
               ("pad",) . showValue <$> (args !? #pad :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "im2col" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_khatri_rao" '(t, u) =
     '[ '("args", AttrOpt [t u])]

_khatri_rao ::
            forall a t u .
              (Tensor t, Fullfilled "_khatri_rao" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "_khatri_rao" '(t, u) a -> TensorApply (t u)
_khatri_rao args
  = let scalarArgs
          = catMaybes [Just ("num_args", showValue (length tensorVarArgs))]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #args :: Maybe [RawTensor t])
      in applyRaw "khatri_rao" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_lamb_update_phase1" '(t, u) =
     '[ '("beta1", AttrOpt Float), '("beta2", AttrOpt Float),
        '("epsilon", AttrOpt Float), '("t", AttrReq Int),
        '("bias_correction", AttrOpt Bool), '("wd", AttrReq Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u)), '("mean", AttrOpt (t u)),
        '("var", AttrOpt (t u))]

_lamb_update_phase1 ::
                    forall a t u .
                      (Tensor t, Fullfilled "_lamb_update_phase1" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "_lamb_update_phase1" '(t, u) a -> TensorApply (t u)
_lamb_update_phase1 args
  = let scalarArgs
          = catMaybes
              [("beta1",) . showValue <$> (args !? #beta1 :: Maybe Float),
               ("beta2",) . showValue <$> (args !? #beta2 :: Maybe Float),
               ("epsilon",) . showValue <$> (args !? #epsilon :: Maybe Float),
               ("t",) . showValue <$> (args !? #t :: Maybe Int),
               ("bias_correction",) . showValue <$>
                 (args !? #bias_correction :: Maybe Bool),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("grad",) . toRaw <$> (args !? #grad :: Maybe (t u)),
               ("mean",) . toRaw <$> (args !? #mean :: Maybe (t u)),
               ("var",) . toRaw <$> (args !? #var :: Maybe (t u))]
      in
      applyRaw "lamb_update_phase1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_lamb_update_phase2" '(t, u) =
     '[ '("lr", AttrReq Float), '("lower_bound", AttrOpt Float),
        '("upper_bound", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("g", AttrOpt (t u)), '("r1", AttrOpt (t u)),
        '("r2", AttrOpt (t u))]

_lamb_update_phase2 ::
                    forall a t u .
                      (Tensor t, Fullfilled "_lamb_update_phase2" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "_lamb_update_phase2" '(t, u) a -> TensorApply (t u)
_lamb_update_phase2 args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("lower_bound",) . showValue <$>
                 (args !? #lower_bound :: Maybe Float),
               ("upper_bound",) . showValue <$>
                 (args !? #upper_bound :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("g",) . toRaw <$> (args !? #g :: Maybe (t u)),
               ("r1",) . toRaw <$> (args !? #r1 :: Maybe (t u)),
               ("r2",) . toRaw <$> (args !? #r2 :: Maybe (t u))]
      in
      applyRaw "lamb_update_phase2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_log" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_log ::
     forall a t u .
       (Tensor t, Fullfilled "_log" '(t, u) a, HasCallStack, DType u) =>
       ArgsHMap "_log" '(t, u) a -> TensorApply (t u)
_log args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "log" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_log10" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_log10 ::
       forall a t u .
         (Tensor t, Fullfilled "_log10" '(t, u) a, HasCallStack, DType u) =>
         ArgsHMap "_log10" '(t, u) a -> TensorApply (t u)
_log10 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "log10" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_log1p" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_log1p ::
       forall a t u .
         (Tensor t, Fullfilled "_log1p" '(t, u) a, HasCallStack, DType u) =>
         ArgsHMap "_log1p" '(t, u) a -> TensorApply (t u)
_log1p args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "log1p" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_log2" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_log2 ::
      forall a t u .
        (Tensor t, Fullfilled "_log2" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "_log2" '(t, u) a -> TensorApply (t u)
_log2 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "log2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_log_softmax" '(t, u) =
     '[ '("axis", AttrOpt Int),
        '("temperature", AttrOpt (Maybe Double)),
        '("dtype",
          AttrOpt (Maybe (EnumType '["float16", "float32", "float64"]))),
        '("use_length", AttrOpt (Maybe Bool)), '("data", AttrOpt (t u))]

_log_softmax ::
             forall a t u .
               (Tensor t, Fullfilled "_log_softmax" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "_log_softmax" '(t, u) a -> TensorApply (t u)
_log_softmax args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("temperature",) . showValue <$>
                 (args !? #temperature :: Maybe (Maybe Double)),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (Maybe (EnumType '["float16", "float32", "float64"]))),
               ("use_length",) . showValue <$>
                 (args !? #use_length :: Maybe (Maybe Bool))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "log_softmax" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_logical_not" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_logical_not ::
             forall a t u .
               (Tensor t, Fullfilled "_logical_not" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "_logical_not" '(t, u) a -> TensorApply (t u)
_logical_not args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "logical_not" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_make_loss" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_make_loss ::
           forall a t u .
             (Tensor t, Fullfilled "_make_loss" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "_make_loss" '(t, u) a -> TensorApply (t u)
_make_loss args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "make_loss" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_max" '(t, u) =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt (t u))]

_max ::
     forall a t u .
       (Tensor t, Fullfilled "_max" '(t, u) a, HasCallStack, DType u) =>
       ArgsHMap "_max" '(t, u) a -> TensorApply (t u)
_max args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "max" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_mean" '(t, u) =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt (t u))]

_mean ::
      forall a t u .
        (Tensor t, Fullfilled "_mean" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "_mean" '(t, u) a -> TensorApply (t u)
_mean args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "mean" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_min" '(t, u) =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt (t u))]

_min ::
     forall a t u .
       (Tensor t, Fullfilled "_min" '(t, u) a, HasCallStack, DType u) =>
       ArgsHMap "_min" '(t, u) a -> TensorApply (t u)
_min args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "min" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_moments" '(t, u) =
     '[ '("axes", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("data", AttrOpt (t u))]

_moments ::
         forall a t u .
           (Tensor t, Fullfilled "_moments" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "_moments" '(t, u) a -> TensorApply (t u)
_moments args
  = let scalarArgs
          = catMaybes
              [("axes",) . showValue <$> (args !? #axes :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "moments" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_mp_lamb_update_phase1" '(t, u) =
     '[ '("beta1", AttrOpt Float), '("beta2", AttrOpt Float),
        '("epsilon", AttrOpt Float), '("t", AttrReq Int),
        '("bias_correction", AttrOpt Bool), '("wd", AttrReq Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u)), '("mean", AttrOpt (t u)),
        '("var", AttrOpt (t u)), '("weight32", AttrOpt (t u))]

_mp_lamb_update_phase1 ::
                       forall a t u .
                         (Tensor t, Fullfilled "_mp_lamb_update_phase1" '(t, u) a,
                          HasCallStack, DType u) =>
                         ArgsHMap "_mp_lamb_update_phase1" '(t, u) a -> TensorApply (t u)
_mp_lamb_update_phase1 args
  = let scalarArgs
          = catMaybes
              [("beta1",) . showValue <$> (args !? #beta1 :: Maybe Float),
               ("beta2",) . showValue <$> (args !? #beta2 :: Maybe Float),
               ("epsilon",) . showValue <$> (args !? #epsilon :: Maybe Float),
               ("t",) . showValue <$> (args !? #t :: Maybe Int),
               ("bias_correction",) . showValue <$>
                 (args !? #bias_correction :: Maybe Bool),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("grad",) . toRaw <$> (args !? #grad :: Maybe (t u)),
               ("mean",) . toRaw <$> (args !? #mean :: Maybe (t u)),
               ("var",) . toRaw <$> (args !? #var :: Maybe (t u)),
               ("weight32",) . toRaw <$> (args !? #weight32 :: Maybe (t u))]
      in
      applyRaw "mp_lamb_update_phase1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_mp_lamb_update_phase2" '(t, u) =
     '[ '("lr", AttrReq Float), '("lower_bound", AttrOpt Float),
        '("upper_bound", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("g", AttrOpt (t u)), '("r1", AttrOpt (t u)),
        '("r2", AttrOpt (t u)), '("weight32", AttrOpt (t u))]

_mp_lamb_update_phase2 ::
                       forall a t u .
                         (Tensor t, Fullfilled "_mp_lamb_update_phase2" '(t, u) a,
                          HasCallStack, DType u) =>
                         ArgsHMap "_mp_lamb_update_phase2" '(t, u) a -> TensorApply (t u)
_mp_lamb_update_phase2 args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("lower_bound",) . showValue <$>
                 (args !? #lower_bound :: Maybe Float),
               ("upper_bound",) . showValue <$>
                 (args !? #upper_bound :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("g",) . toRaw <$> (args !? #g :: Maybe (t u)),
               ("r1",) . toRaw <$> (args !? #r1 :: Maybe (t u)),
               ("r2",) . toRaw <$> (args !? #r2 :: Maybe (t u)),
               ("weight32",) . toRaw <$> (args !? #weight32 :: Maybe (t u))]
      in
      applyRaw "mp_lamb_update_phase2" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_mp_nag_mom_update" '(t, u) =
     '[ '("lr", AttrReq Float), '("momentum", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u)), '("mom", AttrOpt (t u)),
        '("weight32", AttrOpt (t u))]

_mp_nag_mom_update ::
                   forall a t u .
                     (Tensor t, Fullfilled "_mp_nag_mom_update" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "_mp_nag_mom_update" '(t, u) a -> TensorApply (t u)
_mp_nag_mom_update args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("grad",) . toRaw <$> (args !? #grad :: Maybe (t u)),
               ("mom",) . toRaw <$> (args !? #mom :: Maybe (t u)),
               ("weight32",) . toRaw <$> (args !? #weight32 :: Maybe (t u))]
      in
      applyRaw "mp_nag_mom_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_mp_sgd_mom_update" '(t, u) =
     '[ '("lr", AttrReq Float), '("momentum", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("lazy_update", AttrOpt Bool),
        '("weight", AttrOpt (t u)), '("grad", AttrOpt (t u)),
        '("mom", AttrOpt (t u)), '("weight32", AttrOpt (t u))]

_mp_sgd_mom_update ::
                   forall a t u .
                     (Tensor t, Fullfilled "_mp_sgd_mom_update" '(t, u) a, HasCallStack,
                      DType u) =>
                     ArgsHMap "_mp_sgd_mom_update" '(t, u) a -> TensorApply (t u)
_mp_sgd_mom_update args
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
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("grad",) . toRaw <$> (args !? #grad :: Maybe (t u)),
               ("mom",) . toRaw <$> (args !? #mom :: Maybe (t u)),
               ("weight32",) . toRaw <$> (args !? #weight32 :: Maybe (t u))]
      in
      applyRaw "mp_sgd_mom_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_mp_sgd_update" '(t, u) =
     '[ '("lr", AttrReq Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("lazy_update", AttrOpt Bool),
        '("weight", AttrOpt (t u)), '("grad", AttrOpt (t u)),
        '("weight32", AttrOpt (t u))]

_mp_sgd_update ::
               forall a t u .
                 (Tensor t, Fullfilled "_mp_sgd_update" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "_mp_sgd_update" '(t, u) a -> TensorApply (t u)
_mp_sgd_update args
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
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("grad",) . toRaw <$> (args !? #grad :: Maybe (t u)),
               ("weight32",) . toRaw <$> (args !? #weight32 :: Maybe (t u))]
      in
      applyRaw "mp_sgd_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_multi_all_finite" '(t, u) =
     '[ '("num_arrays", AttrOpt Int), '("init_output", AttrOpt Bool),
        '("data", AttrOpt [t u])]

_multi_all_finite ::
                  forall a t u .
                    (Tensor t, Fullfilled "_multi_all_finite" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "_multi_all_finite" '(t, u) a -> TensorApply (t u)
_multi_all_finite args
  = let scalarArgs
          = catMaybes
              [("num_arrays",) . showValue <$>
                 (args !? #num_arrays :: Maybe Int),
               ("init_output",) . showValue <$>
                 (args !? #init_output :: Maybe Bool)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "multi_all_finite" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_multi_lars" '(t, u) =
     '[ '("eta", AttrReq Float), '("eps", AttrReq Float),
        '("rescale_grad", AttrOpt Float), '("lrs", AttrOpt (t u)),
        '("weights_sum_sq", AttrOpt (t u)),
        '("grads_sum_sq", AttrOpt (t u)), '("wds", AttrOpt (t u))]

_multi_lars ::
            forall a t u .
              (Tensor t, Fullfilled "_multi_lars" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "_multi_lars" '(t, u) a -> TensorApply (t u)
_multi_lars args
  = let scalarArgs
          = catMaybes
              [("eta",) . showValue <$> (args !? #eta :: Maybe Float),
               ("eps",) . showValue <$> (args !? #eps :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("lrs",) . toRaw <$> (args !? #lrs :: Maybe (t u)),
               ("weights_sum_sq",) . toRaw <$>
                 (args !? #weights_sum_sq :: Maybe (t u)),
               ("grads_sum_sq",) . toRaw <$>
                 (args !? #grads_sum_sq :: Maybe (t u)),
               ("wds",) . toRaw <$> (args !? #wds :: Maybe (t u))]
      in
      applyRaw "multi_lars" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_multi_mp_sgd_mom_update" '(t, u) =
     '[ '("lrs", AttrReq [Float]), '("wds", AttrReq [Float]),
        '("momentum", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("data", AttrOpt [t u])]

_multi_mp_sgd_mom_update ::
                         forall a t u .
                           (Tensor t, Fullfilled "_multi_mp_sgd_mom_update" '(t, u) a,
                            HasCallStack, DType u) =>
                           ArgsHMap "_multi_mp_sgd_mom_update" '(t, u) a -> TensorApply (t u)
_multi_mp_sgd_mom_update args
  = let scalarArgs
          = catMaybes
              [("lrs",) . showValue <$> (args !? #lrs :: Maybe [Float]),
               ("wds",) . showValue <$> (args !? #wds :: Maybe [Float]),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("num_weights",) . showValue <$>
                 (args !? #num_weights :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in
      applyRaw "multi_mp_sgd_mom_update" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_multi_mp_sgd_update" '(t, u) =
     '[ '("lrs", AttrReq [Float]), '("wds", AttrReq [Float]),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("data", AttrOpt [t u])]

_multi_mp_sgd_update ::
                     forall a t u .
                       (Tensor t, Fullfilled "_multi_mp_sgd_update" '(t, u) a,
                        HasCallStack, DType u) =>
                       ArgsHMap "_multi_mp_sgd_update" '(t, u) a -> TensorApply (t u)
_multi_mp_sgd_update args
  = let scalarArgs
          = catMaybes
              [("lrs",) . showValue <$> (args !? #lrs :: Maybe [Float]),
               ("wds",) . showValue <$> (args !? #wds :: Maybe [Float]),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("num_weights",) . showValue <$>
                 (args !? #num_weights :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "multi_mp_sgd_update" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_multi_sgd_mom_update" '(t, u) =
     '[ '("lrs", AttrReq [Float]), '("wds", AttrReq [Float]),
        '("momentum", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("data", AttrOpt [t u])]

_multi_sgd_mom_update ::
                      forall a t u .
                        (Tensor t, Fullfilled "_multi_sgd_mom_update" '(t, u) a,
                         HasCallStack, DType u) =>
                        ArgsHMap "_multi_sgd_mom_update" '(t, u) a -> TensorApply (t u)
_multi_sgd_mom_update args
  = let scalarArgs
          = catMaybes
              [("lrs",) . showValue <$> (args !? #lrs :: Maybe [Float]),
               ("wds",) . showValue <$> (args !? #wds :: Maybe [Float]),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("num_weights",) . showValue <$>
                 (args !? #num_weights :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "multi_sgd_mom_update" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_multi_sgd_update" '(t, u) =
     '[ '("lrs", AttrReq [Float]), '("wds", AttrReq [Float]),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("data", AttrOpt [t u])]

_multi_sgd_update ::
                  forall a t u .
                    (Tensor t, Fullfilled "_multi_sgd_update" '(t, u) a, HasCallStack,
                     DType u) =>
                    ArgsHMap "_multi_sgd_update" '(t, u) a -> TensorApply (t u)
_multi_sgd_update args
  = let scalarArgs
          = catMaybes
              [("lrs",) . showValue <$> (args !? #lrs :: Maybe [Float]),
               ("wds",) . showValue <$> (args !? #wds :: Maybe [Float]),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("num_weights",) . showValue <$>
                 (args !? #num_weights :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "multi_sgd_update" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_multi_sum_sq" '(t, u) =
     '[ '("num_arrays", AttrReq Int), '("data", AttrOpt [t u])]

_multi_sum_sq ::
              forall a t u .
                (Tensor t, Fullfilled "_multi_sum_sq" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "_multi_sum_sq" '(t, u) a -> TensorApply (t u)
_multi_sum_sq args
  = let scalarArgs
          = catMaybes
              [("num_arrays",) . showValue <$>
                 (args !? #num_arrays :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "multi_sum_sq" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_nag_mom_update" '(t, u) =
     '[ '("lr", AttrReq Float), '("momentum", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u)), '("mom", AttrOpt (t u))]

_nag_mom_update ::
                forall a t u .
                  (Tensor t, Fullfilled "_nag_mom_update" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "_nag_mom_update" '(t, u) a -> TensorApply (t u)
_nag_mom_update args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("grad",) . toRaw <$> (args !? #grad :: Maybe (t u)),
               ("mom",) . toRaw <$> (args !? #mom :: Maybe (t u))]
      in
      applyRaw "nag_mom_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_nanprod" '(t, u) =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt (t u))]

_nanprod ::
         forall a t u .
           (Tensor t, Fullfilled "_nanprod" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "_nanprod" '(t, u) a -> TensorApply (t u)
_nanprod args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "nanprod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_nansum" '(t, u) =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt (t u))]

_nansum ::
        forall a t u .
          (Tensor t, Fullfilled "_nansum" '(t, u) a, HasCallStack,
           DType u) =>
          ArgsHMap "_nansum" '(t, u) a -> TensorApply (t u)
_nansum args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "nansum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_negative" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_negative ::
          forall a t u .
            (Tensor t, Fullfilled "_negative" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "_negative" '(t, u) a -> TensorApply (t u)
_negative args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "negative" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_norm" '(t, u) =
     '[ '("ord", AttrOpt Int), '("axis", AttrOpt (Maybe [Int])),
        '("out_dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["float16", "float32", "float64", "int32", "int64", "int8"]))),
        '("keepdims", AttrOpt Bool), '("data", AttrOpt (t u))]

_norm ::
      forall a t u .
        (Tensor t, Fullfilled "_norm" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "_norm" '(t, u) a -> TensorApply (t u)
_norm args
  = let scalarArgs
          = catMaybes
              [("ord",) . showValue <$> (args !? #ord :: Maybe Int),
               ("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("out_dtype",) . showValue <$>
                 (args !? #out_dtype ::
                    Maybe
                      (Maybe
                         (EnumType
                            '["float16", "float32", "float64", "int32", "int64", "int8"]))),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "norm" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_one_hot" '(t, u) =
     '[ '("depth", AttrReq Int), '("on_value", AttrOpt Double),
        '("off_value", AttrOpt Double),
        '("dtype",
          AttrOpt
            (EnumType
               '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"])),
        '("indices", AttrOpt (t u))]

_one_hot ::
         forall a t u .
           (Tensor t, Fullfilled "_one_hot" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "_one_hot" '(t, u) a -> TensorApply (t u)
_one_hot args
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
                         '["bfloat16", "float16", "float32", "float64", "int32", "int64",
                           "int8", "uint8"]))]
        tensorKeyArgs
          = catMaybes
              [("indices",) . toRaw <$> (args !? #indices :: Maybe (t u))]
      in
      applyRaw "one_hot" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_ones_like" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_ones_like ::
           forall a t u .
             (Tensor t, Fullfilled "_ones_like" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "_ones_like" '(t, u) a -> TensorApply (t u)
_ones_like args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "ones_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_pick" '(t, u) =
     '[ '("axis", AttrOpt (Maybe Int)), '("keepdims", AttrOpt Bool),
        '("mode", AttrOpt (EnumType '["clip", "wrap"])),
        '("data", AttrOpt (t u)), '("index", AttrOpt (t u))]

_pick ::
      forall a t u .
        (Tensor t, Fullfilled "_pick" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "_pick" '(t, u) a -> TensorApply (t u)
_pick args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("mode",) . showValue <$>
                 (args !? #mode :: Maybe (EnumType '["clip", "wrap"]))]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("index",) . toRaw <$> (args !? #index :: Maybe (t u))]
      in
      applyRaw "pick" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance
     ParameterList "_preloaded_multi_mp_sgd_mom_update" '(t, u) =
     '[ '("momentum", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("data", AttrOpt [t u])]

_preloaded_multi_mp_sgd_mom_update ::
                                   forall a t u .
                                     (Tensor t,
                                      Fullfilled "_preloaded_multi_mp_sgd_mom_update" '(t, u) a,
                                      HasCallStack, DType u) =>
                                     ArgsHMap "_preloaded_multi_mp_sgd_mom_update" '(t, u) a ->
                                       TensorApply (t u)
_preloaded_multi_mp_sgd_mom_update args
  = let scalarArgs
          = catMaybes
              [("momentum",) . showValue <$> (args !? #momentum :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("num_weights",) . showValue <$>
                 (args !? #num_weights :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in
      applyRaw "preloaded_multi_mp_sgd_mom_update" scalarArgs
        (Right tensorVarArgs)

type instance
     ParameterList "_preloaded_multi_mp_sgd_update" '(t, u) =
     '[ '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("data", AttrOpt [t u])]

_preloaded_multi_mp_sgd_update ::
                               forall a t u .
                                 (Tensor t, Fullfilled "_preloaded_multi_mp_sgd_update" '(t, u) a,
                                  HasCallStack, DType u) =>
                                 ArgsHMap "_preloaded_multi_mp_sgd_update" '(t, u) a ->
                                   TensorApply (t u)
_preloaded_multi_mp_sgd_update args
  = let scalarArgs
          = catMaybes
              [("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("num_weights",) . showValue <$>
                 (args !? #num_weights :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in
      applyRaw "preloaded_multi_mp_sgd_update" scalarArgs
        (Right tensorVarArgs)

type instance
     ParameterList "_preloaded_multi_sgd_mom_update" '(t, u) =
     '[ '("momentum", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("data", AttrOpt [t u])]

_preloaded_multi_sgd_mom_update ::
                                forall a t u .
                                  (Tensor t, Fullfilled "_preloaded_multi_sgd_mom_update" '(t, u) a,
                                   HasCallStack, DType u) =>
                                  ArgsHMap "_preloaded_multi_sgd_mom_update" '(t, u) a ->
                                    TensorApply (t u)
_preloaded_multi_sgd_mom_update args
  = let scalarArgs
          = catMaybes
              [("momentum",) . showValue <$> (args !? #momentum :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("num_weights",) . showValue <$>
                 (args !? #num_weights :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in
      applyRaw "preloaded_multi_sgd_mom_update" scalarArgs
        (Right tensorVarArgs)

type instance ParameterList "_preloaded_multi_sgd_update" '(t, u) =
     '[ '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("data", AttrOpt [t u])]

_preloaded_multi_sgd_update ::
                            forall a t u .
                              (Tensor t, Fullfilled "_preloaded_multi_sgd_update" '(t, u) a,
                               HasCallStack, DType u) =>
                              ArgsHMap "_preloaded_multi_sgd_update" '(t, u) a ->
                                TensorApply (t u)
_preloaded_multi_sgd_update args
  = let scalarArgs
          = catMaybes
              [("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float),
               ("num_weights",) . showValue <$>
                 (args !? #num_weights :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in
      applyRaw "preloaded_multi_sgd_update" scalarArgs
        (Right tensorVarArgs)

type instance ParameterList "_prod" '(t, u) =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt (t u))]

_prod ::
      forall a t u .
        (Tensor t, Fullfilled "_prod" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "_prod" '(t, u) a -> TensorApply (t u)
_prod args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "prod" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_radians" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_radians ::
         forall a t u .
           (Tensor t, Fullfilled "_radians" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "_radians" '(t, u) a -> TensorApply (t u)
_radians args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "radians" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_rcbrt" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_rcbrt ::
       forall a t u .
         (Tensor t, Fullfilled "_rcbrt" '(t, u) a, HasCallStack, DType u) =>
         ArgsHMap "_rcbrt" '(t, u) a -> TensorApply (t u)
_rcbrt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "rcbrt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_reciprocal" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_reciprocal ::
            forall a t u .
              (Tensor t, Fullfilled "_reciprocal" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "_reciprocal" '(t, u) a -> TensorApply (t u)
_reciprocal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "reciprocal" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_relu" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_relu ::
      forall a t u .
        (Tensor t, Fullfilled "_relu" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "_relu" '(t, u) a -> TensorApply (t u)
_relu args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "relu" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_repeat" '(t, u) =
     '[ '("repeats", AttrReq Int), '("axis", AttrOpt (Maybe Int)),
        '("data", AttrOpt (t u))]

_repeat ::
        forall a t u .
          (Tensor t, Fullfilled "_repeat" '(t, u) a, HasCallStack,
           DType u) =>
          ArgsHMap "_repeat" '(t, u) a -> TensorApply (t u)
_repeat args
  = let scalarArgs
          = catMaybes
              [("repeats",) . showValue <$> (args !? #repeats :: Maybe Int),
               ("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "repeat" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_reset_arrays" '(t, u) =
     '[ '("num_arrays", AttrReq Int), '("data", AttrOpt [t u])]

_reset_arrays ::
              forall a t u .
                (Tensor t, Fullfilled "_reset_arrays" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "_reset_arrays" '(t, u) a -> TensorApply (t u)
_reset_arrays args
  = let scalarArgs
          = catMaybes
              [("num_arrays",) . showValue <$>
                 (args !? #num_arrays :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "reset_arrays" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_reshape_like" '(t, u) =
     '[ '("lhs_begin", AttrOpt (Maybe Int)),
        '("lhs_end", AttrOpt (Maybe Int)),
        '("rhs_begin", AttrOpt (Maybe Int)),
        '("rhs_end", AttrOpt (Maybe Int)), '("lhs", AttrOpt (t u)),
        '("rhs", AttrOpt (t u))]

_reshape_like ::
              forall a t u .
                (Tensor t, Fullfilled "_reshape_like" '(t, u) a, HasCallStack,
                 DType u) =>
                ArgsHMap "_reshape_like" '(t, u) a -> TensorApply (t u)
_reshape_like args
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
        tensorKeyArgs
          = catMaybes
              [("lhs",) . toRaw <$> (args !? #lhs :: Maybe (t u)),
               ("rhs",) . toRaw <$> (args !? #rhs :: Maybe (t u))]
      in
      applyRaw "reshape_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_reverse" '(t, u) =
     '[ '("axis", AttrReq [Int]), '("data", AttrOpt (t u))]

_reverse ::
         forall a t u .
           (Tensor t, Fullfilled "_reverse" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "_reverse" '(t, u) a -> TensorApply (t u)
_reverse args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "reverse" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_rint" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_rint ::
      forall a t u .
        (Tensor t, Fullfilled "_rint" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "_rint" '(t, u) a -> TensorApply (t u)
_rint args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "rint" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_rmsprop_update" '(t, u) =
     '[ '("lr", AttrReq Float), '("gamma1", AttrOpt Float),
        '("epsilon", AttrOpt Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("clip_weights", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u)), '("n", AttrOpt (t u))]

_rmsprop_update ::
                forall a t u .
                  (Tensor t, Fullfilled "_rmsprop_update" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "_rmsprop_update" '(t, u) a -> TensorApply (t u)
_rmsprop_update args
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
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("grad",) . toRaw <$> (args !? #grad :: Maybe (t u)),
               ("n",) . toRaw <$> (args !? #n :: Maybe (t u))]
      in
      applyRaw "rmsprop_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_rmspropalex_update" '(t, u) =
     '[ '("lr", AttrReq Float), '("gamma1", AttrOpt Float),
        '("gamma2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("clip_weights", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u)), '("n", AttrOpt (t u)),
        '("g", AttrOpt (t u)), '("delta", AttrOpt (t u))]

_rmspropalex_update ::
                    forall a t u .
                      (Tensor t, Fullfilled "_rmspropalex_update" '(t, u) a,
                       HasCallStack, DType u) =>
                      ArgsHMap "_rmspropalex_update" '(t, u) a -> TensorApply (t u)
_rmspropalex_update args
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
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("grad",) . toRaw <$> (args !? #grad :: Maybe (t u)),
               ("n",) . toRaw <$> (args !? #n :: Maybe (t u)),
               ("g",) . toRaw <$> (args !? #g :: Maybe (t u)),
               ("delta",) . toRaw <$> (args !? #delta :: Maybe (t u))]
      in
      applyRaw "rmspropalex_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_round" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_round ::
       forall a t u .
         (Tensor t, Fullfilled "_round" '(t, u) a, HasCallStack, DType u) =>
         ArgsHMap "_round" '(t, u) a -> TensorApply (t u)
_round args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "round" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_rsqrt" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_rsqrt ::
       forall a t u .
         (Tensor t, Fullfilled "_rsqrt" '(t, u) a, HasCallStack, DType u) =>
         ArgsHMap "_rsqrt" '(t, u) a -> TensorApply (t u)
_rsqrt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "rsqrt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_scatter_nd" '(t, u) =
     '[ '("shape", AttrReq [Int]), '("data", AttrOpt (t u)),
        '("indices", AttrOpt (t u))]

_scatter_nd ::
            forall a t u .
              (Tensor t, Fullfilled "_scatter_nd" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "_scatter_nd" '(t, u) a -> TensorApply (t u)
_scatter_nd args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("indices",) . toRaw <$> (args !? #indices :: Maybe (t u))]
      in
      applyRaw "scatter_nd" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_sgd_mom_update" '(t, u) =
     '[ '("lr", AttrReq Float), '("momentum", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("lazy_update", AttrOpt Bool),
        '("weight", AttrOpt (t u)), '("grad", AttrOpt (t u)),
        '("mom", AttrOpt (t u))]

_sgd_mom_update ::
                forall a t u .
                  (Tensor t, Fullfilled "_sgd_mom_update" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "_sgd_mom_update" '(t, u) a -> TensorApply (t u)
_sgd_mom_update args
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
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("grad",) . toRaw <$> (args !? #grad :: Maybe (t u)),
               ("mom",) . toRaw <$> (args !? #mom :: Maybe (t u))]
      in
      applyRaw "sgd_mom_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_sgd_update" '(t, u) =
     '[ '("lr", AttrReq Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("lazy_update", AttrOpt Bool),
        '("weight", AttrOpt (t u)), '("grad", AttrOpt (t u))]

_sgd_update ::
            forall a t u .
              (Tensor t, Fullfilled "_sgd_update" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "_sgd_update" '(t, u) a -> TensorApply (t u)
_sgd_update args
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
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("grad",) . toRaw <$> (args !? #grad :: Maybe (t u))]
      in
      applyRaw "sgd_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_shape_array" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_shape_array ::
             forall a t u .
               (Tensor t, Fullfilled "_shape_array" '(t, u) a, HasCallStack,
                DType u) =>
               ArgsHMap "_shape_array" '(t, u) a -> TensorApply (t u)
_shape_array args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "shape_array" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_sigmoid" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_sigmoid ::
         forall a t u .
           (Tensor t, Fullfilled "_sigmoid" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "_sigmoid" '(t, u) a -> TensorApply (t u)
_sigmoid args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "sigmoid" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_sign" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_sign ::
      forall a t u .
        (Tensor t, Fullfilled "_sign" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "_sign" '(t, u) a -> TensorApply (t u)
_sign args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "sign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_signsgd_update" '(t, u) =
     '[ '("lr", AttrReq Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt (t u)),
        '("grad", AttrOpt (t u))]

_signsgd_update ::
                forall a t u .
                  (Tensor t, Fullfilled "_signsgd_update" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "_signsgd_update" '(t, u) a -> TensorApply (t u)
_signsgd_update args
  = let scalarArgs
          = catMaybes
              [("lr",) . showValue <$> (args !? #lr :: Maybe Float),
               ("wd",) . showValue <$> (args !? #wd :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float),
               ("clip_gradient",) . showValue <$>
                 (args !? #clip_gradient :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("grad",) . toRaw <$> (args !? #grad :: Maybe (t u))]
      in
      applyRaw "signsgd_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_signum_update" '(t, u) =
     '[ '("lr", AttrReq Float), '("momentum", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("wd_lh", AttrOpt Float),
        '("weight", AttrOpt (t u)), '("grad", AttrOpt (t u)),
        '("mom", AttrOpt (t u))]

_signum_update ::
               forall a t u .
                 (Tensor t, Fullfilled "_signum_update" '(t, u) a, HasCallStack,
                  DType u) =>
                 ArgsHMap "_signum_update" '(t, u) a -> TensorApply (t u)
_signum_update args
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
        tensorKeyArgs
          = catMaybes
              [("weight",) . toRaw <$> (args !? #weight :: Maybe (t u)),
               ("grad",) . toRaw <$> (args !? #grad :: Maybe (t u)),
               ("mom",) . toRaw <$> (args !? #mom :: Maybe (t u))]
      in
      applyRaw "signum_update" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_sin" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_sin ::
     forall a t u .
       (Tensor t, Fullfilled "_sin" '(t, u) a, HasCallStack, DType u) =>
       ArgsHMap "_sin" '(t, u) a -> TensorApply (t u)
_sin args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "sin" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_sinh" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_sinh ::
      forall a t u .
        (Tensor t, Fullfilled "_sinh" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "_sinh" '(t, u) a -> TensorApply (t u)
_sinh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "sinh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_size_array" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_size_array ::
            forall a t u .
              (Tensor t, Fullfilled "_size_array" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "_size_array" '(t, u) a -> TensorApply (t u)
_size_array args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "size_array" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_slice" '(t, u) =
     '[ '("begin", AttrReq [Int]), '("end", AttrReq [Int]),
        '("step", AttrOpt [Int]), '("data", AttrOpt (t u))]

_slice ::
       forall a t u .
         (Tensor t, Fullfilled "_slice" '(t, u) a, HasCallStack, DType u) =>
         ArgsHMap "_slice" '(t, u) a -> TensorApply (t u)
_slice args
  = let scalarArgs
          = catMaybes
              [("begin",) . showValue <$> (args !? #begin :: Maybe [Int]),
               ("end",) . showValue <$> (args !? #end :: Maybe [Int]),
               ("step",) . showValue <$> (args !? #step :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "slice" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_slice_axis" '(t, u) =
     '[ '("axis", AttrReq Int), '("begin", AttrReq Int),
        '("end", AttrReq (Maybe Int)), '("data", AttrOpt (t u))]

_slice_axis ::
            forall a t u .
              (Tensor t, Fullfilled "_slice_axis" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "_slice_axis" '(t, u) a -> TensorApply (t u)
_slice_axis args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("begin",) . showValue <$> (args !? #begin :: Maybe Int),
               ("end",) . showValue <$> (args !? #end :: Maybe (Maybe Int))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "slice_axis" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_slice_like" '(t, u) =
     '[ '("axes", AttrOpt [Int]), '("data", AttrOpt (t u)),
        '("shape_like", AttrOpt (t u))]

_slice_like ::
            forall a t u .
              (Tensor t, Fullfilled "_slice_like" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "_slice_like" '(t, u) a -> TensorApply (t u)
_slice_like args
  = let scalarArgs
          = catMaybes
              [("axes",) . showValue <$> (args !? #axes :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("shape_like",) . toRaw <$> (args !? #shape_like :: Maybe (t u))]
      in
      applyRaw "slice_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_smooth_l1" '(t, u) =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt (t u))]

_smooth_l1 ::
           forall a t u .
             (Tensor t, Fullfilled "_smooth_l1" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "_smooth_l1" '(t, u) a -> TensorApply (t u)
_smooth_l1 args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "smooth_l1" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_softmax" '(t, u) =
     '[ '("axis", AttrOpt Int),
        '("temperature", AttrOpt (Maybe Double)),
        '("dtype",
          AttrOpt (Maybe (EnumType '["float16", "float32", "float64"]))),
        '("use_length", AttrOpt (Maybe Bool)), '("data", AttrOpt (t u)),
        '("length", AttrOpt (t u))]

_softmax ::
         forall a t u .
           (Tensor t, Fullfilled "_softmax" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "_softmax" '(t, u) a -> TensorApply (t u)
_softmax args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("temperature",) . showValue <$>
                 (args !? #temperature :: Maybe (Maybe Double)),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (Maybe (EnumType '["float16", "float32", "float64"]))),
               ("use_length",) . showValue <$>
                 (args !? #use_length :: Maybe (Maybe Bool))]
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("length",) . toRaw <$> (args !? #length :: Maybe (t u))]
      in
      applyRaw "softmax" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_softmax_cross_entropy" '(t, u) =
     '[ '("data", AttrOpt (t u)), '("label", AttrOpt (t u))]

_softmax_cross_entropy ::
                       forall a t u .
                         (Tensor t, Fullfilled "_softmax_cross_entropy" '(t, u) a,
                          HasCallStack, DType u) =>
                         ArgsHMap "_softmax_cross_entropy" '(t, u) a -> TensorApply (t u)
_softmax_cross_entropy args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("data",) . toRaw <$> (args !? #data :: Maybe (t u)),
               ("label",) . toRaw <$> (args !? #label :: Maybe (t u))]
      in
      applyRaw "softmax_cross_entropy" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_softmin" '(t, u) =
     '[ '("axis", AttrOpt Int),
        '("temperature", AttrOpt (Maybe Double)),
        '("dtype",
          AttrOpt (Maybe (EnumType '["float16", "float32", "float64"]))),
        '("use_length", AttrOpt (Maybe Bool)), '("data", AttrOpt (t u))]

_softmin ::
         forall a t u .
           (Tensor t, Fullfilled "_softmin" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "_softmin" '(t, u) a -> TensorApply (t u)
_softmin args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("temperature",) . showValue <$>
                 (args !? #temperature :: Maybe (Maybe Double)),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (Maybe (EnumType '["float16", "float32", "float64"]))),
               ("use_length",) . showValue <$>
                 (args !? #use_length :: Maybe (Maybe Bool))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "softmin" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_softsign" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_softsign ::
          forall a t u .
            (Tensor t, Fullfilled "_softsign" '(t, u) a, HasCallStack,
             DType u) =>
            ArgsHMap "_softsign" '(t, u) a -> TensorApply (t u)
_softsign args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "softsign" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_sort" '(t, u) =
     '[ '("axis", AttrOpt (Maybe Int)), '("is_ascend", AttrOpt Bool),
        '("data", AttrOpt (t u))]

_sort ::
      forall a t u .
        (Tensor t, Fullfilled "_sort" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "_sort" '(t, u) a -> TensorApply (t u)
_sort args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("is_ascend",) . showValue <$> (args !? #is_ascend :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "sort" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_space_to_depth" '(t, u) =
     '[ '("block_size", AttrReq Int), '("data", AttrOpt (t u))]

_space_to_depth ::
                forall a t u .
                  (Tensor t, Fullfilled "_space_to_depth" '(t, u) a, HasCallStack,
                   DType u) =>
                  ArgsHMap "_space_to_depth" '(t, u) a -> TensorApply (t u)
_space_to_depth args
  = let scalarArgs
          = catMaybes
              [("block_size",) . showValue <$>
                 (args !? #block_size :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "space_to_depth" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_sqrt" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_sqrt ::
      forall a t u .
        (Tensor t, Fullfilled "_sqrt" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "_sqrt" '(t, u) a -> TensorApply (t u)
_sqrt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "sqrt" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_square" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_square ::
        forall a t u .
          (Tensor t, Fullfilled "_square" '(t, u) a, HasCallStack,
           DType u) =>
          ArgsHMap "_square" '(t, u) a -> TensorApply (t u)
_square args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "square" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_squeeze" '(t, u) =
     '[ '("axis", AttrOpt (Maybe [Int])), '("data", AttrOpt (t u))]

_squeeze ::
         forall a t u .
           (Tensor t, Fullfilled "_squeeze" '(t, u) a, HasCallStack,
            DType u) =>
           ArgsHMap "_squeeze" '(t, u) a -> TensorApply (t u)
_squeeze args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int]))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "squeeze" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_stack" '(t, u) =
     '[ '("axis", AttrOpt Int), '("num_args", AttrReq Int),
        '("data", AttrOpt [t u])]

_stack ::
       forall a t u .
         (Tensor t, Fullfilled "_stack" '(t, u) a, HasCallStack, DType u) =>
         ArgsHMap "_stack" '(t, u) a -> TensorApply (t u)
_stack args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("num_args",) . showValue <$> (args !? #num_args :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs
          = fromMaybe [] (map toRaw <$> args !? #data :: Maybe [RawTensor t])
      in applyRaw "stack" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_sum" '(t, u) =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt (t u))]

_sum ::
     forall a t u .
       (Tensor t, Fullfilled "_sum" '(t, u) a, HasCallStack, DType u) =>
       ArgsHMap "_sum" '(t, u) a -> TensorApply (t u)
_sum args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "sum" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_take" '(t, u) =
     '[ '("axis", AttrOpt Int),
        '("mode", AttrOpt (EnumType '["clip", "raise", "wrap"])),
        '("a", AttrOpt (t u)), '("indices", AttrOpt (t u))]

_take ::
      forall a t u .
        (Tensor t, Fullfilled "_take" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "_take" '(t, u) a -> TensorApply (t u)
_take args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("mode",) . showValue <$>
                 (args !? #mode :: Maybe (EnumType '["clip", "raise", "wrap"]))]
        tensorKeyArgs
          = catMaybes
              [("a",) . toRaw <$> (args !? #a :: Maybe (t u)),
               ("indices",) . toRaw <$> (args !? #indices :: Maybe (t u))]
      in
      applyRaw "take" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_tan" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_tan ::
     forall a t u .
       (Tensor t, Fullfilled "_tan" '(t, u) a, HasCallStack, DType u) =>
       ArgsHMap "_tan" '(t, u) a -> TensorApply (t u)
_tan args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "tan" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_tanh" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_tanh ::
      forall a t u .
        (Tensor t, Fullfilled "_tanh" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "_tanh" '(t, u) a -> TensorApply (t u)
_tanh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "tanh" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_tile" '(t, u) =
     '[ '("reps", AttrReq [Int]), '("data", AttrOpt (t u))]

_tile ::
      forall a t u .
        (Tensor t, Fullfilled "_tile" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "_tile" '(t, u) a -> TensorApply (t u)
_tile args
  = let scalarArgs
          = catMaybes
              [("reps",) . showValue <$> (args !? #reps :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "tile" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_topk" '(t, u) =
     '[ '("axis", AttrOpt (Maybe Int)), '("k", AttrOpt Int),
        '("ret_typ",
          AttrOpt (EnumType '["both", "indices", "mask", "value"])),
        '("is_ascend", AttrOpt Bool),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "uint8"])),
        '("data", AttrOpt (t u))]

_topk ::
      forall a t u .
        (Tensor t, Fullfilled "_topk" '(t, u) a, HasCallStack, DType u) =>
        ArgsHMap "_topk" '(t, u) a -> TensorApply (t u)
_topk args
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
                      (EnumType
                         '["float16", "float32", "float64", "int32", "int64", "uint8"]))]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "topk" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_transpose" '(t, u) =
     '[ '("axes", AttrOpt [Int]), '("data", AttrOpt (t u))]

_transpose ::
           forall a t u .
             (Tensor t, Fullfilled "_transpose" '(t, u) a, HasCallStack,
              DType u) =>
             ArgsHMap "_transpose" '(t, u) a -> TensorApply (t u)
_transpose args
  = let scalarArgs
          = catMaybes
              [("axes",) . showValue <$> (args !? #axes :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "transpose" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_trunc" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_trunc ::
       forall a t u .
         (Tensor t, Fullfilled "_trunc" '(t, u) a, HasCallStack, DType u) =>
         ArgsHMap "_trunc" '(t, u) a -> TensorApply (t u)
_trunc args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "trunc" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_where" '(t, u) =
     '[ '("condition", AttrOpt (t u)), '("x", AttrOpt (t u)),
        '("y", AttrOpt (t u))]

_where ::
       forall a t u .
         (Tensor t, Fullfilled "_where" '(t, u) a, HasCallStack, DType u) =>
         ArgsHMap "_where" '(t, u) a -> TensorApply (t u)
_where args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("condition",) . toRaw <$> (args !? #condition :: Maybe (t u)),
               ("x",) . toRaw <$> (args !? #x :: Maybe (t u)),
               ("y",) . toRaw <$> (args !? #y :: Maybe (t u))]
      in
      applyRaw "where" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))

type instance ParameterList "_zeros_like" '(t, u) =
     '[ '("data", AttrOpt (t u))]

_zeros_like ::
            forall a t u .
              (Tensor t, Fullfilled "_zeros_like" '(t, u) a, HasCallStack,
               DType u) =>
              ArgsHMap "_zeros_like" '(t, u) a -> TensorApply (t u)
_zeros_like args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) . toRaw <$> (args !? #data :: Maybe (t u))]
      in
      applyRaw "zeros_like" scalarArgs
        (Left (tensorKeyArgs :: [(Text, RawTensor t)]))