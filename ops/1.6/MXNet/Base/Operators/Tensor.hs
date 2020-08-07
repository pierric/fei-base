module MXNet.Base.Operators.Tensor where
import RIO
import RIO.List
import MXNet.Base.Raw
import MXNet.Base.Spec.Operator
import MXNet.Base.Spec.HMap
import MXNet.Base.Tensor
import Data.Maybe (catMaybes, fromMaybe)

type instance ParameterList "_Activation" t =
     '[ '("act_type",
          AttrReq
            (EnumType '["relu", "sigmoid", "softrelu", "softsign", "tanh"])),
        '("data", AttrOpt t)]

_Activation ::
            forall a t . (Tensor t, Fullfilled "_Activation" t a) =>
              ArgsHMap "_Activation" t a -> TensorApply t
_Activation args
  = let scalarArgs
          = catMaybes
              [("act_type",) . showValue <$>
                 (args !? #act_type ::
                    Maybe
                      (EnumType '["relu", "sigmoid", "softrelu", "softsign", "tanh"]))]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "Activation" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_BatchNorm" t =
     '[ '("eps", AttrOpt Double), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool), '("axis", AttrOpt Int),
        '("cudnn_off", AttrOpt Bool),
        '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)), '("data", AttrOpt t),
        '("gamma", AttrOpt t), '("beta", AttrOpt t),
        '("moving_mean", AttrOpt t), '("moving_var", AttrOpt t)]

_BatchNorm ::
           forall a t . (Tensor t, Fullfilled "_BatchNorm" t a) =>
             ArgsHMap "_BatchNorm" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("gamma",) <$> (args !? #gamma :: Maybe t),
               ("beta",) <$> (args !? #beta :: Maybe t),
               ("moving_mean",) <$> (args !? #moving_mean :: Maybe t),
               ("moving_var",) <$> (args !? #moving_var :: Maybe t)]
      in apply "BatchNorm" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_BatchNorm_v1" t =
     '[ '("eps", AttrOpt Float), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool), '("data", AttrOpt t),
        '("gamma", AttrOpt t), '("beta", AttrOpt t)]

_BatchNorm_v1 ::
              forall a t . (Tensor t, Fullfilled "_BatchNorm_v1" t a) =>
                ArgsHMap "_BatchNorm_v1" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("gamma",) <$> (args !? #gamma :: Maybe t),
               ("beta",) <$> (args !? #beta :: Maybe t)]
      in apply "BatchNorm_v1" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_BilinearSampler" t =
     '[ '("cudnn_off", AttrOpt (Maybe Bool)), '("data", AttrOpt t),
        '("grid", AttrOpt t)]

_BilinearSampler ::
                 forall a t . (Tensor t, Fullfilled "_BilinearSampler" t a) =>
                   ArgsHMap "_BilinearSampler" t a -> TensorApply t
_BilinearSampler args
  = let scalarArgs
          = catMaybes
              [("cudnn_off",) . showValue <$>
                 (args !? #cudnn_off :: Maybe (Maybe Bool))]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("grid",) <$> (args !? #grid :: Maybe t)]
      in apply "BilinearSampler" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_BlockGrad" t =
     '[ '("data", AttrOpt t)]

_BlockGrad ::
           forall a t . (Tensor t, Fullfilled "_BlockGrad" t a) =>
             ArgsHMap "_BlockGrad" t a -> TensorApply t
_BlockGrad args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "BlockGrad" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_CTCLoss" t =
     '[ '("use_data_lengths", AttrOpt Bool),
        '("use_label_lengths", AttrOpt Bool),
        '("blank_label", AttrOpt (EnumType '["first", "last"])),
        '("data", AttrOpt t), '("label", AttrOpt t),
        '("data_lengths", AttrOpt t), '("label_lengths", AttrOpt t)]

_CTCLoss ::
         forall a t . (Tensor t, Fullfilled "_CTCLoss" t a) =>
           ArgsHMap "_CTCLoss" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("label",) <$> (args !? #label :: Maybe t),
               ("data_lengths",) <$> (args !? #data_lengths :: Maybe t),
               ("label_lengths",) <$> (args !? #label_lengths :: Maybe t)]
      in apply "CTCLoss" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_Cast" t =
     '[ '("dtype",
          AttrReq
            (EnumType
               '["bool", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"])),
        '("data", AttrOpt t)]

_Cast ::
      forall a t . (Tensor t, Fullfilled "_Cast" t a) =>
        ArgsHMap "_Cast" t a -> TensorApply t
_Cast args
  = let scalarArgs
          = catMaybes
              [("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bool", "float16", "float32", "float64", "int32", "int64",
                           "int8", "uint8"]))]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "Cast" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_Concat" t =
     '[ '("num_args", AttrReq Int), '("dim", AttrOpt Int),
        '("data", AttrOpt [t])]

_Concat ::
        forall a t . (Tensor t, Fullfilled "_Concat" t a) =>
          ArgsHMap "_Concat" t a -> TensorApply t
_Concat args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("dim",) . showValue <$> (args !? #dim :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "Concat" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_Convolution" t =
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
        '("data", AttrOpt t), '("weight", AttrOpt t), '("bias", AttrOpt t)]

_Convolution ::
             forall a t . (Tensor t, Fullfilled "_Convolution" t a) =>
               ArgsHMap "_Convolution" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("weight",) <$> (args !? #weight :: Maybe t),
               ("bias",) <$> (args !? #bias :: Maybe t)]
      in apply "Convolution" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_Convolution_v1" t =
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
        '("data", AttrOpt t), '("weight", AttrOpt t), '("bias", AttrOpt t)]

_Convolution_v1 ::
                forall a t . (Tensor t, Fullfilled "_Convolution_v1" t a) =>
                  ArgsHMap "_Convolution_v1" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("weight",) <$> (args !? #weight :: Maybe t),
               ("bias",) <$> (args !? #bias :: Maybe t)]
      in apply "Convolution_v1" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_Correlation" t =
     '[ '("kernel_size", AttrOpt Int),
        '("max_displacement", AttrOpt Int), '("stride1", AttrOpt Int),
        '("stride2", AttrOpt Int), '("pad_size", AttrOpt Int),
        '("is_multiply", AttrOpt Bool), '("data1", AttrOpt t),
        '("data2", AttrOpt t)]

_Correlation ::
             forall a t . (Tensor t, Fullfilled "_Correlation" t a) =>
               ArgsHMap "_Correlation" t a -> TensorApply t
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
              [("data1",) <$> (args !? #data1 :: Maybe t),
               ("data2",) <$> (args !? #data2 :: Maybe t)]
      in apply "Correlation" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_Crop" t =
     '[ '("num_args", AttrReq Int), '("offset", AttrOpt [Int]),
        '("h_w", AttrOpt [Int]), '("center_crop", AttrOpt Bool),
        '("data", AttrOpt [t])]

_Crop ::
      forall a t . (Tensor t, Fullfilled "_Crop" t a) =>
        ArgsHMap "_Crop" t a -> TensorApply t
_Crop args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("offset",) . showValue <$> (args !? #offset :: Maybe [Int]),
               ("h_w",) . showValue <$> (args !? #h_w :: Maybe [Int]),
               ("center_crop",) . showValue <$>
                 (args !? #center_crop :: Maybe Bool)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "Crop" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_CuDNNBatchNorm" t =
     '[ '("eps", AttrOpt Double), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool), '("axis", AttrOpt Int),
        '("cudnn_off", AttrOpt Bool),
        '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)), '("data", AttrOpt t),
        '("gamma", AttrOpt t), '("beta", AttrOpt t),
        '("moving_mean", AttrOpt t), '("moving_var", AttrOpt t)]

_CuDNNBatchNorm ::
                forall a t . (Tensor t, Fullfilled "_CuDNNBatchNorm" t a) =>
                  ArgsHMap "_CuDNNBatchNorm" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("gamma",) <$> (args !? #gamma :: Maybe t),
               ("beta",) <$> (args !? #beta :: Maybe t),
               ("moving_mean",) <$> (args !? #moving_mean :: Maybe t),
               ("moving_var",) <$> (args !? #moving_var :: Maybe t)]
      in apply "CuDNNBatchNorm" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_Custom" t =
     '[ '("op_type", AttrOpt Text), '("data", AttrOpt [t])]

_Custom ::
        forall a t .
          (Tensor t, Fullfilled "_Custom" t a,
           PopKey (ArgOf "_Custom" t) a "data",
           Dump (PopResult (ArgOf "_Custom" t) a "data")) =>
          ArgsHMap "_Custom" t a -> TensorApply t
_Custom args
  = let scalarArgs = dump (pop args #data)
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "Custom" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_Deconvolution" t =
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
        '("data", AttrOpt t), '("weight", AttrOpt t), '("bias", AttrOpt t)]

_Deconvolution ::
               forall a t . (Tensor t, Fullfilled "_Deconvolution" t a) =>
                 ArgsHMap "_Deconvolution" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("weight",) <$> (args !? #weight :: Maybe t),
               ("bias",) <$> (args !? #bias :: Maybe t)]
      in apply "Deconvolution" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_Dropout" t =
     '[ '("p", AttrOpt Float),
        '("mode", AttrOpt (EnumType '["always", "training"])),
        '("axes", AttrOpt [Int]), '("cudnn_off", AttrOpt (Maybe Bool)),
        '("data", AttrOpt t)]

_Dropout ::
         forall a t . (Tensor t, Fullfilled "_Dropout" t a) =>
           ArgsHMap "_Dropout" t a -> TensorApply t
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
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "Dropout" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_Embedding" t =
     '[ '("input_dim", AttrReq Int), '("output_dim", AttrReq Int),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"])),
        '("sparse_grad", AttrOpt Bool), '("data", AttrOpt t),
        '("weight", AttrOpt t)]

_Embedding ::
           forall a t . (Tensor t, Fullfilled "_Embedding" t a) =>
             ArgsHMap "_Embedding" t a -> TensorApply t
_Embedding args
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
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("weight",) <$> (args !? #weight :: Maybe t)]
      in apply "Embedding" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_Flatten" t = '[ '("data", AttrOpt t)]

_Flatten ::
         forall a t . (Tensor t, Fullfilled "_Flatten" t a) =>
           ArgsHMap "_Flatten" t a -> TensorApply t
_Flatten args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "Flatten" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_FullyConnected" t =
     '[ '("num_hidden", AttrReq Int), '("no_bias", AttrOpt Bool),
        '("flatten", AttrOpt Bool), '("data", AttrOpt t),
        '("weight", AttrOpt t), '("bias", AttrOpt t)]

_FullyConnected ::
                forall a t . (Tensor t, Fullfilled "_FullyConnected" t a) =>
                  ArgsHMap "_FullyConnected" t a -> TensorApply t
_FullyConnected args
  = let scalarArgs
          = catMaybes
              [("num_hidden",) . showValue <$>
                 (args !? #num_hidden :: Maybe Int),
               ("no_bias",) . showValue <$> (args !? #no_bias :: Maybe Bool),
               ("flatten",) . showValue <$> (args !? #flatten :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("weight",) <$> (args !? #weight :: Maybe t),
               ("bias",) <$> (args !? #bias :: Maybe t)]
      in apply "FullyConnected" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_GridGenerator" t =
     '[ '("transform_type", AttrReq (EnumType '["affine", "warp"])),
        '("target_shape", AttrOpt [Int]), '("data", AttrOpt t)]

_GridGenerator ::
               forall a t . (Tensor t, Fullfilled "_GridGenerator" t a) =>
                 ArgsHMap "_GridGenerator" t a -> TensorApply t
_GridGenerator args
  = let scalarArgs
          = catMaybes
              [("transform_type",) . showValue <$>
                 (args !? #transform_type :: Maybe (EnumType '["affine", "warp"])),
               ("target_shape",) . showValue <$>
                 (args !? #target_shape :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "GridGenerator" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_GroupNorm" t =
     '[ '("num_groups", AttrOpt Int), '("eps", AttrOpt Float),
        '("output_mean_var", AttrOpt Bool), '("data", AttrOpt t),
        '("gamma", AttrOpt t), '("beta", AttrOpt t)]

_GroupNorm ::
           forall a t . (Tensor t, Fullfilled "_GroupNorm" t a) =>
             ArgsHMap "_GroupNorm" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("gamma",) <$> (args !? #gamma :: Maybe t),
               ("beta",) <$> (args !? #beta :: Maybe t)]
      in apply "GroupNorm" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_IdentityAttachKLSparseReg" t =
     '[ '("sparseness_target", AttrOpt Float),
        '("penalty", AttrOpt Float), '("momentum", AttrOpt Float),
        '("data", AttrOpt t)]

_IdentityAttachKLSparseReg ::
                           forall a t .
                             (Tensor t, Fullfilled "_IdentityAttachKLSparseReg" t a) =>
                             ArgsHMap "_IdentityAttachKLSparseReg" t a -> TensorApply t
_IdentityAttachKLSparseReg args
  = let scalarArgs
          = catMaybes
              [("sparseness_target",) . showValue <$>
                 (args !? #sparseness_target :: Maybe Float),
               ("penalty",) . showValue <$> (args !? #penalty :: Maybe Float),
               ("momentum",) . showValue <$> (args !? #momentum :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in
      apply "IdentityAttachKLSparseReg" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_InstanceNorm" t =
     '[ '("eps", AttrOpt Float), '("data", AttrOpt t),
        '("gamma", AttrOpt t), '("beta", AttrOpt t)]

_InstanceNorm ::
              forall a t . (Tensor t, Fullfilled "_InstanceNorm" t a) =>
                ArgsHMap "_InstanceNorm" t a -> TensorApply t
_InstanceNorm args
  = let scalarArgs
          = catMaybes
              [("eps",) . showValue <$> (args !? #eps :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("gamma",) <$> (args !? #gamma :: Maybe t),
               ("beta",) <$> (args !? #beta :: Maybe t)]
      in apply "InstanceNorm" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_L2Normalization" t =
     '[ '("eps", AttrOpt Float),
        '("mode", AttrOpt (EnumType '["channel", "instance", "spatial"])),
        '("data", AttrOpt t)]

_L2Normalization ::
                 forall a t . (Tensor t, Fullfilled "_L2Normalization" t a) =>
                   ArgsHMap "_L2Normalization" t a -> TensorApply t
_L2Normalization args
  = let scalarArgs
          = catMaybes
              [("eps",) . showValue <$> (args !? #eps :: Maybe Float),
               ("mode",) . showValue <$>
                 (args !? #mode ::
                    Maybe (EnumType '["channel", "instance", "spatial"]))]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "L2Normalization" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_LRN" t =
     '[ '("alpha", AttrOpt Float), '("beta", AttrOpt Float),
        '("knorm", AttrOpt Float), '("nsize", AttrReq Int),
        '("data", AttrOpt t)]

_LRN ::
     forall a t . (Tensor t, Fullfilled "_LRN" t a) =>
       ArgsHMap "_LRN" t a -> TensorApply t
_LRN args
  = let scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> (args !? #alpha :: Maybe Float),
               ("beta",) . showValue <$> (args !? #beta :: Maybe Float),
               ("knorm",) . showValue <$> (args !? #knorm :: Maybe Float),
               ("nsize",) . showValue <$> (args !? #nsize :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "LRN" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_LayerNorm" t =
     '[ '("axis", AttrOpt Int), '("eps", AttrOpt Float),
        '("output_mean_var", AttrOpt Bool), '("data", AttrOpt t),
        '("gamma", AttrOpt t), '("beta", AttrOpt t)]

_LayerNorm ::
           forall a t . (Tensor t, Fullfilled "_LayerNorm" t a) =>
             ArgsHMap "_LayerNorm" t a -> TensorApply t
_LayerNorm args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("eps",) . showValue <$> (args !? #eps :: Maybe Float),
               ("output_mean_var",) . showValue <$>
                 (args !? #output_mean_var :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("gamma",) <$> (args !? #gamma :: Maybe t),
               ("beta",) <$> (args !? #beta :: Maybe t)]
      in apply "LayerNorm" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_LeakyReLU" t =
     '[ '("act_type",
          AttrOpt
            (EnumType '["elu", "gelu", "leaky", "prelu", "rrelu", "selu"])),
        '("slope", AttrOpt Float), '("lower_bound", AttrOpt Float),
        '("upper_bound", AttrOpt Float), '("data", AttrOpt t),
        '("gamma", AttrOpt t)]

_LeakyReLU ::
           forall a t . (Tensor t, Fullfilled "_LeakyReLU" t a) =>
             ArgsHMap "_LeakyReLU" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("gamma",) <$> (args !? #gamma :: Maybe t)]
      in apply "LeakyReLU" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_LinearRegressionOutput" t =
     '[ '("grad_scale", AttrOpt Float), '("data", AttrOpt t),
        '("label", AttrOpt t)]

_LinearRegressionOutput ::
                        forall a t .
                          (Tensor t, Fullfilled "_LinearRegressionOutput" t a) =>
                          ArgsHMap "_LinearRegressionOutput" t a -> TensorApply t
_LinearRegressionOutput args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("label",) <$> (args !? #label :: Maybe t)]
      in apply "LinearRegressionOutput" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_LogisticRegressionOutput" t =
     '[ '("grad_scale", AttrOpt Float), '("data", AttrOpt t),
        '("label", AttrOpt t)]

_LogisticRegressionOutput ::
                          forall a t .
                            (Tensor t, Fullfilled "_LogisticRegressionOutput" t a) =>
                            ArgsHMap "_LogisticRegressionOutput" t a -> TensorApply t
_LogisticRegressionOutput args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("label",) <$> (args !? #label :: Maybe t)]
      in apply "LogisticRegressionOutput" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_MAERegressionOutput" t =
     '[ '("grad_scale", AttrOpt Float), '("data", AttrOpt t),
        '("label", AttrOpt t)]

_MAERegressionOutput ::
                     forall a t . (Tensor t, Fullfilled "_MAERegressionOutput" t a) =>
                       ArgsHMap "_MAERegressionOutput" t a -> TensorApply t
_MAERegressionOutput args
  = let scalarArgs
          = catMaybes
              [("grad_scale",) . showValue <$>
                 (args !? #grad_scale :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("label",) <$> (args !? #label :: Maybe t)]
      in apply "MAERegressionOutput" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_MakeLoss" t =
     '[ '("grad_scale", AttrOpt Float),
        '("valid_thresh", AttrOpt Float),
        '("normalization", AttrOpt (EnumType '["batch", "null", "valid"])),
        '("data", AttrOpt t)]

_MakeLoss ::
          forall a t . (Tensor t, Fullfilled "_MakeLoss" t a) =>
            ArgsHMap "_MakeLoss" t a -> TensorApply t
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
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "MakeLoss" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_Pad" t =
     '[ '("mode", AttrReq (EnumType '["constant", "edge", "reflect"])),
        '("pad_width", AttrReq [Int]), '("constant_value", AttrOpt Double),
        '("data", AttrOpt t)]

_Pad ::
     forall a t . (Tensor t, Fullfilled "_Pad" t a) =>
       ArgsHMap "_Pad" t a -> TensorApply t
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
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "Pad" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_Pooling" t =
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
        '("data", AttrOpt t)]

_Pooling ::
         forall a t . (Tensor t, Fullfilled "_Pooling" t a) =>
           ArgsHMap "_Pooling" t a -> TensorApply t
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
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "Pooling" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_Pooling_v1" t =
     '[ '("kernel", AttrOpt [Int]),
        '("pool_type", AttrOpt (EnumType '["avg", "max", "sum"])),
        '("global_pool", AttrOpt Bool),
        '("pooling_convention", AttrOpt (EnumType '["full", "valid"])),
        '("stride", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("data", AttrOpt t)]

_Pooling_v1 ::
            forall a t . (Tensor t, Fullfilled "_Pooling_v1" t a) =>
              ArgsHMap "_Pooling_v1" t a -> TensorApply t
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
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "Pooling_v1" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_RNN" t =
     '[ '("state_size", AttrReq Int), '("num_layers", AttrReq Int),
        '("bidirectional", AttrOpt Bool),
        '("mode",
          AttrReq (EnumType '["gru", "lstm", "rnn_relu", "rnn_tanh"])),
        '("p", AttrOpt Float), '("state_outputs", AttrOpt Bool),
        '("projection_size", AttrOpt (Maybe Int)),
        '("lstm_state_clip_min", AttrOpt (Maybe Double)),
        '("lstm_state_clip_max", AttrOpt (Maybe Double)),
        '("lstm_state_clip_nan", AttrOpt Bool),
        '("use_sequence_length", AttrOpt Bool), '("data", AttrOpt t),
        '("parameters", AttrOpt t), '("state", AttrOpt t),
        '("state_cell", AttrOpt t), '("sequence_length", AttrOpt t)]

_RNN ::
     forall a t . (Tensor t, Fullfilled "_RNN" t a) =>
       ArgsHMap "_RNN" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("parameters",) <$> (args !? #parameters :: Maybe t),
               ("state",) <$> (args !? #state :: Maybe t),
               ("state_cell",) <$> (args !? #state_cell :: Maybe t),
               ("sequence_length",) <$> (args !? #sequence_length :: Maybe t)]
      in apply "RNN" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_ROIPooling" t =
     '[ '("pooled_size", AttrReq [Int]),
        '("spatial_scale", AttrReq Float), '("data", AttrOpt t),
        '("rois", AttrOpt t)]

_ROIPooling ::
            forall a t . (Tensor t, Fullfilled "_ROIPooling" t a) =>
              ArgsHMap "_ROIPooling" t a -> TensorApply t
_ROIPooling args
  = let scalarArgs
          = catMaybes
              [("pooled_size",) . showValue <$>
                 (args !? #pooled_size :: Maybe [Int]),
               ("spatial_scale",) . showValue <$>
                 (args !? #spatial_scale :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("rois",) <$> (args !? #rois :: Maybe t)]
      in apply "ROIPooling" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_Reshape" t =
     '[ '("shape", AttrOpt [Int]), '("reverse", AttrOpt Bool),
        '("target_shape", AttrOpt [Int]), '("keep_highest", AttrOpt Bool),
        '("data", AttrOpt t)]

_Reshape ::
         forall a t . (Tensor t, Fullfilled "_Reshape" t a) =>
           ArgsHMap "_Reshape" t a -> TensorApply t
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
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "Reshape" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_SVMOutput" t =
     '[ '("margin", AttrOpt Float),
        '("regularization_coefficient", AttrOpt Float),
        '("use_linear", AttrOpt Bool), '("data", AttrOpt t),
        '("label", AttrOpt t)]

_SVMOutput ::
           forall a t . (Tensor t, Fullfilled "_SVMOutput" t a) =>
             ArgsHMap "_SVMOutput" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("label",) <$> (args !? #label :: Maybe t)]
      in apply "SVMOutput" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_SequenceLast" t =
     '[ '("use_sequence_length", AttrOpt Bool), '("axis", AttrOpt Int),
        '("data", AttrOpt t), '("sequence_length", AttrOpt t)]

_SequenceLast ::
              forall a t . (Tensor t, Fullfilled "_SequenceLast" t a) =>
                ArgsHMap "_SequenceLast" t a -> TensorApply t
_SequenceLast args
  = let scalarArgs
          = catMaybes
              [("use_sequence_length",) . showValue <$>
                 (args !? #use_sequence_length :: Maybe Bool),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("sequence_length",) <$> (args !? #sequence_length :: Maybe t)]
      in apply "SequenceLast" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_SequenceMask" t =
     '[ '("use_sequence_length", AttrOpt Bool),
        '("value", AttrOpt Float), '("axis", AttrOpt Int),
        '("data", AttrOpt t), '("sequence_length", AttrOpt t)]

_SequenceMask ::
              forall a t . (Tensor t, Fullfilled "_SequenceMask" t a) =>
                ArgsHMap "_SequenceMask" t a -> TensorApply t
_SequenceMask args
  = let scalarArgs
          = catMaybes
              [("use_sequence_length",) . showValue <$>
                 (args !? #use_sequence_length :: Maybe Bool),
               ("value",) . showValue <$> (args !? #value :: Maybe Float),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("sequence_length",) <$> (args !? #sequence_length :: Maybe t)]
      in apply "SequenceMask" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_SequenceReverse" t =
     '[ '("use_sequence_length", AttrOpt Bool), '("axis", AttrOpt Int),
        '("data", AttrOpt t), '("sequence_length", AttrOpt t)]

_SequenceReverse ::
                 forall a t . (Tensor t, Fullfilled "_SequenceReverse" t a) =>
                   ArgsHMap "_SequenceReverse" t a -> TensorApply t
_SequenceReverse args
  = let scalarArgs
          = catMaybes
              [("use_sequence_length",) . showValue <$>
                 (args !? #use_sequence_length :: Maybe Bool),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("sequence_length",) <$> (args !? #sequence_length :: Maybe t)]
      in apply "SequenceReverse" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_SliceChannel" t =
     '[ '("num_outputs", AttrReq Int), '("axis", AttrOpt Int),
        '("squeeze_axis", AttrOpt Bool), '("data", AttrOpt t)]

_SliceChannel ::
              forall a t . (Tensor t, Fullfilled "_SliceChannel" t a) =>
                ArgsHMap "_SliceChannel" t a -> TensorApply t
_SliceChannel args
  = let scalarArgs
          = catMaybes
              [("num_outputs",) . showValue <$>
                 (args !? #num_outputs :: Maybe Int),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("squeeze_axis",) . showValue <$>
                 (args !? #squeeze_axis :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "SliceChannel" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_SoftmaxActivation" t =
     '[ '("mode", AttrOpt (EnumType '["channel", "instance"])),
        '("data", AttrOpt t)]

_SoftmaxActivation ::
                   forall a t . (Tensor t, Fullfilled "_SoftmaxActivation" t a) =>
                     ArgsHMap "_SoftmaxActivation" t a -> TensorApply t
_SoftmaxActivation args
  = let scalarArgs
          = catMaybes
              [("mode",) . showValue <$>
                 (args !? #mode :: Maybe (EnumType '["channel", "instance"]))]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "SoftmaxActivation" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_SoftmaxOutput" t =
     '[ '("grad_scale", AttrOpt Float),
        '("ignore_label", AttrOpt Float), '("multi_output", AttrOpt Bool),
        '("use_ignore", AttrOpt Bool), '("preserve_shape", AttrOpt Bool),
        '("normalization", AttrOpt (EnumType '["batch", "null", "valid"])),
        '("out_grad", AttrOpt Bool), '("smooth_alpha", AttrOpt Float),
        '("data", AttrOpt t), '("label", AttrOpt t)]

_SoftmaxOutput ::
               forall a t . (Tensor t, Fullfilled "_SoftmaxOutput" t a) =>
                 ArgsHMap "_SoftmaxOutput" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("label",) <$> (args !? #label :: Maybe t)]
      in apply "SoftmaxOutput" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_SpatialTransformer" t =
     '[ '("target_shape", AttrOpt [Int]),
        '("transform_type", AttrReq (EnumType '["affine"])),
        '("sampler_type", AttrReq (EnumType '["bilinear"])),
        '("cudnn_off", AttrOpt (Maybe Bool)), '("data", AttrOpt t),
        '("loc", AttrOpt t)]

_SpatialTransformer ::
                    forall a t . (Tensor t, Fullfilled "_SpatialTransformer" t a) =>
                      ArgsHMap "_SpatialTransformer" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("loc",) <$> (args !? #loc :: Maybe t)]
      in apply "SpatialTransformer" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_SwapAxis" t =
     '[ '("dim1", AttrOpt Int), '("dim2", AttrOpt Int),
        '("data", AttrOpt t)]

_SwapAxis ::
          forall a t . (Tensor t, Fullfilled "_SwapAxis" t a) =>
            ArgsHMap "_SwapAxis" t a -> TensorApply t
_SwapAxis args
  = let scalarArgs
          = catMaybes
              [("dim1",) . showValue <$> (args !? #dim1 :: Maybe Int),
               ("dim2",) . showValue <$> (args !? #dim2 :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "SwapAxis" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_UpSampling" t =
     '[ '("scale", AttrReq Int), '("num_filter", AttrOpt Int),
        '("sample_type", AttrReq (EnumType '["bilinear", "nearest"])),
        '("multi_input_mode", AttrOpt (EnumType '["concat", "sum"])),
        '("num_args", AttrReq Int), '("workspace", AttrOpt Int),
        '("data", AttrOpt [t])]

_UpSampling ::
            forall a t . (Tensor t, Fullfilled "_UpSampling" t a) =>
              ArgsHMap "_UpSampling" t a -> TensorApply t
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
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "UpSampling" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__CachedOp" t =
     '[ '("data", AttrOpt [t])]

__CachedOp ::
           forall a t . (Tensor t, Fullfilled "__CachedOp" t a) =>
             ArgsHMap "__CachedOp" t a -> TensorApply t
__CachedOp args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "_CachedOp" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__CrossDeviceCopy" t = '[]

__CrossDeviceCopy ::
                  forall a t . (Tensor t, Fullfilled "__CrossDeviceCopy" t a) =>
                    ArgsHMap "__CrossDeviceCopy" t a -> TensorApply t
__CrossDeviceCopy args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_CrossDeviceCopy" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__CustomFunction" t = '[]

__CustomFunction ::
                 forall a t . (Tensor t, Fullfilled "__CustomFunction" t a) =>
                   ArgsHMap "__CustomFunction" t a -> TensorApply t
__CustomFunction args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_CustomFunction" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__FusedOp" t =
     '[ '("data", AttrOpt [t])]

__FusedOp ::
          forall a t . (Tensor t, Fullfilled "__FusedOp" t a) =>
            ArgsHMap "__FusedOp" t a -> TensorApply t
__FusedOp args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "_FusedOp" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__FusedOpHelper" t = '[]

__FusedOpHelper ::
                forall a t . (Tensor t, Fullfilled "__FusedOpHelper" t a) =>
                  ArgsHMap "__FusedOpHelper" t a -> TensorApply t
__FusedOpHelper args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_FusedOpHelper" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__FusedOpOutHelper" t = '[]

__FusedOpOutHelper ::
                   forall a t . (Tensor t, Fullfilled "__FusedOpOutHelper" t a) =>
                     ArgsHMap "__FusedOpOutHelper" t a -> TensorApply t
__FusedOpOutHelper args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_FusedOpOutHelper" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__NoGradient" t = '[]

__NoGradient ::
             forall a t . (Tensor t, Fullfilled "__NoGradient" t a) =>
               ArgsHMap "__NoGradient" t a -> TensorApply t
__NoGradient args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_NoGradient" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__adamw_update" t =
     '[ '("lr", AttrReq Float), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wd", AttrOpt Float), '("eta", AttrReq Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt t),
        '("grad", AttrOpt t), '("mean", AttrOpt t), '("var", AttrOpt t),
        '("rescale_grad", AttrOpt t)]

__adamw_update ::
               forall a t . (Tensor t, Fullfilled "__adamw_update" t a) =>
                 ArgsHMap "__adamw_update" t a -> TensorApply t
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
              [("weight",) <$> (args !? #weight :: Maybe t),
               ("grad",) <$> (args !? #grad :: Maybe t),
               ("mean",) <$> (args !? #mean :: Maybe t),
               ("var",) <$> (args !? #var :: Maybe t),
               ("rescale_grad",) <$> (args !? #rescale_grad :: Maybe t)]
      in apply "_adamw_update" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__arange" t =
     '[ '("start", AttrReq Double), '("stop", AttrOpt (Maybe Double)),
        '("step", AttrOpt Double), '("repeat", AttrOpt Int),
        '("infer_range", AttrOpt Bool), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"]))]

__arange ::
         forall a t . (Tensor t, Fullfilled "__arange" t a) =>
           ArgsHMap "__arange" t a -> TensorApply t
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
                         '["float16", "float32", "float64", "int32", "int64", "int8",
                           "uint8"]))]
        tensorKeyArgs = catMaybes []
      in apply "_arange" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_Activation" t = '[]

__backward_Activation ::
                      forall a t . (Tensor t, Fullfilled "__backward_Activation" t a) =>
                        ArgsHMap "__backward_Activation" t a -> TensorApply t
__backward_Activation args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_Activation" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_BatchNorm" t = '[]

__backward_BatchNorm ::
                     forall a t . (Tensor t, Fullfilled "__backward_BatchNorm" t a) =>
                       ArgsHMap "__backward_BatchNorm" t a -> TensorApply t
__backward_BatchNorm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_BatchNorm" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_BatchNorm_v1" t = '[]

__backward_BatchNorm_v1 ::
                        forall a t .
                          (Tensor t, Fullfilled "__backward_BatchNorm_v1" t a) =>
                          ArgsHMap "__backward_BatchNorm_v1" t a -> TensorApply t
__backward_BatchNorm_v1 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_BatchNorm_v1" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_BilinearSampler" t = '[]

__backward_BilinearSampler ::
                           forall a t .
                             (Tensor t, Fullfilled "__backward_BilinearSampler" t a) =>
                             ArgsHMap "__backward_BilinearSampler" t a -> TensorApply t
__backward_BilinearSampler args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_BilinearSampler" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_CachedOp" t = '[]

__backward_CachedOp ::
                    forall a t . (Tensor t, Fullfilled "__backward_CachedOp" t a) =>
                      ArgsHMap "__backward_CachedOp" t a -> TensorApply t
__backward_CachedOp args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_CachedOp" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_Concat" t = '[]

__backward_Concat ::
                  forall a t . (Tensor t, Fullfilled "__backward_Concat" t a) =>
                    ArgsHMap "__backward_Concat" t a -> TensorApply t
__backward_Concat args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_Concat" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_Convolution" t = '[]

__backward_Convolution ::
                       forall a t . (Tensor t, Fullfilled "__backward_Convolution" t a) =>
                         ArgsHMap "__backward_Convolution" t a -> TensorApply t
__backward_Convolution args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_Convolution" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_Convolution_v1" t = '[]

__backward_Convolution_v1 ::
                          forall a t .
                            (Tensor t, Fullfilled "__backward_Convolution_v1" t a) =>
                            ArgsHMap "__backward_Convolution_v1" t a -> TensorApply t
__backward_Convolution_v1 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_Convolution_v1" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_Correlation" t = '[]

__backward_Correlation ::
                       forall a t . (Tensor t, Fullfilled "__backward_Correlation" t a) =>
                         ArgsHMap "__backward_Correlation" t a -> TensorApply t
__backward_Correlation args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_Correlation" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_Crop" t = '[]

__backward_Crop ::
                forall a t . (Tensor t, Fullfilled "__backward_Crop" t a) =>
                  ArgsHMap "__backward_Crop" t a -> TensorApply t
__backward_Crop args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_Crop" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_CuDNNBatchNorm" t = '[]

__backward_CuDNNBatchNorm ::
                          forall a t .
                            (Tensor t, Fullfilled "__backward_CuDNNBatchNorm" t a) =>
                            ArgsHMap "__backward_CuDNNBatchNorm" t a -> TensorApply t
__backward_CuDNNBatchNorm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_CuDNNBatchNorm" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_Custom" t = '[]

__backward_Custom ::
                  forall a t . (Tensor t, Fullfilled "__backward_Custom" t a) =>
                    ArgsHMap "__backward_Custom" t a -> TensorApply t
__backward_Custom args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_Custom" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_CustomFunction" t = '[]

__backward_CustomFunction ::
                          forall a t .
                            (Tensor t, Fullfilled "__backward_CustomFunction" t a) =>
                            ArgsHMap "__backward_CustomFunction" t a -> TensorApply t
__backward_CustomFunction args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_CustomFunction" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_Deconvolution" t = '[]

__backward_Deconvolution ::
                         forall a t .
                           (Tensor t, Fullfilled "__backward_Deconvolution" t a) =>
                           ArgsHMap "__backward_Deconvolution" t a -> TensorApply t
__backward_Deconvolution args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_Deconvolution" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_Dropout" t = '[]

__backward_Dropout ::
                   forall a t . (Tensor t, Fullfilled "__backward_Dropout" t a) =>
                     ArgsHMap "__backward_Dropout" t a -> TensorApply t
__backward_Dropout args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_Dropout" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_Embedding" t = '[]

__backward_Embedding ::
                     forall a t . (Tensor t, Fullfilled "__backward_Embedding" t a) =>
                       ArgsHMap "__backward_Embedding" t a -> TensorApply t
__backward_Embedding args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_Embedding" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_FullyConnected" t = '[]

__backward_FullyConnected ::
                          forall a t .
                            (Tensor t, Fullfilled "__backward_FullyConnected" t a) =>
                            ArgsHMap "__backward_FullyConnected" t a -> TensorApply t
__backward_FullyConnected args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_FullyConnected" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_GridGenerator" t = '[]

__backward_GridGenerator ::
                         forall a t .
                           (Tensor t, Fullfilled "__backward_GridGenerator" t a) =>
                           ArgsHMap "__backward_GridGenerator" t a -> TensorApply t
__backward_GridGenerator args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_GridGenerator" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_GroupNorm" t = '[]

__backward_GroupNorm ::
                     forall a t . (Tensor t, Fullfilled "__backward_GroupNorm" t a) =>
                       ArgsHMap "__backward_GroupNorm" t a -> TensorApply t
__backward_GroupNorm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_GroupNorm" scalarArgs (Left tensorKeyArgs)

type instance
     ParameterList "__backward_IdentityAttachKLSparseReg" t = '[]

__backward_IdentityAttachKLSparseReg ::
                                     forall a t .
                                       (Tensor t,
                                        Fullfilled "__backward_IdentityAttachKLSparseReg" t a) =>
                                       ArgsHMap "__backward_IdentityAttachKLSparseReg" t a ->
                                         TensorApply t
__backward_IdentityAttachKLSparseReg args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_IdentityAttachKLSparseReg" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward_InstanceNorm" t = '[]

__backward_InstanceNorm ::
                        forall a t .
                          (Tensor t, Fullfilled "__backward_InstanceNorm" t a) =>
                          ArgsHMap "__backward_InstanceNorm" t a -> TensorApply t
__backward_InstanceNorm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_InstanceNorm" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_L2Normalization" t = '[]

__backward_L2Normalization ::
                           forall a t .
                             (Tensor t, Fullfilled "__backward_L2Normalization" t a) =>
                             ArgsHMap "__backward_L2Normalization" t a -> TensorApply t
__backward_L2Normalization args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_L2Normalization" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_LRN" t = '[]

__backward_LRN ::
               forall a t . (Tensor t, Fullfilled "__backward_LRN" t a) =>
                 ArgsHMap "__backward_LRN" t a -> TensorApply t
__backward_LRN args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_LRN" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_LayerNorm" t = '[]

__backward_LayerNorm ::
                     forall a t . (Tensor t, Fullfilled "__backward_LayerNorm" t a) =>
                       ArgsHMap "__backward_LayerNorm" t a -> TensorApply t
__backward_LayerNorm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_LayerNorm" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_LeakyReLU" t = '[]

__backward_LeakyReLU ::
                     forall a t . (Tensor t, Fullfilled "__backward_LeakyReLU" t a) =>
                       ArgsHMap "__backward_LeakyReLU" t a -> TensorApply t
__backward_LeakyReLU args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_LeakyReLU" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_MakeLoss" t = '[]

__backward_MakeLoss ::
                    forall a t . (Tensor t, Fullfilled "__backward_MakeLoss" t a) =>
                      ArgsHMap "__backward_MakeLoss" t a -> TensorApply t
__backward_MakeLoss args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_MakeLoss" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_Pad" t = '[]

__backward_Pad ::
               forall a t . (Tensor t, Fullfilled "__backward_Pad" t a) =>
                 ArgsHMap "__backward_Pad" t a -> TensorApply t
__backward_Pad args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_Pad" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_Pooling" t = '[]

__backward_Pooling ::
                   forall a t . (Tensor t, Fullfilled "__backward_Pooling" t a) =>
                     ArgsHMap "__backward_Pooling" t a -> TensorApply t
__backward_Pooling args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_Pooling" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_Pooling_v1" t = '[]

__backward_Pooling_v1 ::
                      forall a t . (Tensor t, Fullfilled "__backward_Pooling_v1" t a) =>
                        ArgsHMap "__backward_Pooling_v1" t a -> TensorApply t
__backward_Pooling_v1 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_Pooling_v1" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_RNN" t = '[]

__backward_RNN ::
               forall a t . (Tensor t, Fullfilled "__backward_RNN" t a) =>
                 ArgsHMap "__backward_RNN" t a -> TensorApply t
__backward_RNN args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_RNN" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_ROIAlign" t = '[]

__backward_ROIAlign ::
                    forall a t . (Tensor t, Fullfilled "__backward_ROIAlign" t a) =>
                      ArgsHMap "__backward_ROIAlign" t a -> TensorApply t
__backward_ROIAlign args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_ROIAlign" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_ROIPooling" t = '[]

__backward_ROIPooling ::
                      forall a t . (Tensor t, Fullfilled "__backward_ROIPooling" t a) =>
                        ArgsHMap "__backward_ROIPooling" t a -> TensorApply t
__backward_ROIPooling args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_ROIPooling" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_RROIAlign" t = '[]

__backward_RROIAlign ::
                     forall a t . (Tensor t, Fullfilled "__backward_RROIAlign" t a) =>
                       ArgsHMap "__backward_RROIAlign" t a -> TensorApply t
__backward_RROIAlign args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_RROIAlign" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_SVMOutput" t = '[]

__backward_SVMOutput ::
                     forall a t . (Tensor t, Fullfilled "__backward_SVMOutput" t a) =>
                       ArgsHMap "__backward_SVMOutput" t a -> TensorApply t
__backward_SVMOutput args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_SVMOutput" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_SequenceLast" t = '[]

__backward_SequenceLast ::
                        forall a t .
                          (Tensor t, Fullfilled "__backward_SequenceLast" t a) =>
                          ArgsHMap "__backward_SequenceLast" t a -> TensorApply t
__backward_SequenceLast args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_SequenceLast" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_SequenceMask" t = '[]

__backward_SequenceMask ::
                        forall a t .
                          (Tensor t, Fullfilled "__backward_SequenceMask" t a) =>
                          ArgsHMap "__backward_SequenceMask" t a -> TensorApply t
__backward_SequenceMask args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_SequenceMask" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_SequenceReverse" t = '[]

__backward_SequenceReverse ::
                           forall a t .
                             (Tensor t, Fullfilled "__backward_SequenceReverse" t a) =>
                             ArgsHMap "__backward_SequenceReverse" t a -> TensorApply t
__backward_SequenceReverse args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_SequenceReverse" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_SliceChannel" t = '[]

__backward_SliceChannel ::
                        forall a t .
                          (Tensor t, Fullfilled "__backward_SliceChannel" t a) =>
                          ArgsHMap "__backward_SliceChannel" t a -> TensorApply t
__backward_SliceChannel args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_SliceChannel" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_SoftmaxActivation" t = '[]

__backward_SoftmaxActivation ::
                             forall a t .
                               (Tensor t, Fullfilled "__backward_SoftmaxActivation" t a) =>
                               ArgsHMap "__backward_SoftmaxActivation" t a -> TensorApply t
__backward_SoftmaxActivation args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_SoftmaxActivation" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_SoftmaxOutput" t = '[]

__backward_SoftmaxOutput ::
                         forall a t .
                           (Tensor t, Fullfilled "__backward_SoftmaxOutput" t a) =>
                           ArgsHMap "__backward_SoftmaxOutput" t a -> TensorApply t
__backward_SoftmaxOutput args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_SoftmaxOutput" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_SparseEmbedding" t = '[]

__backward_SparseEmbedding ::
                           forall a t .
                             (Tensor t, Fullfilled "__backward_SparseEmbedding" t a) =>
                             ArgsHMap "__backward_SparseEmbedding" t a -> TensorApply t
__backward_SparseEmbedding args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_SparseEmbedding" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_SpatialTransformer" t = '[]

__backward_SpatialTransformer ::
                              forall a t .
                                (Tensor t, Fullfilled "__backward_SpatialTransformer" t a) =>
                                ArgsHMap "__backward_SpatialTransformer" t a -> TensorApply t
__backward_SpatialTransformer args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_SpatialTransformer" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward_SwapAxis" t = '[]

__backward_SwapAxis ::
                    forall a t . (Tensor t, Fullfilled "__backward_SwapAxis" t a) =>
                      ArgsHMap "__backward_SwapAxis" t a -> TensorApply t
__backward_SwapAxis args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_SwapAxis" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_UpSampling" t = '[]

__backward_UpSampling ::
                      forall a t . (Tensor t, Fullfilled "__backward_UpSampling" t a) =>
                        ArgsHMap "__backward_UpSampling" t a -> TensorApply t
__backward_UpSampling args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_UpSampling" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward__CrossDeviceCopy" t = '[]

__backward__CrossDeviceCopy ::
                            forall a t .
                              (Tensor t, Fullfilled "__backward__CrossDeviceCopy" t a) =>
                              ArgsHMap "__backward__CrossDeviceCopy" t a -> TensorApply t
__backward__CrossDeviceCopy args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward__CrossDeviceCopy" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward__NDArray" t = '[]

__backward__NDArray ::
                    forall a t . (Tensor t, Fullfilled "__backward__NDArray" t a) =>
                      ArgsHMap "__backward__NDArray" t a -> TensorApply t
__backward__NDArray args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward__NDArray" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward__Native" t = '[]

__backward__Native ::
                   forall a t . (Tensor t, Fullfilled "__backward__Native" t a) =>
                     ArgsHMap "__backward__Native" t a -> TensorApply t
__backward__Native args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward__Native" scalarArgs (Left tensorKeyArgs)

type instance
     ParameterList "__backward__contrib_DeformableConvolution" t = '[]

__backward__contrib_DeformableConvolution ::
                                          forall a t .
                                            (Tensor t,
                                             Fullfilled "__backward__contrib_DeformableConvolution"
                                               t a) =>
                                            ArgsHMap "__backward__contrib_DeformableConvolution" t a
                                              -> TensorApply t
__backward__contrib_DeformableConvolution args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward__contrib_DeformableConvolution" scalarArgs
        (Left tensorKeyArgs)

type instance
     ParameterList "__backward__contrib_DeformablePSROIPooling" t = '[]

__backward__contrib_DeformablePSROIPooling ::
                                           forall a t .
                                             (Tensor t,
                                              Fullfilled
                                                "__backward__contrib_DeformablePSROIPooling" t a) =>
                                             ArgsHMap "__backward__contrib_DeformablePSROIPooling" t
                                               a
                                               -> TensorApply t
__backward__contrib_DeformablePSROIPooling args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward__contrib_DeformablePSROIPooling" scalarArgs
        (Left tensorKeyArgs)

type instance
     ParameterList "__backward__contrib_MultiBoxDetection" t = '[]

__backward__contrib_MultiBoxDetection ::
                                      forall a t .
                                        (Tensor t,
                                         Fullfilled "__backward__contrib_MultiBoxDetection" t a) =>
                                        ArgsHMap "__backward__contrib_MultiBoxDetection" t a ->
                                          TensorApply t
__backward__contrib_MultiBoxDetection args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward__contrib_MultiBoxDetection" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward__contrib_MultiBoxPrior" t =
     '[]

__backward__contrib_MultiBoxPrior ::
                                  forall a t .
                                    (Tensor t,
                                     Fullfilled "__backward__contrib_MultiBoxPrior" t a) =>
                                    ArgsHMap "__backward__contrib_MultiBoxPrior" t a ->
                                      TensorApply t
__backward__contrib_MultiBoxPrior args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward__contrib_MultiBoxPrior" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward__contrib_MultiBoxTarget" t
     = '[]

__backward__contrib_MultiBoxTarget ::
                                   forall a t .
                                     (Tensor t,
                                      Fullfilled "__backward__contrib_MultiBoxTarget" t a) =>
                                     ArgsHMap "__backward__contrib_MultiBoxTarget" t a ->
                                       TensorApply t
__backward__contrib_MultiBoxTarget args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward__contrib_MultiBoxTarget" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward__contrib_MultiProposal" t =
     '[]

__backward__contrib_MultiProposal ::
                                  forall a t .
                                    (Tensor t,
                                     Fullfilled "__backward__contrib_MultiProposal" t a) =>
                                    ArgsHMap "__backward__contrib_MultiProposal" t a ->
                                      TensorApply t
__backward__contrib_MultiProposal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward__contrib_MultiProposal" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward__contrib_PSROIPooling" t =
     '[]

__backward__contrib_PSROIPooling ::
                                 forall a t .
                                   (Tensor t, Fullfilled "__backward__contrib_PSROIPooling" t a) =>
                                   ArgsHMap "__backward__contrib_PSROIPooling" t a -> TensorApply t
__backward__contrib_PSROIPooling args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward__contrib_PSROIPooling" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward__contrib_Proposal" t = '[]

__backward__contrib_Proposal ::
                             forall a t .
                               (Tensor t, Fullfilled "__backward__contrib_Proposal" t a) =>
                               ArgsHMap "__backward__contrib_Proposal" t a -> TensorApply t
__backward__contrib_Proposal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward__contrib_Proposal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward__contrib_SyncBatchNorm" t =
     '[]

__backward__contrib_SyncBatchNorm ::
                                  forall a t .
                                    (Tensor t,
                                     Fullfilled "__backward__contrib_SyncBatchNorm" t a) =>
                                    ArgsHMap "__backward__contrib_SyncBatchNorm" t a ->
                                      TensorApply t
__backward__contrib_SyncBatchNorm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward__contrib_SyncBatchNorm" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward__contrib_count_sketch" t =
     '[]

__backward__contrib_count_sketch ::
                                 forall a t .
                                   (Tensor t, Fullfilled "__backward__contrib_count_sketch" t a) =>
                                   ArgsHMap "__backward__contrib_count_sketch" t a -> TensorApply t
__backward__contrib_count_sketch args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward__contrib_count_sketch" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward__contrib_fft" t = '[]

__backward__contrib_fft ::
                        forall a t .
                          (Tensor t, Fullfilled "__backward__contrib_fft" t a) =>
                          ArgsHMap "__backward__contrib_fft" t a -> TensorApply t
__backward__contrib_fft args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward__contrib_fft" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward__contrib_ifft" t = '[]

__backward__contrib_ifft ::
                         forall a t .
                           (Tensor t, Fullfilled "__backward__contrib_ifft" t a) =>
                           ArgsHMap "__backward__contrib_ifft" t a -> TensorApply t
__backward__contrib_ifft args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward__contrib_ifft" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_abs" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_abs ::
               forall a t . (Tensor t, Fullfilled "__backward_abs" t a) =>
                 ArgsHMap "__backward_abs" t a -> TensorApply t
__backward_abs args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_abs" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_add" t = '[]

__backward_add ::
               forall a t . (Tensor t, Fullfilled "__backward_add" t a) =>
                 ArgsHMap "__backward_add" t a -> TensorApply t
__backward_add args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_add" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_amp_cast" t = '[]

__backward_amp_cast ::
                    forall a t . (Tensor t, Fullfilled "__backward_amp_cast" t a) =>
                      ArgsHMap "__backward_amp_cast" t a -> TensorApply t
__backward_amp_cast args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_amp_cast" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_amp_multicast" t =
     '[ '("num_outputs", AttrReq Int), '("cast_narrow", AttrOpt Bool),
        '("grad", AttrOpt [t])]

__backward_amp_multicast ::
                         forall a t .
                           (Tensor t, Fullfilled "__backward_amp_multicast" t a) =>
                           ArgsHMap "__backward_amp_multicast" t a -> TensorApply t
__backward_amp_multicast args
  = let scalarArgs
          = catMaybes
              [("num_outputs",) . showValue <$>
                 (args !? #num_outputs :: Maybe Int),
               ("cast_narrow",) . showValue <$>
                 (args !? #cast_narrow :: Maybe Bool)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #grad :: Maybe [t])
      in apply "_backward_amp_multicast" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__backward_arccos" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_arccos ::
                  forall a t . (Tensor t, Fullfilled "__backward_arccos" t a) =>
                    ArgsHMap "__backward_arccos" t a -> TensorApply t
__backward_arccos args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_arccos" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_arccosh" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_arccosh ::
                   forall a t . (Tensor t, Fullfilled "__backward_arccosh" t a) =>
                     ArgsHMap "__backward_arccosh" t a -> TensorApply t
__backward_arccosh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_arccosh" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_arcsin" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_arcsin ::
                  forall a t . (Tensor t, Fullfilled "__backward_arcsin" t a) =>
                    ArgsHMap "__backward_arcsin" t a -> TensorApply t
__backward_arcsin args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_arcsin" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_arcsinh" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_arcsinh ::
                   forall a t . (Tensor t, Fullfilled "__backward_arcsinh" t a) =>
                     ArgsHMap "__backward_arcsinh" t a -> TensorApply t
__backward_arcsinh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_arcsinh" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_arctan" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_arctan ::
                  forall a t . (Tensor t, Fullfilled "__backward_arctan" t a) =>
                    ArgsHMap "__backward_arctan" t a -> TensorApply t
__backward_arctan args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_arctan" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_arctanh" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_arctanh ::
                   forall a t . (Tensor t, Fullfilled "__backward_arctanh" t a) =>
                     ArgsHMap "__backward_arctanh" t a -> TensorApply t
__backward_arctanh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_arctanh" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_backward_FullyConnected" t
     = '[]

__backward_backward_FullyConnected ::
                                   forall a t .
                                     (Tensor t,
                                      Fullfilled "__backward_backward_FullyConnected" t a) =>
                                     ArgsHMap "__backward_backward_FullyConnected" t a ->
                                       TensorApply t
__backward_backward_FullyConnected args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_backward_FullyConnected" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward_broadcast_add" t = '[]

__backward_broadcast_add ::
                         forall a t .
                           (Tensor t, Fullfilled "__backward_broadcast_add" t a) =>
                           ArgsHMap "__backward_broadcast_add" t a -> TensorApply t
__backward_broadcast_add args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_broadcast_add" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_broadcast_div" t = '[]

__backward_broadcast_div ::
                         forall a t .
                           (Tensor t, Fullfilled "__backward_broadcast_div" t a) =>
                           ArgsHMap "__backward_broadcast_div" t a -> TensorApply t
__backward_broadcast_div args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_broadcast_div" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_broadcast_hypot" t = '[]

__backward_broadcast_hypot ::
                           forall a t .
                             (Tensor t, Fullfilled "__backward_broadcast_hypot" t a) =>
                             ArgsHMap "__backward_broadcast_hypot" t a -> TensorApply t
__backward_broadcast_hypot args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_broadcast_hypot" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_broadcast_maximum" t = '[]

__backward_broadcast_maximum ::
                             forall a t .
                               (Tensor t, Fullfilled "__backward_broadcast_maximum" t a) =>
                               ArgsHMap "__backward_broadcast_maximum" t a -> TensorApply t
__backward_broadcast_maximum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_broadcast_maximum" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_broadcast_minimum" t = '[]

__backward_broadcast_minimum ::
                             forall a t .
                               (Tensor t, Fullfilled "__backward_broadcast_minimum" t a) =>
                               ArgsHMap "__backward_broadcast_minimum" t a -> TensorApply t
__backward_broadcast_minimum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_broadcast_minimum" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_broadcast_mod" t = '[]

__backward_broadcast_mod ::
                         forall a t .
                           (Tensor t, Fullfilled "__backward_broadcast_mod" t a) =>
                           ArgsHMap "__backward_broadcast_mod" t a -> TensorApply t
__backward_broadcast_mod args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_broadcast_mod" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_broadcast_mul" t = '[]

__backward_broadcast_mul ::
                         forall a t .
                           (Tensor t, Fullfilled "__backward_broadcast_mul" t a) =>
                           ArgsHMap "__backward_broadcast_mul" t a -> TensorApply t
__backward_broadcast_mul args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_broadcast_mul" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_broadcast_power" t = '[]

__backward_broadcast_power ::
                           forall a t .
                             (Tensor t, Fullfilled "__backward_broadcast_power" t a) =>
                             ArgsHMap "__backward_broadcast_power" t a -> TensorApply t
__backward_broadcast_power args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_broadcast_power" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_broadcast_sub" t = '[]

__backward_broadcast_sub ::
                         forall a t .
                           (Tensor t, Fullfilled "__backward_broadcast_sub" t a) =>
                           ArgsHMap "__backward_broadcast_sub" t a -> TensorApply t
__backward_broadcast_sub args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_broadcast_sub" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_cast" t = '[]

__backward_cast ::
                forall a t . (Tensor t, Fullfilled "__backward_cast" t a) =>
                  ArgsHMap "__backward_cast" t a -> TensorApply t
__backward_cast args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_cast" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_cbrt" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_cbrt ::
                forall a t . (Tensor t, Fullfilled "__backward_cbrt" t a) =>
                  ArgsHMap "__backward_cbrt" t a -> TensorApply t
__backward_cbrt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_cbrt" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_clip" t = '[]

__backward_clip ::
                forall a t . (Tensor t, Fullfilled "__backward_clip" t a) =>
                  ArgsHMap "__backward_clip" t a -> TensorApply t
__backward_clip args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_clip" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_cond" t = '[]

__backward_cond ::
                forall a t . (Tensor t, Fullfilled "__backward_cond" t a) =>
                  ArgsHMap "__backward_cond" t a -> TensorApply t
__backward_cond args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_cond" scalarArgs (Left tensorKeyArgs)

type instance
     ParameterList "__backward_contrib_AdaptiveAvgPooling2D" t = '[]

__backward_contrib_AdaptiveAvgPooling2D ::
                                        forall a t .
                                          (Tensor t,
                                           Fullfilled "__backward_contrib_AdaptiveAvgPooling2D" t
                                             a) =>
                                          ArgsHMap "__backward_contrib_AdaptiveAvgPooling2D" t a ->
                                            TensorApply t
__backward_contrib_AdaptiveAvgPooling2D args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_contrib_AdaptiveAvgPooling2D" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward_contrib_BilinearResize2D" t
     = '[]

__backward_contrib_BilinearResize2D ::
                                    forall a t .
                                      (Tensor t,
                                       Fullfilled "__backward_contrib_BilinearResize2D" t a) =>
                                      ArgsHMap "__backward_contrib_BilinearResize2D" t a ->
                                        TensorApply t
__backward_contrib_BilinearResize2D args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_contrib_BilinearResize2D" scalarArgs
        (Left tensorKeyArgs)

type instance
     ParameterList "__backward_contrib_bipartite_matching" t =
     '[ '("is_ascend", AttrOpt Bool), '("threshold", AttrReq Float),
        '("topk", AttrOpt Int)]

__backward_contrib_bipartite_matching ::
                                      forall a t .
                                        (Tensor t,
                                         Fullfilled "__backward_contrib_bipartite_matching" t a) =>
                                        ArgsHMap "__backward_contrib_bipartite_matching" t a ->
                                          TensorApply t
__backward_contrib_bipartite_matching args
  = let scalarArgs
          = catMaybes
              [("is_ascend",) . showValue <$> (args !? #is_ascend :: Maybe Bool),
               ("threshold",) . showValue <$> (args !? #threshold :: Maybe Float),
               ("topk",) . showValue <$> (args !? #topk :: Maybe Int)]
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_contrib_bipartite_matching" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward_contrib_boolean_mask" t =
     '[ '("axis", AttrOpt Int)]

__backward_contrib_boolean_mask ::
                                forall a t .
                                  (Tensor t, Fullfilled "__backward_contrib_boolean_mask" t a) =>
                                  ArgsHMap "__backward_contrib_boolean_mask" t a -> TensorApply t
__backward_contrib_boolean_mask args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_contrib_boolean_mask" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward_contrib_box_iou" t =
     '[ '("format", AttrOpt (EnumType '["center", "corner"]))]

__backward_contrib_box_iou ::
                           forall a t .
                             (Tensor t, Fullfilled "__backward_contrib_box_iou" t a) =>
                             ArgsHMap "__backward_contrib_box_iou" t a -> TensorApply t
__backward_contrib_box_iou args
  = let scalarArgs
          = catMaybes
              [("format",) . showValue <$>
                 (args !? #format :: Maybe (EnumType '["center", "corner"]))]
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_contrib_box_iou" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_contrib_box_nms" t =
     '[ '("overlap_thresh", AttrOpt Float),
        '("valid_thresh", AttrOpt Float), '("topk", AttrOpt Int),
        '("coord_start", AttrOpt Int), '("score_index", AttrOpt Int),
        '("id_index", AttrOpt Int), '("background_id", AttrOpt Int),
        '("force_suppress", AttrOpt Bool),
        '("in_format", AttrOpt (EnumType '["center", "corner"])),
        '("out_format", AttrOpt (EnumType '["center", "corner"]))]

__backward_contrib_box_nms ::
                           forall a t .
                             (Tensor t, Fullfilled "__backward_contrib_box_nms" t a) =>
                             ArgsHMap "__backward_contrib_box_nms" t a -> TensorApply t
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
      apply "_backward_contrib_box_nms" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_copy" t = '[]

__backward_copy ::
                forall a t . (Tensor t, Fullfilled "__backward_copy" t a) =>
                  ArgsHMap "__backward_copy" t a -> TensorApply t
__backward_copy args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_copy" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_cos" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_cos ::
               forall a t . (Tensor t, Fullfilled "__backward_cos" t a) =>
                 ArgsHMap "__backward_cos" t a -> TensorApply t
__backward_cos args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_cos" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_cosh" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_cosh ::
                forall a t . (Tensor t, Fullfilled "__backward_cosh" t a) =>
                  ArgsHMap "__backward_cosh" t a -> TensorApply t
__backward_cosh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_cosh" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_ctc_loss" t = '[]

__backward_ctc_loss ::
                    forall a t . (Tensor t, Fullfilled "__backward_ctc_loss" t a) =>
                      ArgsHMap "__backward_ctc_loss" t a -> TensorApply t
__backward_ctc_loss args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_ctc_loss" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_degrees" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_degrees ::
                   forall a t . (Tensor t, Fullfilled "__backward_degrees" t a) =>
                     ArgsHMap "__backward_degrees" t a -> TensorApply t
__backward_degrees args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_degrees" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_diag" t = '[]

__backward_diag ::
                forall a t . (Tensor t, Fullfilled "__backward_diag" t a) =>
                  ArgsHMap "__backward_diag" t a -> TensorApply t
__backward_diag args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_diag" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_div" t = '[]

__backward_div ::
               forall a t . (Tensor t, Fullfilled "__backward_div" t a) =>
                 ArgsHMap "__backward_div" t a -> TensorApply t
__backward_div args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_div" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_div_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__backward_div_scalar ::
                      forall a t . (Tensor t, Fullfilled "__backward_div_scalar" t a) =>
                        ArgsHMap "__backward_div_scalar" t a -> TensorApply t
__backward_div_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_backward_div_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_dot" t =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("forward_stype",
          AttrOpt (Maybe (EnumType '["csr", "default", "row_sparse"])))]

__backward_dot ::
               forall a t . (Tensor t, Fullfilled "__backward_dot" t a) =>
                 ArgsHMap "__backward_dot" t a -> TensorApply t
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
      in apply "_backward_dot" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_erf" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_erf ::
               forall a t . (Tensor t, Fullfilled "__backward_erf" t a) =>
                 ArgsHMap "__backward_erf" t a -> TensorApply t
__backward_erf args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_erf" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_erfinv" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_erfinv ::
                  forall a t . (Tensor t, Fullfilled "__backward_erfinv" t a) =>
                    ArgsHMap "__backward_erfinv" t a -> TensorApply t
__backward_erfinv args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_erfinv" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_expm1" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_expm1 ::
                 forall a t . (Tensor t, Fullfilled "__backward_expm1" t a) =>
                   ArgsHMap "__backward_expm1" t a -> TensorApply t
__backward_expm1 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_expm1" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_foreach" t = '[]

__backward_foreach ::
                   forall a t . (Tensor t, Fullfilled "__backward_foreach" t a) =>
                     ArgsHMap "__backward_foreach" t a -> TensorApply t
__backward_foreach args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_foreach" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_gamma" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_gamma ::
                 forall a t . (Tensor t, Fullfilled "__backward_gamma" t a) =>
                   ArgsHMap "__backward_gamma" t a -> TensorApply t
__backward_gamma args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_gamma" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_gammaln" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_gammaln ::
                   forall a t . (Tensor t, Fullfilled "__backward_gammaln" t a) =>
                     ArgsHMap "__backward_gammaln" t a -> TensorApply t
__backward_gammaln args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_gammaln" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_gather_nd" t =
     '[ '("shape", AttrReq [Int]), '("data", AttrOpt t),
        '("indices", AttrOpt t)]

__backward_gather_nd ::
                     forall a t . (Tensor t, Fullfilled "__backward_gather_nd" t a) =>
                       ArgsHMap "__backward_gather_nd" t a -> TensorApply t
__backward_gather_nd args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("indices",) <$> (args !? #indices :: Maybe t)]
      in apply "_backward_gather_nd" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_hard_sigmoid" t = '[]

__backward_hard_sigmoid ::
                        forall a t .
                          (Tensor t, Fullfilled "__backward_hard_sigmoid" t a) =>
                          ArgsHMap "__backward_hard_sigmoid" t a -> TensorApply t
__backward_hard_sigmoid args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_hard_sigmoid" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_hypot" t = '[]

__backward_hypot ::
                 forall a t . (Tensor t, Fullfilled "__backward_hypot" t a) =>
                   ArgsHMap "__backward_hypot" t a -> TensorApply t
__backward_hypot args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_hypot" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_hypot_scalar" t =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt t),
        '("rhs", AttrOpt t)]

__backward_hypot_scalar ::
                        forall a t .
                          (Tensor t, Fullfilled "__backward_hypot_scalar" t a) =>
                          ArgsHMap "__backward_hypot_scalar" t a -> TensorApply t
__backward_hypot_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_hypot_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_image_crop" t = '[]

__backward_image_crop ::
                      forall a t . (Tensor t, Fullfilled "__backward_image_crop" t a) =>
                        ArgsHMap "__backward_image_crop" t a -> TensorApply t
__backward_image_crop args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_image_crop" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_image_normalize" t = '[]

__backward_image_normalize ::
                           forall a t .
                             (Tensor t, Fullfilled "__backward_image_normalize" t a) =>
                             ArgsHMap "__backward_image_normalize" t a -> TensorApply t
__backward_image_normalize args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_image_normalize" scalarArgs (Left tensorKeyArgs)

type instance
     ParameterList "__backward_interleaved_matmul_encdec_qk" t = '[]

__backward_interleaved_matmul_encdec_qk ::
                                        forall a t .
                                          (Tensor t,
                                           Fullfilled "__backward_interleaved_matmul_encdec_qk" t
                                             a) =>
                                          ArgsHMap "__backward_interleaved_matmul_encdec_qk" t a ->
                                            TensorApply t
__backward_interleaved_matmul_encdec_qk args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_interleaved_matmul_encdec_qk" scalarArgs
        (Left tensorKeyArgs)

type instance
     ParameterList "__backward_interleaved_matmul_encdec_valatt" t = '[]

__backward_interleaved_matmul_encdec_valatt ::
                                            forall a t .
                                              (Tensor t,
                                               Fullfilled
                                                 "__backward_interleaved_matmul_encdec_valatt" t
                                                 a) =>
                                              ArgsHMap "__backward_interleaved_matmul_encdec_valatt"
                                                t
                                                a
                                                -> TensorApply t
__backward_interleaved_matmul_encdec_valatt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_interleaved_matmul_encdec_valatt" scalarArgs
        (Left tensorKeyArgs)

type instance
     ParameterList "__backward_interleaved_matmul_selfatt_qk" t = '[]

__backward_interleaved_matmul_selfatt_qk ::
                                         forall a t .
                                           (Tensor t,
                                            Fullfilled "__backward_interleaved_matmul_selfatt_qk" t
                                              a) =>
                                           ArgsHMap "__backward_interleaved_matmul_selfatt_qk" t a
                                             -> TensorApply t
__backward_interleaved_matmul_selfatt_qk args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_interleaved_matmul_selfatt_qk" scalarArgs
        (Left tensorKeyArgs)

type instance
     ParameterList "__backward_interleaved_matmul_selfatt_valatt" t =
     '[]

__backward_interleaved_matmul_selfatt_valatt ::
                                             forall a t .
                                               (Tensor t,
                                                Fullfilled
                                                  "__backward_interleaved_matmul_selfatt_valatt" t
                                                  a) =>
                                               ArgsHMap
                                                 "__backward_interleaved_matmul_selfatt_valatt"
                                                 t
                                                 a
                                                 -> TensorApply t
__backward_interleaved_matmul_selfatt_valatt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_interleaved_matmul_selfatt_valatt" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward_linalg_det" t = '[]

__backward_linalg_det ::
                      forall a t . (Tensor t, Fullfilled "__backward_linalg_det" t a) =>
                        ArgsHMap "__backward_linalg_det" t a -> TensorApply t
__backward_linalg_det args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_linalg_det" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_linalg_extractdiag" t = '[]

__backward_linalg_extractdiag ::
                              forall a t .
                                (Tensor t, Fullfilled "__backward_linalg_extractdiag" t a) =>
                                ArgsHMap "__backward_linalg_extractdiag" t a -> TensorApply t
__backward_linalg_extractdiag args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_linalg_extractdiag" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward_linalg_extracttrian" t =
     '[]

__backward_linalg_extracttrian ::
                               forall a t .
                                 (Tensor t, Fullfilled "__backward_linalg_extracttrian" t a) =>
                                 ArgsHMap "__backward_linalg_extracttrian" t a -> TensorApply t
__backward_linalg_extracttrian args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_linalg_extracttrian" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward_linalg_gelqf" t = '[]

__backward_linalg_gelqf ::
                        forall a t .
                          (Tensor t, Fullfilled "__backward_linalg_gelqf" t a) =>
                          ArgsHMap "__backward_linalg_gelqf" t a -> TensorApply t
__backward_linalg_gelqf args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_linalg_gelqf" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_linalg_gemm" t = '[]

__backward_linalg_gemm ::
                       forall a t . (Tensor t, Fullfilled "__backward_linalg_gemm" t a) =>
                         ArgsHMap "__backward_linalg_gemm" t a -> TensorApply t
__backward_linalg_gemm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_linalg_gemm" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_linalg_gemm2" t = '[]

__backward_linalg_gemm2 ::
                        forall a t .
                          (Tensor t, Fullfilled "__backward_linalg_gemm2" t a) =>
                          ArgsHMap "__backward_linalg_gemm2" t a -> TensorApply t
__backward_linalg_gemm2 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_linalg_gemm2" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_linalg_inverse" t = '[]

__backward_linalg_inverse ::
                          forall a t .
                            (Tensor t, Fullfilled "__backward_linalg_inverse" t a) =>
                            ArgsHMap "__backward_linalg_inverse" t a -> TensorApply t
__backward_linalg_inverse args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_linalg_inverse" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_linalg_makediag" t = '[]

__backward_linalg_makediag ::
                           forall a t .
                             (Tensor t, Fullfilled "__backward_linalg_makediag" t a) =>
                             ArgsHMap "__backward_linalg_makediag" t a -> TensorApply t
__backward_linalg_makediag args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_linalg_makediag" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_linalg_maketrian" t = '[]

__backward_linalg_maketrian ::
                            forall a t .
                              (Tensor t, Fullfilled "__backward_linalg_maketrian" t a) =>
                              ArgsHMap "__backward_linalg_maketrian" t a -> TensorApply t
__backward_linalg_maketrian args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_linalg_maketrian" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_linalg_potrf" t = '[]

__backward_linalg_potrf ::
                        forall a t .
                          (Tensor t, Fullfilled "__backward_linalg_potrf" t a) =>
                          ArgsHMap "__backward_linalg_potrf" t a -> TensorApply t
__backward_linalg_potrf args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_linalg_potrf" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_linalg_potri" t = '[]

__backward_linalg_potri ::
                        forall a t .
                          (Tensor t, Fullfilled "__backward_linalg_potri" t a) =>
                          ArgsHMap "__backward_linalg_potri" t a -> TensorApply t
__backward_linalg_potri args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_linalg_potri" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_linalg_slogdet" t = '[]

__backward_linalg_slogdet ::
                          forall a t .
                            (Tensor t, Fullfilled "__backward_linalg_slogdet" t a) =>
                            ArgsHMap "__backward_linalg_slogdet" t a -> TensorApply t
__backward_linalg_slogdet args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_linalg_slogdet" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_linalg_sumlogdiag" t = '[]

__backward_linalg_sumlogdiag ::
                             forall a t .
                               (Tensor t, Fullfilled "__backward_linalg_sumlogdiag" t a) =>
                               ArgsHMap "__backward_linalg_sumlogdiag" t a -> TensorApply t
__backward_linalg_sumlogdiag args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_linalg_sumlogdiag" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_linalg_syevd" t = '[]

__backward_linalg_syevd ::
                        forall a t .
                          (Tensor t, Fullfilled "__backward_linalg_syevd" t a) =>
                          ArgsHMap "__backward_linalg_syevd" t a -> TensorApply t
__backward_linalg_syevd args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_linalg_syevd" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_linalg_syrk" t = '[]

__backward_linalg_syrk ::
                       forall a t . (Tensor t, Fullfilled "__backward_linalg_syrk" t a) =>
                         ArgsHMap "__backward_linalg_syrk" t a -> TensorApply t
__backward_linalg_syrk args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_linalg_syrk" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_linalg_trmm" t = '[]

__backward_linalg_trmm ::
                       forall a t . (Tensor t, Fullfilled "__backward_linalg_trmm" t a) =>
                         ArgsHMap "__backward_linalg_trmm" t a -> TensorApply t
__backward_linalg_trmm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_linalg_trmm" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_linalg_trsm" t = '[]

__backward_linalg_trsm ::
                       forall a t . (Tensor t, Fullfilled "__backward_linalg_trsm" t a) =>
                         ArgsHMap "__backward_linalg_trsm" t a -> TensorApply t
__backward_linalg_trsm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_linalg_trsm" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_linear_reg_out" t = '[]

__backward_linear_reg_out ::
                          forall a t .
                            (Tensor t, Fullfilled "__backward_linear_reg_out" t a) =>
                            ArgsHMap "__backward_linear_reg_out" t a -> TensorApply t
__backward_linear_reg_out args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_linear_reg_out" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_log" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_log ::
               forall a t . (Tensor t, Fullfilled "__backward_log" t a) =>
                 ArgsHMap "__backward_log" t a -> TensorApply t
__backward_log args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_log" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_log10" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_log10 ::
                 forall a t . (Tensor t, Fullfilled "__backward_log10" t a) =>
                   ArgsHMap "__backward_log10" t a -> TensorApply t
__backward_log10 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_log10" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_log1p" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_log1p ::
                 forall a t . (Tensor t, Fullfilled "__backward_log1p" t a) =>
                   ArgsHMap "__backward_log1p" t a -> TensorApply t
__backward_log1p args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_log1p" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_log2" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_log2 ::
                forall a t . (Tensor t, Fullfilled "__backward_log2" t a) =>
                  ArgsHMap "__backward_log2" t a -> TensorApply t
__backward_log2 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_log2" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_log_softmax" t =
     '[ '("args", AttrOpt [t])]

__backward_log_softmax ::
                       forall a t . (Tensor t, Fullfilled "__backward_log_softmax" t a) =>
                         ArgsHMap "__backward_log_softmax" t a -> TensorApply t
__backward_log_softmax args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #args :: Maybe [t])
      in apply "_backward_log_softmax" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__backward_logistic_reg_out" t = '[]

__backward_logistic_reg_out ::
                            forall a t .
                              (Tensor t, Fullfilled "__backward_logistic_reg_out" t a) =>
                              ArgsHMap "__backward_logistic_reg_out" t a -> TensorApply t
__backward_logistic_reg_out args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_logistic_reg_out" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_mae_reg_out" t = '[]

__backward_mae_reg_out ::
                       forall a t . (Tensor t, Fullfilled "__backward_mae_reg_out" t a) =>
                         ArgsHMap "__backward_mae_reg_out" t a -> TensorApply t
__backward_mae_reg_out args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_mae_reg_out" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_max" t = '[]

__backward_max ::
               forall a t . (Tensor t, Fullfilled "__backward_max" t a) =>
                 ArgsHMap "__backward_max" t a -> TensorApply t
__backward_max args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_max" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_maximum" t = '[]

__backward_maximum ::
                   forall a t . (Tensor t, Fullfilled "__backward_maximum" t a) =>
                     ArgsHMap "__backward_maximum" t a -> TensorApply t
__backward_maximum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_maximum" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_maximum_scalar" t =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt t),
        '("rhs", AttrOpt t)]

__backward_maximum_scalar ::
                          forall a t .
                            (Tensor t, Fullfilled "__backward_maximum_scalar" t a) =>
                            ArgsHMap "__backward_maximum_scalar" t a -> TensorApply t
__backward_maximum_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_maximum_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_mean" t = '[]

__backward_mean ::
                forall a t . (Tensor t, Fullfilled "__backward_mean" t a) =>
                  ArgsHMap "__backward_mean" t a -> TensorApply t
__backward_mean args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_mean" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_min" t = '[]

__backward_min ::
               forall a t . (Tensor t, Fullfilled "__backward_min" t a) =>
                 ArgsHMap "__backward_min" t a -> TensorApply t
__backward_min args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_min" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_minimum" t = '[]

__backward_minimum ::
                   forall a t . (Tensor t, Fullfilled "__backward_minimum" t a) =>
                     ArgsHMap "__backward_minimum" t a -> TensorApply t
__backward_minimum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_minimum" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_minimum_scalar" t =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt t),
        '("rhs", AttrOpt t)]

__backward_minimum_scalar ::
                          forall a t .
                            (Tensor t, Fullfilled "__backward_minimum_scalar" t a) =>
                            ArgsHMap "__backward_minimum_scalar" t a -> TensorApply t
__backward_minimum_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_minimum_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_mod" t = '[]

__backward_mod ::
               forall a t . (Tensor t, Fullfilled "__backward_mod" t a) =>
                 ArgsHMap "__backward_mod" t a -> TensorApply t
__backward_mod args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_mod" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_mod_scalar" t =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt t),
        '("rhs", AttrOpt t)]

__backward_mod_scalar ::
                      forall a t . (Tensor t, Fullfilled "__backward_mod_scalar" t a) =>
                        ArgsHMap "__backward_mod_scalar" t a -> TensorApply t
__backward_mod_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_mod_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_moments" t = '[]

__backward_moments ::
                   forall a t . (Tensor t, Fullfilled "__backward_moments" t a) =>
                     ArgsHMap "__backward_moments" t a -> TensorApply t
__backward_moments args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_moments" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_mul" t = '[]

__backward_mul ::
               forall a t . (Tensor t, Fullfilled "__backward_mul" t a) =>
                 ArgsHMap "__backward_mul" t a -> TensorApply t
__backward_mul args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_mul" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_mul_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__backward_mul_scalar ::
                      forall a t . (Tensor t, Fullfilled "__backward_mul_scalar" t a) =>
                        ArgsHMap "__backward_mul_scalar" t a -> TensorApply t
__backward_mul_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_backward_mul_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_nanprod" t = '[]

__backward_nanprod ::
                   forall a t . (Tensor t, Fullfilled "__backward_nanprod" t a) =>
                     ArgsHMap "__backward_nanprod" t a -> TensorApply t
__backward_nanprod args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_nanprod" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_nansum" t = '[]

__backward_nansum ::
                  forall a t . (Tensor t, Fullfilled "__backward_nansum" t a) =>
                    ArgsHMap "__backward_nansum" t a -> TensorApply t
__backward_nansum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_nansum" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_norm" t = '[]

__backward_norm ::
                forall a t . (Tensor t, Fullfilled "__backward_norm" t a) =>
                  ArgsHMap "__backward_norm" t a -> TensorApply t
__backward_norm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_norm" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_np_broadcast_to" t = '[]

__backward_np_broadcast_to ::
                           forall a t .
                             (Tensor t, Fullfilled "__backward_np_broadcast_to" t a) =>
                             ArgsHMap "__backward_np_broadcast_to" t a -> TensorApply t
__backward_np_broadcast_to args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_np_broadcast_to" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_np_column_stack" t = '[]

__backward_np_column_stack ::
                           forall a t .
                             (Tensor t, Fullfilled "__backward_np_column_stack" t a) =>
                             ArgsHMap "__backward_np_column_stack" t a -> TensorApply t
__backward_np_column_stack args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_np_column_stack" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_np_concat" t = '[]

__backward_np_concat ::
                     forall a t . (Tensor t, Fullfilled "__backward_np_concat" t a) =>
                       ArgsHMap "__backward_np_concat" t a -> TensorApply t
__backward_np_concat args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_np_concat" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_np_cumsum" t = '[]

__backward_np_cumsum ::
                     forall a t . (Tensor t, Fullfilled "__backward_np_cumsum" t a) =>
                       ArgsHMap "__backward_np_cumsum" t a -> TensorApply t
__backward_np_cumsum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_np_cumsum" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_np_dot" t = '[]

__backward_np_dot ::
                  forall a t . (Tensor t, Fullfilled "__backward_np_dot" t a) =>
                    ArgsHMap "__backward_np_dot" t a -> TensorApply t
__backward_np_dot args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_np_dot" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_np_dstack" t = '[]

__backward_np_dstack ::
                     forall a t . (Tensor t, Fullfilled "__backward_np_dstack" t a) =>
                       ArgsHMap "__backward_np_dstack" t a -> TensorApply t
__backward_np_dstack args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_np_dstack" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_np_max" t = '[]

__backward_np_max ::
                  forall a t . (Tensor t, Fullfilled "__backward_np_max" t a) =>
                    ArgsHMap "__backward_np_max" t a -> TensorApply t
__backward_np_max args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_np_max" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_np_mean" t = '[]

__backward_np_mean ::
                   forall a t . (Tensor t, Fullfilled "__backward_np_mean" t a) =>
                     ArgsHMap "__backward_np_mean" t a -> TensorApply t
__backward_np_mean args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_np_mean" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_np_min" t = '[]

__backward_np_min ::
                  forall a t . (Tensor t, Fullfilled "__backward_np_min" t a) =>
                    ArgsHMap "__backward_np_min" t a -> TensorApply t
__backward_np_min args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_np_min" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_np_prod" t = '[]

__backward_np_prod ::
                   forall a t . (Tensor t, Fullfilled "__backward_np_prod" t a) =>
                     ArgsHMap "__backward_np_prod" t a -> TensorApply t
__backward_np_prod args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_np_prod" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_np_sum" t = '[]

__backward_np_sum ::
                  forall a t . (Tensor t, Fullfilled "__backward_np_sum" t a) =>
                    ArgsHMap "__backward_np_sum" t a -> TensorApply t
__backward_np_sum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_np_sum" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_np_trace" t = '[]

__backward_np_trace ::
                    forall a t . (Tensor t, Fullfilled "__backward_np_trace" t a) =>
                      ArgsHMap "__backward_np_trace" t a -> TensorApply t
__backward_np_trace args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_np_trace" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_np_vstack" t = '[]

__backward_np_vstack ::
                     forall a t . (Tensor t, Fullfilled "__backward_np_vstack" t a) =>
                       ArgsHMap "__backward_np_vstack" t a -> TensorApply t
__backward_np_vstack args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_np_vstack" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_npi_arctan2" t = '[]

__backward_npi_arctan2 ::
                       forall a t . (Tensor t, Fullfilled "__backward_npi_arctan2" t a) =>
                         ArgsHMap "__backward_npi_arctan2" t a -> TensorApply t
__backward_npi_arctan2 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_npi_arctan2" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_npi_arctan2_scalar" t =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt t),
        '("rhs", AttrOpt t)]

__backward_npi_arctan2_scalar ::
                              forall a t .
                                (Tensor t, Fullfilled "__backward_npi_arctan2_scalar" t a) =>
                                ArgsHMap "__backward_npi_arctan2_scalar" t a -> TensorApply t
__backward_npi_arctan2_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in
      apply "_backward_npi_arctan2_scalar" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward_npi_broadcast_mul" t = '[]

__backward_npi_broadcast_mul ::
                             forall a t .
                               (Tensor t, Fullfilled "__backward_npi_broadcast_mul" t a) =>
                               ArgsHMap "__backward_npi_broadcast_mul" t a -> TensorApply t
__backward_npi_broadcast_mul args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_npi_broadcast_mul" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_npi_copysign" t = '[]

__backward_npi_copysign ::
                        forall a t .
                          (Tensor t, Fullfilled "__backward_npi_copysign" t a) =>
                          ArgsHMap "__backward_npi_copysign" t a -> TensorApply t
__backward_npi_copysign args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_npi_copysign" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_npi_copysign_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__backward_npi_copysign_scalar ::
                               forall a t .
                                 (Tensor t, Fullfilled "__backward_npi_copysign_scalar" t a) =>
                                 ArgsHMap "__backward_npi_copysign_scalar" t a -> TensorApply t
__backward_npi_copysign_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in
      apply "_backward_npi_copysign_scalar" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward_npi_diff" t = '[]

__backward_npi_diff ::
                    forall a t . (Tensor t, Fullfilled "__backward_npi_diff" t a) =>
                      ArgsHMap "__backward_npi_diff" t a -> TensorApply t
__backward_npi_diff args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_npi_diff" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_npi_einsum" t = '[]

__backward_npi_einsum ::
                      forall a t . (Tensor t, Fullfilled "__backward_npi_einsum" t a) =>
                        ArgsHMap "__backward_npi_einsum" t a -> TensorApply t
__backward_npi_einsum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_npi_einsum" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_npi_flip" t = '[]

__backward_npi_flip ::
                    forall a t . (Tensor t, Fullfilled "__backward_npi_flip" t a) =>
                      ArgsHMap "__backward_npi_flip" t a -> TensorApply t
__backward_npi_flip args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_npi_flip" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_npi_hypot" t = '[]

__backward_npi_hypot ::
                     forall a t . (Tensor t, Fullfilled "__backward_npi_hypot" t a) =>
                       ArgsHMap "__backward_npi_hypot" t a -> TensorApply t
__backward_npi_hypot args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_npi_hypot" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_npi_ldexp" t = '[]

__backward_npi_ldexp ::
                     forall a t . (Tensor t, Fullfilled "__backward_npi_ldexp" t a) =>
                       ArgsHMap "__backward_npi_ldexp" t a -> TensorApply t
__backward_npi_ldexp args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_npi_ldexp" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_npi_ldexp_scalar" t =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt t),
        '("rhs", AttrOpt t)]

__backward_npi_ldexp_scalar ::
                            forall a t .
                              (Tensor t, Fullfilled "__backward_npi_ldexp_scalar" t a) =>
                              ArgsHMap "__backward_npi_ldexp_scalar" t a -> TensorApply t
__backward_npi_ldexp_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in
      apply "_backward_npi_ldexp_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_npi_rarctan2_scalar" t =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt t),
        '("rhs", AttrOpt t)]

__backward_npi_rarctan2_scalar ::
                               forall a t .
                                 (Tensor t, Fullfilled "__backward_npi_rarctan2_scalar" t a) =>
                                 ArgsHMap "__backward_npi_rarctan2_scalar" t a -> TensorApply t
__backward_npi_rarctan2_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in
      apply "_backward_npi_rarctan2_scalar" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward_npi_rcopysign_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__backward_npi_rcopysign_scalar ::
                                forall a t .
                                  (Tensor t, Fullfilled "__backward_npi_rcopysign_scalar" t a) =>
                                  ArgsHMap "__backward_npi_rcopysign_scalar" t a -> TensorApply t
__backward_npi_rcopysign_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in
      apply "_backward_npi_rcopysign_scalar" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward_npi_rldexp_scalar" t =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt t),
        '("rhs", AttrOpt t)]

__backward_npi_rldexp_scalar ::
                             forall a t .
                               (Tensor t, Fullfilled "__backward_npi_rldexp_scalar" t a) =>
                               ArgsHMap "__backward_npi_rldexp_scalar" t a -> TensorApply t
__backward_npi_rldexp_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in
      apply "_backward_npi_rldexp_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_npi_svd" t = '[]

__backward_npi_svd ::
                   forall a t . (Tensor t, Fullfilled "__backward_npi_svd" t a) =>
                     ArgsHMap "__backward_npi_svd" t a -> TensorApply t
__backward_npi_svd args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_npi_svd" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_npi_tensordot" t = '[]

__backward_npi_tensordot ::
                         forall a t .
                           (Tensor t, Fullfilled "__backward_npi_tensordot" t a) =>
                           ArgsHMap "__backward_npi_tensordot" t a -> TensorApply t
__backward_npi_tensordot args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_npi_tensordot" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_npi_tensordot_int_axes" t =
     '[]

__backward_npi_tensordot_int_axes ::
                                  forall a t .
                                    (Tensor t,
                                     Fullfilled "__backward_npi_tensordot_int_axes" t a) =>
                                    ArgsHMap "__backward_npi_tensordot_int_axes" t a ->
                                      TensorApply t
__backward_npi_tensordot_int_axes args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_npi_tensordot_int_axes" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward_pdf_dirichlet" t = '[]

__backward_pdf_dirichlet ::
                         forall a t .
                           (Tensor t, Fullfilled "__backward_pdf_dirichlet" t a) =>
                           ArgsHMap "__backward_pdf_dirichlet" t a -> TensorApply t
__backward_pdf_dirichlet args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_pdf_dirichlet" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_pdf_exponential" t = '[]

__backward_pdf_exponential ::
                           forall a t .
                             (Tensor t, Fullfilled "__backward_pdf_exponential" t a) =>
                             ArgsHMap "__backward_pdf_exponential" t a -> TensorApply t
__backward_pdf_exponential args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_pdf_exponential" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_pdf_gamma" t = '[]

__backward_pdf_gamma ::
                     forall a t . (Tensor t, Fullfilled "__backward_pdf_gamma" t a) =>
                       ArgsHMap "__backward_pdf_gamma" t a -> TensorApply t
__backward_pdf_gamma args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_pdf_gamma" scalarArgs (Left tensorKeyArgs)

type instance
     ParameterList "__backward_pdf_generalized_negative_binomial" t =
     '[]

__backward_pdf_generalized_negative_binomial ::
                                             forall a t .
                                               (Tensor t,
                                                Fullfilled
                                                  "__backward_pdf_generalized_negative_binomial" t
                                                  a) =>
                                               ArgsHMap
                                                 "__backward_pdf_generalized_negative_binomial"
                                                 t
                                                 a
                                                 -> TensorApply t
__backward_pdf_generalized_negative_binomial args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_pdf_generalized_negative_binomial" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward_pdf_negative_binomial" t =
     '[]

__backward_pdf_negative_binomial ::
                                 forall a t .
                                   (Tensor t, Fullfilled "__backward_pdf_negative_binomial" t a) =>
                                   ArgsHMap "__backward_pdf_negative_binomial" t a -> TensorApply t
__backward_pdf_negative_binomial args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_pdf_negative_binomial" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward_pdf_normal" t = '[]

__backward_pdf_normal ::
                      forall a t . (Tensor t, Fullfilled "__backward_pdf_normal" t a) =>
                        ArgsHMap "__backward_pdf_normal" t a -> TensorApply t
__backward_pdf_normal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_pdf_normal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_pdf_poisson" t = '[]

__backward_pdf_poisson ::
                       forall a t . (Tensor t, Fullfilled "__backward_pdf_poisson" t a) =>
                         ArgsHMap "__backward_pdf_poisson" t a -> TensorApply t
__backward_pdf_poisson args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_pdf_poisson" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_pdf_uniform" t = '[]

__backward_pdf_uniform ::
                       forall a t . (Tensor t, Fullfilled "__backward_pdf_uniform" t a) =>
                         ArgsHMap "__backward_pdf_uniform" t a -> TensorApply t
__backward_pdf_uniform args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_pdf_uniform" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_pick" t = '[]

__backward_pick ::
                forall a t . (Tensor t, Fullfilled "__backward_pick" t a) =>
                  ArgsHMap "__backward_pick" t a -> TensorApply t
__backward_pick args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_pick" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_power" t = '[]

__backward_power ::
                 forall a t . (Tensor t, Fullfilled "__backward_power" t a) =>
                   ArgsHMap "__backward_power" t a -> TensorApply t
__backward_power args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_power" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_power_scalar" t =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt t),
        '("rhs", AttrOpt t)]

__backward_power_scalar ::
                        forall a t .
                          (Tensor t, Fullfilled "__backward_power_scalar" t a) =>
                          ArgsHMap "__backward_power_scalar" t a -> TensorApply t
__backward_power_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_power_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_prod" t = '[]

__backward_prod ::
                forall a t . (Tensor t, Fullfilled "__backward_prod" t a) =>
                  ArgsHMap "__backward_prod" t a -> TensorApply t
__backward_prod args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_prod" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_radians" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_radians ::
                   forall a t . (Tensor t, Fullfilled "__backward_radians" t a) =>
                     ArgsHMap "__backward_radians" t a -> TensorApply t
__backward_radians args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_radians" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_rcbrt" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_rcbrt ::
                 forall a t . (Tensor t, Fullfilled "__backward_rcbrt" t a) =>
                   ArgsHMap "__backward_rcbrt" t a -> TensorApply t
__backward_rcbrt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_rcbrt" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_rdiv_scalar" t =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt t),
        '("rhs", AttrOpt t)]

__backward_rdiv_scalar ::
                       forall a t . (Tensor t, Fullfilled "__backward_rdiv_scalar" t a) =>
                         ArgsHMap "__backward_rdiv_scalar" t a -> TensorApply t
__backward_rdiv_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_rdiv_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_reciprocal" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_reciprocal ::
                      forall a t . (Tensor t, Fullfilled "__backward_reciprocal" t a) =>
                        ArgsHMap "__backward_reciprocal" t a -> TensorApply t
__backward_reciprocal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_reciprocal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_relu" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_relu ::
                forall a t . (Tensor t, Fullfilled "__backward_relu" t a) =>
                  ArgsHMap "__backward_relu" t a -> TensorApply t
__backward_relu args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_relu" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_repeat" t = '[]

__backward_repeat ::
                  forall a t . (Tensor t, Fullfilled "__backward_repeat" t a) =>
                    ArgsHMap "__backward_repeat" t a -> TensorApply t
__backward_repeat args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_repeat" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_reshape" t = '[]

__backward_reshape ::
                   forall a t . (Tensor t, Fullfilled "__backward_reshape" t a) =>
                     ArgsHMap "__backward_reshape" t a -> TensorApply t
__backward_reshape args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_reshape" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_reverse" t = '[]

__backward_reverse ::
                   forall a t . (Tensor t, Fullfilled "__backward_reverse" t a) =>
                     ArgsHMap "__backward_reverse" t a -> TensorApply t
__backward_reverse args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_reverse" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_rmod_scalar" t =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt t),
        '("rhs", AttrOpt t)]

__backward_rmod_scalar ::
                       forall a t . (Tensor t, Fullfilled "__backward_rmod_scalar" t a) =>
                         ArgsHMap "__backward_rmod_scalar" t a -> TensorApply t
__backward_rmod_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_rmod_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_rpower_scalar" t =
     '[ '("scalar", AttrOpt Float), '("lhs", AttrOpt t),
        '("rhs", AttrOpt t)]

__backward_rpower_scalar ::
                         forall a t .
                           (Tensor t, Fullfilled "__backward_rpower_scalar" t a) =>
                           ArgsHMap "__backward_rpower_scalar" t a -> TensorApply t
__backward_rpower_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_rpower_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_rsqrt" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_rsqrt ::
                 forall a t . (Tensor t, Fullfilled "__backward_rsqrt" t a) =>
                   ArgsHMap "__backward_rsqrt" t a -> TensorApply t
__backward_rsqrt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_rsqrt" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_sample_multinomial" t = '[]

__backward_sample_multinomial ::
                              forall a t .
                                (Tensor t, Fullfilled "__backward_sample_multinomial" t a) =>
                                ArgsHMap "__backward_sample_multinomial" t a -> TensorApply t
__backward_sample_multinomial args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_sample_multinomial" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward_sigmoid" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_sigmoid ::
                   forall a t . (Tensor t, Fullfilled "__backward_sigmoid" t a) =>
                     ArgsHMap "__backward_sigmoid" t a -> TensorApply t
__backward_sigmoid args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_sigmoid" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_sign" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_sign ::
                forall a t . (Tensor t, Fullfilled "__backward_sign" t a) =>
                  ArgsHMap "__backward_sign" t a -> TensorApply t
__backward_sign args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_sign" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_sin" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_sin ::
               forall a t . (Tensor t, Fullfilled "__backward_sin" t a) =>
                 ArgsHMap "__backward_sin" t a -> TensorApply t
__backward_sin args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_sin" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_sinh" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_sinh ::
                forall a t . (Tensor t, Fullfilled "__backward_sinh" t a) =>
                  ArgsHMap "__backward_sinh" t a -> TensorApply t
__backward_sinh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_sinh" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_slice" t = '[]

__backward_slice ::
                 forall a t . (Tensor t, Fullfilled "__backward_slice" t a) =>
                   ArgsHMap "__backward_slice" t a -> TensorApply t
__backward_slice args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_slice" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_slice_axis" t = '[]

__backward_slice_axis ::
                      forall a t . (Tensor t, Fullfilled "__backward_slice_axis" t a) =>
                        ArgsHMap "__backward_slice_axis" t a -> TensorApply t
__backward_slice_axis args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_slice_axis" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_slice_like" t = '[]

__backward_slice_like ::
                      forall a t . (Tensor t, Fullfilled "__backward_slice_like" t a) =>
                        ArgsHMap "__backward_slice_like" t a -> TensorApply t
__backward_slice_like args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_slice_like" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_smooth_l1" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_smooth_l1 ::
                     forall a t . (Tensor t, Fullfilled "__backward_smooth_l1" t a) =>
                       ArgsHMap "__backward_smooth_l1" t a -> TensorApply t
__backward_smooth_l1 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_smooth_l1" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_softmax" t =
     '[ '("args", AttrOpt [t])]

__backward_softmax ::
                   forall a t . (Tensor t, Fullfilled "__backward_softmax" t a) =>
                     ArgsHMap "__backward_softmax" t a -> TensorApply t
__backward_softmax args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #args :: Maybe [t])
      in apply "_backward_softmax" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__backward_softmax_cross_entropy" t =
     '[]

__backward_softmax_cross_entropy ::
                                 forall a t .
                                   (Tensor t, Fullfilled "__backward_softmax_cross_entropy" t a) =>
                                   ArgsHMap "__backward_softmax_cross_entropy" t a -> TensorApply t
__backward_softmax_cross_entropy args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_backward_softmax_cross_entropy" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__backward_softmin" t =
     '[ '("args", AttrOpt [t])]

__backward_softmin ::
                   forall a t . (Tensor t, Fullfilled "__backward_softmin" t a) =>
                     ArgsHMap "__backward_softmin" t a -> TensorApply t
__backward_softmin args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #args :: Maybe [t])
      in apply "_backward_softmin" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__backward_softsign" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_softsign ::
                    forall a t . (Tensor t, Fullfilled "__backward_softsign" t a) =>
                      ArgsHMap "__backward_softsign" t a -> TensorApply t
__backward_softsign args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_softsign" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_sparse_retain" t = '[]

__backward_sparse_retain ::
                         forall a t .
                           (Tensor t, Fullfilled "__backward_sparse_retain" t a) =>
                           ArgsHMap "__backward_sparse_retain" t a -> TensorApply t
__backward_sparse_retain args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_sparse_retain" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_sqrt" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_sqrt ::
                forall a t . (Tensor t, Fullfilled "__backward_sqrt" t a) =>
                  ArgsHMap "__backward_sqrt" t a -> TensorApply t
__backward_sqrt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_sqrt" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_square" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_square ::
                  forall a t . (Tensor t, Fullfilled "__backward_square" t a) =>
                    ArgsHMap "__backward_square" t a -> TensorApply t
__backward_square args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_square" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_square_sum" t = '[]

__backward_square_sum ::
                      forall a t . (Tensor t, Fullfilled "__backward_square_sum" t a) =>
                        ArgsHMap "__backward_square_sum" t a -> TensorApply t
__backward_square_sum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_square_sum" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_squeeze" t = '[]

__backward_squeeze ::
                   forall a t . (Tensor t, Fullfilled "__backward_squeeze" t a) =>
                     ArgsHMap "__backward_squeeze" t a -> TensorApply t
__backward_squeeze args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_squeeze" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_stack" t = '[]

__backward_stack ::
                 forall a t . (Tensor t, Fullfilled "__backward_stack" t a) =>
                   ArgsHMap "__backward_stack" t a -> TensorApply t
__backward_stack args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_stack" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_sub" t = '[]

__backward_sub ::
               forall a t . (Tensor t, Fullfilled "__backward_sub" t a) =>
                 ArgsHMap "__backward_sub" t a -> TensorApply t
__backward_sub args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_sub" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_sum" t = '[]

__backward_sum ::
               forall a t . (Tensor t, Fullfilled "__backward_sum" t a) =>
                 ArgsHMap "__backward_sum" t a -> TensorApply t
__backward_sum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_sum" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_take" t = '[]

__backward_take ::
                forall a t . (Tensor t, Fullfilled "__backward_take" t a) =>
                  ArgsHMap "__backward_take" t a -> TensorApply t
__backward_take args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_take" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_tan" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_tan ::
               forall a t . (Tensor t, Fullfilled "__backward_tan" t a) =>
                 ArgsHMap "__backward_tan" t a -> TensorApply t
__backward_tan args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_tan" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_tanh" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__backward_tanh ::
                forall a t . (Tensor t, Fullfilled "__backward_tanh" t a) =>
                  ArgsHMap "__backward_tanh" t a -> TensorApply t
__backward_tanh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_backward_tanh" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_tile" t = '[]

__backward_tile ::
                forall a t . (Tensor t, Fullfilled "__backward_tile" t a) =>
                  ArgsHMap "__backward_tile" t a -> TensorApply t
__backward_tile args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_tile" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_topk" t = '[]

__backward_topk ::
                forall a t . (Tensor t, Fullfilled "__backward_topk" t a) =>
                  ArgsHMap "__backward_topk" t a -> TensorApply t
__backward_topk args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_topk" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_tril" t = '[]

__backward_tril ::
                forall a t . (Tensor t, Fullfilled "__backward_tril" t a) =>
                  ArgsHMap "__backward_tril" t a -> TensorApply t
__backward_tril args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_tril" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_where" t = '[]

__backward_where ::
                 forall a t . (Tensor t, Fullfilled "__backward_where" t a) =>
                   ArgsHMap "__backward_where" t a -> TensorApply t
__backward_where args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_where" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__backward_while_loop" t = '[]

__backward_while_loop ::
                      forall a t . (Tensor t, Fullfilled "__backward_while_loop" t a) =>
                        ArgsHMap "__backward_while_loop" t a -> TensorApply t
__backward_while_loop args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_backward_while_loop" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__broadcast_backward" t = '[]

__broadcast_backward ::
                     forall a t . (Tensor t, Fullfilled "__broadcast_backward" t a) =>
                       ArgsHMap "__broadcast_backward" t a -> TensorApply t
__broadcast_backward args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_broadcast_backward" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_AdaptiveAvgPooling2D" t =
     '[ '("output_size", AttrOpt [Int]), '("data", AttrOpt t)]

__contrib_AdaptiveAvgPooling2D ::
                               forall a t .
                                 (Tensor t, Fullfilled "__contrib_AdaptiveAvgPooling2D" t a) =>
                                 ArgsHMap "__contrib_AdaptiveAvgPooling2D" t a -> TensorApply t
__contrib_AdaptiveAvgPooling2D args
  = let scalarArgs
          = catMaybes
              [("output_size",) . showValue <$>
                 (args !? #output_size :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in
      apply "_contrib_AdaptiveAvgPooling2D" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__contrib_BilinearResize2D" t =
     '[ '("height", AttrOpt Int), '("width", AttrOpt Int),
        '("scale_height", AttrOpt (Maybe Float)),
        '("scale_width", AttrOpt (Maybe Float)),
        '("mode",
          AttrOpt
            (EnumType
               '["like", "odd_scale", "size", "to_even_down", "to_even_up",
                 "to_odd_down", "to_odd_up"])),
        '("data", AttrOpt t), '("like", AttrOpt t)]

__contrib_BilinearResize2D ::
                           forall a t .
                             (Tensor t, Fullfilled "__contrib_BilinearResize2D" t a) =>
                             ArgsHMap "__contrib_BilinearResize2D" t a -> TensorApply t
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
                           "to_odd_down", "to_odd_up"]))]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("like",) <$> (args !? #like :: Maybe t)]
      in
      apply "_contrib_BilinearResize2D" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_DeformableConvolution" t =
     '[ '("kernel", AttrReq [Int]), '("stride", AttrOpt [Int]),
        '("dilate", AttrOpt [Int]), '("pad", AttrOpt [Int]),
        '("num_filter", AttrReq Int), '("num_group", AttrOpt Int),
        '("num_deformable_group", AttrOpt Int),
        '("workspace", AttrOpt Int), '("no_bias", AttrOpt Bool),
        '("layout", AttrOpt (Maybe (EnumType '["NCDHW", "NCHW", "NCW"]))),
        '("data", AttrOpt t), '("offset", AttrOpt t),
        '("weight", AttrOpt t), '("bias", AttrOpt t)]

__contrib_DeformableConvolution ::
                                forall a t .
                                  (Tensor t, Fullfilled "__contrib_DeformableConvolution" t a) =>
                                  ArgsHMap "__contrib_DeformableConvolution" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("offset",) <$> (args !? #offset :: Maybe t),
               ("weight",) <$> (args !? #weight :: Maybe t),
               ("bias",) <$> (args !? #bias :: Maybe t)]
      in
      apply "_contrib_DeformableConvolution" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__contrib_DeformablePSROIPooling" t =
     '[ '("spatial_scale", AttrReq Float), '("output_dim", AttrReq Int),
        '("group_size", AttrReq Int), '("pooled_size", AttrReq Int),
        '("part_size", AttrOpt Int), '("sample_per_part", AttrOpt Int),
        '("trans_std", AttrOpt Float), '("no_trans", AttrOpt Bool),
        '("data", AttrOpt t), '("rois", AttrOpt t), '("trans", AttrOpt t)]

__contrib_DeformablePSROIPooling ::
                                 forall a t .
                                   (Tensor t, Fullfilled "__contrib_DeformablePSROIPooling" t a) =>
                                   ArgsHMap "__contrib_DeformablePSROIPooling" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("rois",) <$> (args !? #rois :: Maybe t),
               ("trans",) <$> (args !? #trans :: Maybe t)]
      in
      apply "_contrib_DeformablePSROIPooling" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__contrib_MultiBoxDetection" t =
     '[ '("clip", AttrOpt Bool), '("threshold", AttrOpt Float),
        '("background_id", AttrOpt Int), '("nms_threshold", AttrOpt Float),
        '("force_suppress", AttrOpt Bool), '("variances", AttrOpt [Float]),
        '("nms_topk", AttrOpt Int), '("cls_prob", AttrOpt t),
        '("loc_pred", AttrOpt t), '("anchor", AttrOpt t)]

__contrib_MultiBoxDetection ::
                            forall a t .
                              (Tensor t, Fullfilled "__contrib_MultiBoxDetection" t a) =>
                              ArgsHMap "__contrib_MultiBoxDetection" t a -> TensorApply t
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
              [("cls_prob",) <$> (args !? #cls_prob :: Maybe t),
               ("loc_pred",) <$> (args !? #loc_pred :: Maybe t),
               ("anchor",) <$> (args !? #anchor :: Maybe t)]
      in
      apply "_contrib_MultiBoxDetection" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_MultiBoxPrior" t =
     '[ '("sizes", AttrOpt [Float]), '("ratios", AttrOpt [Float]),
        '("clip", AttrOpt Bool), '("steps", AttrOpt [Float]),
        '("offsets", AttrOpt [Float]), '("data", AttrOpt t)]

__contrib_MultiBoxPrior ::
                        forall a t .
                          (Tensor t, Fullfilled "__contrib_MultiBoxPrior" t a) =>
                          ArgsHMap "__contrib_MultiBoxPrior" t a -> TensorApply t
__contrib_MultiBoxPrior args
  = let scalarArgs
          = catMaybes
              [("sizes",) . showValue <$> (args !? #sizes :: Maybe [Float]),
               ("ratios",) . showValue <$> (args !? #ratios :: Maybe [Float]),
               ("clip",) . showValue <$> (args !? #clip :: Maybe Bool),
               ("steps",) . showValue <$> (args !? #steps :: Maybe [Float]),
               ("offsets",) . showValue <$> (args !? #offsets :: Maybe [Float])]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_contrib_MultiBoxPrior" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_MultiBoxTarget" t =
     '[ '("overlap_threshold", AttrOpt Float),
        '("ignore_label", AttrOpt Float),
        '("negative_mining_ratio", AttrOpt Float),
        '("negative_mining_thresh", AttrOpt Float),
        '("minimum_negative_samples", AttrOpt Int),
        '("variances", AttrOpt [Float]), '("anchor", AttrOpt t),
        '("label", AttrOpt t), '("cls_pred", AttrOpt t)]

__contrib_MultiBoxTarget ::
                         forall a t .
                           (Tensor t, Fullfilled "__contrib_MultiBoxTarget" t a) =>
                           ArgsHMap "__contrib_MultiBoxTarget" t a -> TensorApply t
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
              [("anchor",) <$> (args !? #anchor :: Maybe t),
               ("label",) <$> (args !? #label :: Maybe t),
               ("cls_pred",) <$> (args !? #cls_pred :: Maybe t)]
      in apply "_contrib_MultiBoxTarget" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_MultiProposal" t =
     '[ '("rpn_pre_nms_top_n", AttrOpt Int),
        '("rpn_post_nms_top_n", AttrOpt Int),
        '("threshold", AttrOpt Float), '("rpn_min_size", AttrOpt Int),
        '("scales", AttrOpt [Float]), '("ratios", AttrOpt [Float]),
        '("feature_stride", AttrOpt Int), '("output_score", AttrOpt Bool),
        '("iou_loss", AttrOpt Bool), '("cls_prob", AttrOpt t),
        '("bbox_pred", AttrOpt t), '("im_info", AttrOpt t)]

__contrib_MultiProposal ::
                        forall a t .
                          (Tensor t, Fullfilled "__contrib_MultiProposal" t a) =>
                          ArgsHMap "__contrib_MultiProposal" t a -> TensorApply t
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
              [("cls_prob",) <$> (args !? #cls_prob :: Maybe t),
               ("bbox_pred",) <$> (args !? #bbox_pred :: Maybe t),
               ("im_info",) <$> (args !? #im_info :: Maybe t)]
      in apply "_contrib_MultiProposal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_PSROIPooling" t =
     '[ '("spatial_scale", AttrReq Float), '("output_dim", AttrReq Int),
        '("pooled_size", AttrReq Int), '("group_size", AttrOpt Int),
        '("data", AttrOpt t), '("rois", AttrOpt t)]

__contrib_PSROIPooling ::
                       forall a t . (Tensor t, Fullfilled "__contrib_PSROIPooling" t a) =>
                         ArgsHMap "__contrib_PSROIPooling" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("rois",) <$> (args !? #rois :: Maybe t)]
      in apply "_contrib_PSROIPooling" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_Proposal" t =
     '[ '("rpn_pre_nms_top_n", AttrOpt Int),
        '("rpn_post_nms_top_n", AttrOpt Int),
        '("threshold", AttrOpt Float), '("rpn_min_size", AttrOpt Int),
        '("scales", AttrOpt [Float]), '("ratios", AttrOpt [Float]),
        '("feature_stride", AttrOpt Int), '("output_score", AttrOpt Bool),
        '("iou_loss", AttrOpt Bool), '("cls_prob", AttrOpt t),
        '("bbox_pred", AttrOpt t), '("im_info", AttrOpt t)]

__contrib_Proposal ::
                   forall a t . (Tensor t, Fullfilled "__contrib_Proposal" t a) =>
                     ArgsHMap "__contrib_Proposal" t a -> TensorApply t
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
              [("cls_prob",) <$> (args !? #cls_prob :: Maybe t),
               ("bbox_pred",) <$> (args !? #bbox_pred :: Maybe t),
               ("im_info",) <$> (args !? #im_info :: Maybe t)]
      in apply "_contrib_Proposal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_ROIAlign" t =
     '[ '("pooled_size", AttrReq [Int]),
        '("spatial_scale", AttrReq Float), '("sample_ratio", AttrOpt Int),
        '("position_sensitive", AttrOpt Bool), '("data", AttrOpt t),
        '("rois", AttrOpt t)]

__contrib_ROIAlign ::
                   forall a t . (Tensor t, Fullfilled "__contrib_ROIAlign" t a) =>
                     ArgsHMap "__contrib_ROIAlign" t a -> TensorApply t
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
                 (args !? #position_sensitive :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("rois",) <$> (args !? #rois :: Maybe t)]
      in apply "_contrib_ROIAlign" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_RROIAlign" t =
     '[ '("pooled_size", AttrReq [Int]),
        '("spatial_scale", AttrReq Float),
        '("sampling_ratio", AttrOpt Int), '("data", AttrOpt t),
        '("rois", AttrOpt t)]

__contrib_RROIAlign ::
                    forall a t . (Tensor t, Fullfilled "__contrib_RROIAlign" t a) =>
                      ArgsHMap "__contrib_RROIAlign" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("rois",) <$> (args !? #rois :: Maybe t)]
      in apply "_contrib_RROIAlign" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_SparseEmbedding" t =
     '[ '("input_dim", AttrReq Int), '("output_dim", AttrReq Int),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"])),
        '("sparse_grad", AttrOpt Bool), '("data", AttrOpt t),
        '("weight", AttrOpt t)]

__contrib_SparseEmbedding ::
                          forall a t .
                            (Tensor t, Fullfilled "__contrib_SparseEmbedding" t a) =>
                            ArgsHMap "__contrib_SparseEmbedding" t a -> TensorApply t
__contrib_SparseEmbedding args
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
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("weight",) <$> (args !? #weight :: Maybe t)]
      in apply "_contrib_SparseEmbedding" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_SyncBatchNorm" t =
     '[ '("eps", AttrOpt Float), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool), '("ndev", AttrOpt Int),
        '("key", AttrReq Text), '("data", AttrOpt t),
        '("gamma", AttrOpt t), '("beta", AttrOpt t),
        '("moving_mean", AttrOpt t), '("moving_var", AttrOpt t)]

__contrib_SyncBatchNorm ::
                        forall a t .
                          (Tensor t, Fullfilled "__contrib_SyncBatchNorm" t a) =>
                          ArgsHMap "__contrib_SyncBatchNorm" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("gamma",) <$> (args !? #gamma :: Maybe t),
               ("beta",) <$> (args !? #beta :: Maybe t),
               ("moving_mean",) <$> (args !? #moving_mean :: Maybe t),
               ("moving_var",) <$> (args !? #moving_var :: Maybe t)]
      in apply "_contrib_SyncBatchNorm" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_allclose" t =
     '[ '("rtol", AttrOpt Float), '("atol", AttrOpt Float),
        '("equal_nan", AttrOpt Bool), '("a", AttrOpt t), '("b", AttrOpt t)]

__contrib_allclose ::
                   forall a t . (Tensor t, Fullfilled "__contrib_allclose" t a) =>
                     ArgsHMap "__contrib_allclose" t a -> TensorApply t
__contrib_allclose args
  = let scalarArgs
          = catMaybes
              [("rtol",) . showValue <$> (args !? #rtol :: Maybe Float),
               ("atol",) . showValue <$> (args !? #atol :: Maybe Float),
               ("equal_nan",) . showValue <$> (args !? #equal_nan :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("a",) <$> (args !? #a :: Maybe t),
               ("b",) <$> (args !? #b :: Maybe t)]
      in apply "_contrib_allclose" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_arange_like" t =
     '[ '("start", AttrOpt Double), '("step", AttrOpt Double),
        '("repeat", AttrOpt Int), '("ctx", AttrOpt Text),
        '("axis", AttrOpt (Maybe Int)), '("data", AttrOpt t)]

__contrib_arange_like ::
                      forall a t . (Tensor t, Fullfilled "__contrib_arange_like" t a) =>
                        ArgsHMap "__contrib_arange_like" t a -> TensorApply t
__contrib_arange_like args
  = let scalarArgs
          = catMaybes
              [("start",) . showValue <$> (args !? #start :: Maybe Double),
               ("step",) . showValue <$> (args !? #step :: Maybe Double),
               ("repeat",) . showValue <$> (args !? #repeat :: Maybe Int),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int))]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_contrib_arange_like" scalarArgs (Left tensorKeyArgs)

type instance
     ParameterList "__contrib_backward_gradientmultiplier" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__contrib_backward_gradientmultiplier ::
                                      forall a t .
                                        (Tensor t,
                                         Fullfilled "__contrib_backward_gradientmultiplier" t a) =>
                                        ArgsHMap "__contrib_backward_gradientmultiplier" t a ->
                                          TensorApply t
__contrib_backward_gradientmultiplier args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in
      apply "_contrib_backward_gradientmultiplier" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__contrib_backward_hawkesll" t = '[]

__contrib_backward_hawkesll ::
                            forall a t .
                              (Tensor t, Fullfilled "__contrib_backward_hawkesll" t a) =>
                              ArgsHMap "__contrib_backward_hawkesll" t a -> TensorApply t
__contrib_backward_hawkesll args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_contrib_backward_hawkesll" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_backward_index_copy" t = '[]

__contrib_backward_index_copy ::
                              forall a t .
                                (Tensor t, Fullfilled "__contrib_backward_index_copy" t a) =>
                                ArgsHMap "__contrib_backward_index_copy" t a -> TensorApply t
__contrib_backward_index_copy args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_contrib_backward_index_copy" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__contrib_backward_quadratic" t = '[]

__contrib_backward_quadratic ::
                             forall a t .
                               (Tensor t, Fullfilled "__contrib_backward_quadratic" t a) =>
                               ArgsHMap "__contrib_backward_quadratic" t a -> TensorApply t
__contrib_backward_quadratic args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_contrib_backward_quadratic" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_bipartite_matching" t =
     '[ '("is_ascend", AttrOpt Bool), '("threshold", AttrReq Float),
        '("topk", AttrOpt Int), '("data", AttrOpt t)]

__contrib_bipartite_matching ::
                             forall a t .
                               (Tensor t, Fullfilled "__contrib_bipartite_matching" t a) =>
                               ArgsHMap "__contrib_bipartite_matching" t a -> TensorApply t
__contrib_bipartite_matching args
  = let scalarArgs
          = catMaybes
              [("is_ascend",) . showValue <$> (args !? #is_ascend :: Maybe Bool),
               ("threshold",) . showValue <$> (args !? #threshold :: Maybe Float),
               ("topk",) . showValue <$> (args !? #topk :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in
      apply "_contrib_bipartite_matching" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_boolean_mask" t =
     '[ '("axis", AttrOpt Int), '("data", AttrOpt t),
        '("index", AttrOpt t)]

__contrib_boolean_mask ::
                       forall a t . (Tensor t, Fullfilled "__contrib_boolean_mask" t a) =>
                         ArgsHMap "__contrib_boolean_mask" t a -> TensorApply t
__contrib_boolean_mask args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("index",) <$> (args !? #index :: Maybe t)]
      in apply "_contrib_boolean_mask" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_box_decode" t =
     '[ '("std0", AttrOpt Float), '("std1", AttrOpt Float),
        '("std2", AttrOpt Float), '("std3", AttrOpt Float),
        '("clip", AttrOpt Float),
        '("format", AttrOpt (EnumType '["center", "corner"])),
        '("data", AttrOpt t), '("anchors", AttrOpt t)]

__contrib_box_decode ::
                     forall a t . (Tensor t, Fullfilled "__contrib_box_decode" t a) =>
                       ArgsHMap "__contrib_box_decode" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("anchors",) <$> (args !? #anchors :: Maybe t)]
      in apply "_contrib_box_decode" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_box_encode" t =
     '[ '("samples", AttrOpt t), '("matches", AttrOpt t),
        '("anchors", AttrOpt t), '("refs", AttrOpt t),
        '("means", AttrOpt t), '("stds", AttrOpt t)]

__contrib_box_encode ::
                     forall a t . (Tensor t, Fullfilled "__contrib_box_encode" t a) =>
                       ArgsHMap "__contrib_box_encode" t a -> TensorApply t
__contrib_box_encode args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("samples",) <$> (args !? #samples :: Maybe t),
               ("matches",) <$> (args !? #matches :: Maybe t),
               ("anchors",) <$> (args !? #anchors :: Maybe t),
               ("refs",) <$> (args !? #refs :: Maybe t),
               ("means",) <$> (args !? #means :: Maybe t),
               ("stds",) <$> (args !? #stds :: Maybe t)]
      in apply "_contrib_box_encode" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_box_iou" t =
     '[ '("format", AttrOpt (EnumType '["center", "corner"])),
        '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__contrib_box_iou ::
                  forall a t . (Tensor t, Fullfilled "__contrib_box_iou" t a) =>
                    ArgsHMap "__contrib_box_iou" t a -> TensorApply t
__contrib_box_iou args
  = let scalarArgs
          = catMaybes
              [("format",) . showValue <$>
                 (args !? #format :: Maybe (EnumType '["center", "corner"]))]
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_contrib_box_iou" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_box_nms" t =
     '[ '("overlap_thresh", AttrOpt Float),
        '("valid_thresh", AttrOpt Float), '("topk", AttrOpt Int),
        '("coord_start", AttrOpt Int), '("score_index", AttrOpt Int),
        '("id_index", AttrOpt Int), '("background_id", AttrOpt Int),
        '("force_suppress", AttrOpt Bool),
        '("in_format", AttrOpt (EnumType '["center", "corner"])),
        '("out_format", AttrOpt (EnumType '["center", "corner"])),
        '("data", AttrOpt t)]

__contrib_box_nms ::
                  forall a t . (Tensor t, Fullfilled "__contrib_box_nms" t a) =>
                    ArgsHMap "__contrib_box_nms" t a -> TensorApply t
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
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_contrib_box_nms" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_calibrate_entropy" t =
     '[ '("num_quantized_bins", AttrOpt Int), '("hist", AttrOpt t),
        '("hist_edges", AttrOpt t)]

__contrib_calibrate_entropy ::
                            forall a t .
                              (Tensor t, Fullfilled "__contrib_calibrate_entropy" t a) =>
                              ArgsHMap "__contrib_calibrate_entropy" t a -> TensorApply t
__contrib_calibrate_entropy args
  = let scalarArgs
          = catMaybes
              [("num_quantized_bins",) . showValue <$>
                 (args !? #num_quantized_bins :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("hist",) <$> (args !? #hist :: Maybe t),
               ("hist_edges",) <$> (args !? #hist_edges :: Maybe t)]
      in
      apply "_contrib_calibrate_entropy" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_count_sketch" t =
     '[ '("out_dim", AttrReq Int),
        '("processing_batch_size", AttrOpt Int), '("data", AttrOpt t),
        '("h", AttrOpt t), '("s", AttrOpt t)]

__contrib_count_sketch ::
                       forall a t . (Tensor t, Fullfilled "__contrib_count_sketch" t a) =>
                         ArgsHMap "__contrib_count_sketch" t a -> TensorApply t
__contrib_count_sketch args
  = let scalarArgs
          = catMaybes
              [("out_dim",) . showValue <$> (args !? #out_dim :: Maybe Int),
               ("processing_batch_size",) . showValue <$>
                 (args !? #processing_batch_size :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("h",) <$> (args !? #h :: Maybe t),
               ("s",) <$> (args !? #s :: Maybe t)]
      in apply "_contrib_count_sketch" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_dequantize" t =
     '[ '("out_type", AttrOpt (EnumType '["float32"])),
        '("data", AttrOpt t), '("min_range", AttrOpt t),
        '("max_range", AttrOpt t)]

__contrib_dequantize ::
                     forall a t . (Tensor t, Fullfilled "__contrib_dequantize" t a) =>
                       ArgsHMap "__contrib_dequantize" t a -> TensorApply t
__contrib_dequantize args
  = let scalarArgs
          = catMaybes
              [("out_type",) . showValue <$>
                 (args !? #out_type :: Maybe (EnumType '["float32"]))]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("min_range",) <$> (args !? #min_range :: Maybe t),
               ("max_range",) <$> (args !? #max_range :: Maybe t)]
      in apply "_contrib_dequantize" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_dgl_adjacency" t =
     '[ '("data", AttrOpt t)]

__contrib_dgl_adjacency ::
                        forall a t .
                          (Tensor t, Fullfilled "__contrib_dgl_adjacency" t a) =>
                          ArgsHMap "__contrib_dgl_adjacency" t a -> TensorApply t
__contrib_dgl_adjacency args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_contrib_dgl_adjacency" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_div_sqrt_dim" t =
     '[ '("data", AttrOpt t)]

__contrib_div_sqrt_dim ::
                       forall a t . (Tensor t, Fullfilled "__contrib_div_sqrt_dim" t a) =>
                         ArgsHMap "__contrib_div_sqrt_dim" t a -> TensorApply t
__contrib_div_sqrt_dim args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_contrib_div_sqrt_dim" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_edge_id" t =
     '[ '("data", AttrOpt t), '("u", AttrOpt t), '("v", AttrOpt t)]

__contrib_edge_id ::
                  forall a t . (Tensor t, Fullfilled "__contrib_edge_id" t a) =>
                    ArgsHMap "__contrib_edge_id" t a -> TensorApply t
__contrib_edge_id args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("u",) <$> (args !? #u :: Maybe t),
               ("v",) <$> (args !? #v :: Maybe t)]
      in apply "_contrib_edge_id" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_fft" t =
     '[ '("compute_size", AttrOpt Int), '("data", AttrOpt t)]

__contrib_fft ::
              forall a t . (Tensor t, Fullfilled "__contrib_fft" t a) =>
                ArgsHMap "__contrib_fft" t a -> TensorApply t
__contrib_fft args
  = let scalarArgs
          = catMaybes
              [("compute_size",) . showValue <$>
                 (args !? #compute_size :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_contrib_fft" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_getnnz" t =
     '[ '("axis", AttrOpt (Maybe Int)), '("data", AttrOpt t)]

__contrib_getnnz ::
                 forall a t . (Tensor t, Fullfilled "__contrib_getnnz" t a) =>
                   ArgsHMap "__contrib_getnnz" t a -> TensorApply t
__contrib_getnnz args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int))]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_contrib_getnnz" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_gradientmultiplier" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__contrib_gradientmultiplier ::
                             forall a t .
                               (Tensor t, Fullfilled "__contrib_gradientmultiplier" t a) =>
                               ArgsHMap "__contrib_gradientmultiplier" t a -> TensorApply t
__contrib_gradientmultiplier args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in
      apply "_contrib_gradientmultiplier" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_group_adagrad_update" t =
     '[ '("lr", AttrReq Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("weight", AttrOpt t), '("grad", AttrOpt t),
        '("history", AttrOpt t)]

__contrib_group_adagrad_update ::
                               forall a t .
                                 (Tensor t, Fullfilled "__contrib_group_adagrad_update" t a) =>
                                 ArgsHMap "__contrib_group_adagrad_update" t a -> TensorApply t
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
              [("weight",) <$> (args !? #weight :: Maybe t),
               ("grad",) <$> (args !? #grad :: Maybe t),
               ("history",) <$> (args !? #history :: Maybe t)]
      in
      apply "_contrib_group_adagrad_update" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__contrib_hawkesll" t =
     '[ '("lda", AttrOpt t), '("alpha", AttrOpt t),
        '("beta", AttrOpt t), '("state", AttrOpt t), '("lags", AttrOpt t),
        '("marks", AttrOpt t), '("valid_length", AttrOpt t),
        '("max_time", AttrOpt t)]

__contrib_hawkesll ::
                   forall a t . (Tensor t, Fullfilled "__contrib_hawkesll" t a) =>
                     ArgsHMap "__contrib_hawkesll" t a -> TensorApply t
__contrib_hawkesll args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lda",) <$> (args !? #lda :: Maybe t),
               ("alpha",) <$> (args !? #alpha :: Maybe t),
               ("beta",) <$> (args !? #beta :: Maybe t),
               ("state",) <$> (args !? #state :: Maybe t),
               ("lags",) <$> (args !? #lags :: Maybe t),
               ("marks",) <$> (args !? #marks :: Maybe t),
               ("valid_length",) <$> (args !? #valid_length :: Maybe t),
               ("max_time",) <$> (args !? #max_time :: Maybe t)]
      in apply "_contrib_hawkesll" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_ifft" t =
     '[ '("compute_size", AttrOpt Int), '("data", AttrOpt t)]

__contrib_ifft ::
               forall a t . (Tensor t, Fullfilled "__contrib_ifft" t a) =>
                 ArgsHMap "__contrib_ifft" t a -> TensorApply t
__contrib_ifft args
  = let scalarArgs
          = catMaybes
              [("compute_size",) . showValue <$>
                 (args !? #compute_size :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_contrib_ifft" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_index_array" t =
     '[ '("axes", AttrOpt (Maybe [Int])), '("data", AttrOpt t)]

__contrib_index_array ::
                      forall a t . (Tensor t, Fullfilled "__contrib_index_array" t a) =>
                        ArgsHMap "__contrib_index_array" t a -> TensorApply t
__contrib_index_array args
  = let scalarArgs
          = catMaybes
              [("axes",) . showValue <$> (args !? #axes :: Maybe (Maybe [Int]))]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_contrib_index_array" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_index_copy" t =
     '[ '("old_tensor", AttrOpt t), '("index_vector", AttrOpt t),
        '("new_tensor", AttrOpt t)]

__contrib_index_copy ::
                     forall a t . (Tensor t, Fullfilled "__contrib_index_copy" t a) =>
                       ArgsHMap "__contrib_index_copy" t a -> TensorApply t
__contrib_index_copy args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("old_tensor",) <$> (args !? #old_tensor :: Maybe t),
               ("index_vector",) <$> (args !? #index_vector :: Maybe t),
               ("new_tensor",) <$> (args !? #new_tensor :: Maybe t)]
      in apply "_contrib_index_copy" scalarArgs (Left tensorKeyArgs)

type instance
     ParameterList "__contrib_interleaved_matmul_encdec_qk" t =
     '[ '("heads", AttrReq Int), '("queries", AttrOpt t),
        '("keys_values", AttrOpt t)]

__contrib_interleaved_matmul_encdec_qk ::
                                       forall a t .
                                         (Tensor t,
                                          Fullfilled "__contrib_interleaved_matmul_encdec_qk" t
                                            a) =>
                                         ArgsHMap "__contrib_interleaved_matmul_encdec_qk" t a ->
                                           TensorApply t
__contrib_interleaved_matmul_encdec_qk args
  = let scalarArgs
          = catMaybes
              [("heads",) . showValue <$> (args !? #heads :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("queries",) <$> (args !? #queries :: Maybe t),
               ("keys_values",) <$> (args !? #keys_values :: Maybe t)]
      in
      apply "_contrib_interleaved_matmul_encdec_qk" scalarArgs
        (Left tensorKeyArgs)

type instance
     ParameterList "__contrib_interleaved_matmul_encdec_valatt" t =
     '[ '("heads", AttrReq Int), '("keys_values", AttrOpt t),
        '("attention", AttrOpt t)]

__contrib_interleaved_matmul_encdec_valatt ::
                                           forall a t .
                                             (Tensor t,
                                              Fullfilled
                                                "__contrib_interleaved_matmul_encdec_valatt" t a) =>
                                             ArgsHMap "__contrib_interleaved_matmul_encdec_valatt" t
                                               a
                                               -> TensorApply t
__contrib_interleaved_matmul_encdec_valatt args
  = let scalarArgs
          = catMaybes
              [("heads",) . showValue <$> (args !? #heads :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("keys_values",) <$> (args !? #keys_values :: Maybe t),
               ("attention",) <$> (args !? #attention :: Maybe t)]
      in
      apply "_contrib_interleaved_matmul_encdec_valatt" scalarArgs
        (Left tensorKeyArgs)

type instance
     ParameterList "__contrib_interleaved_matmul_selfatt_qk" t =
     '[ '("heads", AttrReq Int), '("queries_keys_values", AttrOpt t)]

__contrib_interleaved_matmul_selfatt_qk ::
                                        forall a t .
                                          (Tensor t,
                                           Fullfilled "__contrib_interleaved_matmul_selfatt_qk" t
                                             a) =>
                                          ArgsHMap "__contrib_interleaved_matmul_selfatt_qk" t a ->
                                            TensorApply t
__contrib_interleaved_matmul_selfatt_qk args
  = let scalarArgs
          = catMaybes
              [("heads",) . showValue <$> (args !? #heads :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("queries_keys_values",) <$>
                 (args !? #queries_keys_values :: Maybe t)]
      in
      apply "_contrib_interleaved_matmul_selfatt_qk" scalarArgs
        (Left tensorKeyArgs)

type instance
     ParameterList "__contrib_interleaved_matmul_selfatt_valatt" t =
     '[ '("heads", AttrReq Int), '("queries_keys_values", AttrOpt t),
        '("attention", AttrOpt t)]

__contrib_interleaved_matmul_selfatt_valatt ::
                                            forall a t .
                                              (Tensor t,
                                               Fullfilled
                                                 "__contrib_interleaved_matmul_selfatt_valatt" t
                                                 a) =>
                                              ArgsHMap "__contrib_interleaved_matmul_selfatt_valatt"
                                                t
                                                a
                                                -> TensorApply t
__contrib_interleaved_matmul_selfatt_valatt args
  = let scalarArgs
          = catMaybes
              [("heads",) . showValue <$> (args !? #heads :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("queries_keys_values",) <$>
                 (args !? #queries_keys_values :: Maybe t),
               ("attention",) <$> (args !? #attention :: Maybe t)]
      in
      apply "_contrib_interleaved_matmul_selfatt_valatt" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__contrib_mrcnn_mask_target" t =
     '[ '("num_rois", AttrReq Int), '("num_classes", AttrReq Int),
        '("mask_size", AttrReq [Int]), '("sample_ratio", AttrOpt Int),
        '("rois", AttrOpt t), '("gt_masks", AttrOpt t),
        '("matches", AttrOpt t), '("cls_targets", AttrOpt t)]

__contrib_mrcnn_mask_target ::
                            forall a t .
                              (Tensor t, Fullfilled "__contrib_mrcnn_mask_target" t a) =>
                              ArgsHMap "__contrib_mrcnn_mask_target" t a -> TensorApply t
__contrib_mrcnn_mask_target args
  = let scalarArgs
          = catMaybes
              [("num_rois",) . showValue <$> (args !? #num_rois :: Maybe Int),
               ("num_classes",) . showValue <$>
                 (args !? #num_classes :: Maybe Int),
               ("mask_size",) . showValue <$> (args !? #mask_size :: Maybe [Int]),
               ("sample_ratio",) . showValue <$>
                 (args !? #sample_ratio :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("rois",) <$> (args !? #rois :: Maybe t),
               ("gt_masks",) <$> (args !? #gt_masks :: Maybe t),
               ("matches",) <$> (args !? #matches :: Maybe t),
               ("cls_targets",) <$> (args !? #cls_targets :: Maybe t)]
      in
      apply "_contrib_mrcnn_mask_target" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_quadratic" t =
     '[ '("a", AttrOpt Float), '("b", AttrOpt Float),
        '("c", AttrOpt Float), '("data", AttrOpt t)]

__contrib_quadratic ::
                    forall a t . (Tensor t, Fullfilled "__contrib_quadratic" t a) =>
                      ArgsHMap "__contrib_quadratic" t a -> TensorApply t
__contrib_quadratic args
  = let scalarArgs
          = catMaybes
              [("a",) . showValue <$> (args !? #a :: Maybe Float),
               ("b",) . showValue <$> (args !? #b :: Maybe Float),
               ("c",) . showValue <$> (args !? #c :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_contrib_quadratic" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_quantize" t =
     '[ '("out_type", AttrOpt (EnumType '["int8", "uint8"])),
        '("data", AttrOpt t), '("min_range", AttrOpt t),
        '("max_range", AttrOpt t)]

__contrib_quantize ::
                   forall a t . (Tensor t, Fullfilled "__contrib_quantize" t a) =>
                     ArgsHMap "__contrib_quantize" t a -> TensorApply t
__contrib_quantize args
  = let scalarArgs
          = catMaybes
              [("out_type",) . showValue <$>
                 (args !? #out_type :: Maybe (EnumType '["int8", "uint8"]))]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("min_range",) <$> (args !? #min_range :: Maybe t),
               ("max_range",) <$> (args !? #max_range :: Maybe t)]
      in apply "_contrib_quantize" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_quantize_v2" t =
     '[ '("out_type", AttrOpt (EnumType '["auto", "int8", "uint8"])),
        '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)), '("data", AttrOpt t)]

__contrib_quantize_v2 ::
                      forall a t . (Tensor t, Fullfilled "__contrib_quantize_v2" t a) =>
                        ArgsHMap "__contrib_quantize_v2" t a -> TensorApply t
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
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_contrib_quantize_v2" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_quantized_act" t =
     '[ '("act_type",
          AttrReq
            (EnumType '["relu", "sigmoid", "softrelu", "softsign", "tanh"])),
        '("data", AttrOpt t), '("min_data", AttrOpt t),
        '("max_data", AttrOpt t)]

__contrib_quantized_act ::
                        forall a t .
                          (Tensor t, Fullfilled "__contrib_quantized_act" t a) =>
                          ArgsHMap "__contrib_quantized_act" t a -> TensorApply t
__contrib_quantized_act args
  = let scalarArgs
          = catMaybes
              [("act_type",) . showValue <$>
                 (args !? #act_type ::
                    Maybe
                      (EnumType '["relu", "sigmoid", "softrelu", "softsign", "tanh"]))]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("min_data",) <$> (args !? #min_data :: Maybe t),
               ("max_data",) <$> (args !? #max_data :: Maybe t)]
      in apply "_contrib_quantized_act" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_quantized_batch_norm" t =
     '[ '("eps", AttrOpt Double), '("momentum", AttrOpt Float),
        '("fix_gamma", AttrOpt Bool), '("use_global_stats", AttrOpt Bool),
        '("output_mean_var", AttrOpt Bool), '("axis", AttrOpt Int),
        '("cudnn_off", AttrOpt Bool),
        '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)), '("data", AttrOpt t),
        '("gamma", AttrOpt t), '("beta", AttrOpt t),
        '("moving_mean", AttrOpt t), '("moving_var", AttrOpt t),
        '("min_data", AttrOpt t), '("max_data", AttrOpt t)]

__contrib_quantized_batch_norm ::
                               forall a t .
                                 (Tensor t, Fullfilled "__contrib_quantized_batch_norm" t a) =>
                                 ArgsHMap "__contrib_quantized_batch_norm" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("gamma",) <$> (args !? #gamma :: Maybe t),
               ("beta",) <$> (args !? #beta :: Maybe t),
               ("moving_mean",) <$> (args !? #moving_mean :: Maybe t),
               ("moving_var",) <$> (args !? #moving_var :: Maybe t),
               ("min_data",) <$> (args !? #min_data :: Maybe t),
               ("max_data",) <$> (args !? #max_data :: Maybe t)]
      in
      apply "_contrib_quantized_batch_norm" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__contrib_quantized_concat" t =
     '[ '("num_args", AttrReq Int), '("dim", AttrOpt Int),
        '("data", AttrOpt [t])]

__contrib_quantized_concat ::
                           forall a t .
                             (Tensor t, Fullfilled "__contrib_quantized_concat" t a) =>
                             ArgsHMap "__contrib_quantized_concat" t a -> TensorApply t
__contrib_quantized_concat args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("dim",) . showValue <$> (args !? #dim :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in
      apply "_contrib_quantized_concat" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__contrib_quantized_conv" t =
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
        '("data", AttrOpt t), '("weight", AttrOpt t), '("bias", AttrOpt t),
        '("min_data", AttrOpt t), '("max_data", AttrOpt t),
        '("min_weight", AttrOpt t), '("max_weight", AttrOpt t),
        '("min_bias", AttrOpt t), '("max_bias", AttrOpt t)]

__contrib_quantized_conv ::
                         forall a t .
                           (Tensor t, Fullfilled "__contrib_quantized_conv" t a) =>
                           ArgsHMap "__contrib_quantized_conv" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("weight",) <$> (args !? #weight :: Maybe t),
               ("bias",) <$> (args !? #bias :: Maybe t),
               ("min_data",) <$> (args !? #min_data :: Maybe t),
               ("max_data",) <$> (args !? #max_data :: Maybe t),
               ("min_weight",) <$> (args !? #min_weight :: Maybe t),
               ("max_weight",) <$> (args !? #max_weight :: Maybe t),
               ("min_bias",) <$> (args !? #min_bias :: Maybe t),
               ("max_bias",) <$> (args !? #max_bias :: Maybe t)]
      in apply "_contrib_quantized_conv" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_quantized_elemwise_add" t =
     '[ '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)), '("lhs", AttrOpt t),
        '("rhs", AttrOpt t), '("lhs_min", AttrOpt t),
        '("lhs_max", AttrOpt t), '("rhs_min", AttrOpt t),
        '("rhs_max", AttrOpt t)]

__contrib_quantized_elemwise_add ::
                                 forall a t .
                                   (Tensor t, Fullfilled "__contrib_quantized_elemwise_add" t a) =>
                                   ArgsHMap "__contrib_quantized_elemwise_add" t a -> TensorApply t
__contrib_quantized_elemwise_add args
  = let scalarArgs
          = catMaybes
              [("min_calib_range",) . showValue <$>
                 (args !? #min_calib_range :: Maybe (Maybe Float)),
               ("max_calib_range",) . showValue <$>
                 (args !? #max_calib_range :: Maybe (Maybe Float))]
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t),
               ("lhs_min",) <$> (args !? #lhs_min :: Maybe t),
               ("lhs_max",) <$> (args !? #lhs_max :: Maybe t),
               ("rhs_min",) <$> (args !? #rhs_min :: Maybe t),
               ("rhs_max",) <$> (args !? #rhs_max :: Maybe t)]
      in
      apply "_contrib_quantized_elemwise_add" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__contrib_quantized_flatten" t =
     '[ '("data", AttrOpt t), '("min_data", AttrOpt t),
        '("max_data", AttrOpt t)]

__contrib_quantized_flatten ::
                            forall a t .
                              (Tensor t, Fullfilled "__contrib_quantized_flatten" t a) =>
                              ArgsHMap "__contrib_quantized_flatten" t a -> TensorApply t
__contrib_quantized_flatten args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("min_data",) <$> (args !? #min_data :: Maybe t),
               ("max_data",) <$> (args !? #max_data :: Maybe t)]
      in
      apply "_contrib_quantized_flatten" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_quantized_fully_connected" t
     =
     '[ '("num_hidden", AttrReq Int), '("no_bias", AttrOpt Bool),
        '("flatten", AttrOpt Bool), '("data", AttrOpt t),
        '("weight", AttrOpt t), '("bias", AttrOpt t),
        '("min_data", AttrOpt t), '("max_data", AttrOpt t),
        '("min_weight", AttrOpt t), '("max_weight", AttrOpt t),
        '("min_bias", AttrOpt t), '("max_bias", AttrOpt t)]

__contrib_quantized_fully_connected ::
                                    forall a t .
                                      (Tensor t,
                                       Fullfilled "__contrib_quantized_fully_connected" t a) =>
                                      ArgsHMap "__contrib_quantized_fully_connected" t a ->
                                        TensorApply t
__contrib_quantized_fully_connected args
  = let scalarArgs
          = catMaybes
              [("num_hidden",) . showValue <$>
                 (args !? #num_hidden :: Maybe Int),
               ("no_bias",) . showValue <$> (args !? #no_bias :: Maybe Bool),
               ("flatten",) . showValue <$> (args !? #flatten :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("weight",) <$> (args !? #weight :: Maybe t),
               ("bias",) <$> (args !? #bias :: Maybe t),
               ("min_data",) <$> (args !? #min_data :: Maybe t),
               ("max_data",) <$> (args !? #max_data :: Maybe t),
               ("min_weight",) <$> (args !? #min_weight :: Maybe t),
               ("max_weight",) <$> (args !? #max_weight :: Maybe t),
               ("min_bias",) <$> (args !? #min_bias :: Maybe t),
               ("max_bias",) <$> (args !? #max_bias :: Maybe t)]
      in
      apply "_contrib_quantized_fully_connected" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__contrib_quantized_pooling" t =
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
        '("data", AttrOpt t), '("min_data", AttrOpt t),
        '("max_data", AttrOpt t)]

__contrib_quantized_pooling ::
                            forall a t .
                              (Tensor t, Fullfilled "__contrib_quantized_pooling" t a) =>
                              ArgsHMap "__contrib_quantized_pooling" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("min_data",) <$> (args !? #min_data :: Maybe t),
               ("max_data",) <$> (args !? #max_data :: Maybe t)]
      in
      apply "_contrib_quantized_pooling" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_requantize" t =
     '[ '("out_type", AttrOpt (EnumType '["auto", "int8", "uint8"])),
        '("min_calib_range", AttrOpt (Maybe Float)),
        '("max_calib_range", AttrOpt (Maybe Float)), '("data", AttrOpt t),
        '("min_range", AttrOpt t), '("max_range", AttrOpt t)]

__contrib_requantize ::
                     forall a t . (Tensor t, Fullfilled "__contrib_requantize" t a) =>
                       ArgsHMap "__contrib_requantize" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("min_range",) <$> (args !? #min_range :: Maybe t),
               ("max_range",) <$> (args !? #max_range :: Maybe t)]
      in apply "_contrib_requantize" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_round_ste" t =
     '[ '("data", AttrOpt t)]

__contrib_round_ste ::
                    forall a t . (Tensor t, Fullfilled "__contrib_round_ste" t a) =>
                      ArgsHMap "__contrib_round_ste" t a -> TensorApply t
__contrib_round_ste args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_contrib_round_ste" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__contrib_sign_ste" t =
     '[ '("data", AttrOpt t)]

__contrib_sign_ste ::
                   forall a t . (Tensor t, Fullfilled "__contrib_sign_ste" t a) =>
                     ArgsHMap "__contrib_sign_ste" t a -> TensorApply t
__contrib_sign_ste args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_contrib_sign_ste" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__copy" t = '[ '("data", AttrOpt t)]

__copy ::
       forall a t . (Tensor t, Fullfilled "__copy" t a) =>
         ArgsHMap "__copy" t a -> TensorApply t
__copy args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_copy" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__copyto" t = '[ '("data", AttrOpt t)]

__copyto ::
         forall a t . (Tensor t, Fullfilled "__copyto" t a) =>
           ArgsHMap "__copyto" t a -> TensorApply t
__copyto args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_copyto" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__cvcopyMakeBorder" t =
     '[ '("top", AttrReq Int), '("bot", AttrReq Int),
        '("left", AttrReq Int), '("right", AttrReq Int),
        '("type", AttrOpt Int), '("value", AttrOpt Double),
        '("values", AttrOpt [Double]), '("src", AttrOpt t)]

__cvcopyMakeBorder ::
                   forall a t . (Tensor t, Fullfilled "__cvcopyMakeBorder" t a) =>
                     ArgsHMap "__cvcopyMakeBorder" t a -> TensorApply t
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
        tensorKeyArgs = catMaybes [("src",) <$> (args !? #src :: Maybe t)]
      in apply "_cvcopyMakeBorder" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__cvimdecode" t =
     '[ '("flag", AttrOpt Int), '("to_rgb", AttrOpt Bool),
        '("buf", AttrOpt t)]

__cvimdecode ::
             forall a t . (Tensor t, Fullfilled "__cvimdecode" t a) =>
               ArgsHMap "__cvimdecode" t a -> TensorApply t
__cvimdecode args
  = let scalarArgs
          = catMaybes
              [("flag",) . showValue <$> (args !? #flag :: Maybe Int),
               ("to_rgb",) . showValue <$> (args !? #to_rgb :: Maybe Bool)]
        tensorKeyArgs = catMaybes [("buf",) <$> (args !? #buf :: Maybe t)]
      in apply "_cvimdecode" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__cvimread" t =
     '[ '("filename", AttrReq Text), '("flag", AttrOpt Int),
        '("to_rgb", AttrOpt Bool)]

__cvimread ::
           forall a t . (Tensor t, Fullfilled "__cvimread" t a) =>
             ArgsHMap "__cvimread" t a -> TensorApply t
__cvimread args
  = let scalarArgs
          = catMaybes
              [("filename",) . showValue <$> (args !? #filename :: Maybe Text),
               ("flag",) . showValue <$> (args !? #flag :: Maybe Int),
               ("to_rgb",) . showValue <$> (args !? #to_rgb :: Maybe Bool)]
        tensorKeyArgs = catMaybes []
      in apply "_cvimread" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__cvimresize" t =
     '[ '("w", AttrReq Int), '("h", AttrReq Int),
        '("interp", AttrOpt Int), '("src", AttrOpt t)]

__cvimresize ::
             forall a t . (Tensor t, Fullfilled "__cvimresize" t a) =>
               ArgsHMap "__cvimresize" t a -> TensorApply t
__cvimresize args
  = let scalarArgs
          = catMaybes
              [("w",) . showValue <$> (args !? #w :: Maybe Int),
               ("h",) . showValue <$> (args !? #h :: Maybe Int),
               ("interp",) . showValue <$> (args !? #interp :: Maybe Int)]
        tensorKeyArgs = catMaybes [("src",) <$> (args !? #src :: Maybe t)]
      in apply "_cvimresize" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__div_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__div_scalar ::
             forall a t . (Tensor t, Fullfilled "__div_scalar" t a) =>
               ArgsHMap "__div_scalar" t a -> TensorApply t
__div_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_div_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__equal" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__equal ::
        forall a t . (Tensor t, Fullfilled "__equal" t a) =>
          ArgsHMap "__equal" t a -> TensorApply t
__equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_equal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__equal_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__equal_scalar ::
               forall a t . (Tensor t, Fullfilled "__equal_scalar" t a) =>
                 ArgsHMap "__equal_scalar" t a -> TensorApply t
__equal_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_equal_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__full" t =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"])),
        '("value", AttrReq Double)]

__full ::
       forall a t . (Tensor t, Fullfilled "__full" t a) =>
         ArgsHMap "__full" t a -> TensorApply t
__full args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["float16", "float32", "float64", "int32", "int64", "int8",
                           "uint8"])),
               ("value",) . showValue <$> (args !? #value :: Maybe Double)]
        tensorKeyArgs = catMaybes []
      in apply "_full" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__grad_add" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__grad_add ::
           forall a t . (Tensor t, Fullfilled "__grad_add" t a) =>
             ArgsHMap "__grad_add" t a -> TensorApply t
__grad_add args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_grad_add" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__greater" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__greater ::
          forall a t . (Tensor t, Fullfilled "__greater" t a) =>
            ArgsHMap "__greater" t a -> TensorApply t
__greater args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_greater" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__greater_equal" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__greater_equal ::
                forall a t . (Tensor t, Fullfilled "__greater_equal" t a) =>
                  ArgsHMap "__greater_equal" t a -> TensorApply t
__greater_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_greater_equal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__greater_equal_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__greater_equal_scalar ::
                       forall a t . (Tensor t, Fullfilled "__greater_equal_scalar" t a) =>
                         ArgsHMap "__greater_equal_scalar" t a -> TensorApply t
__greater_equal_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_greater_equal_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__greater_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__greater_scalar ::
                 forall a t . (Tensor t, Fullfilled "__greater_scalar" t a) =>
                   ArgsHMap "__greater_scalar" t a -> TensorApply t
__greater_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_greater_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__histogram" t =
     '[ '("bin_cnt", AttrOpt (Maybe Int)), '("range", AttrOpt Int),
        '("data", AttrOpt t), '("bins", AttrOpt t)]

__histogram ::
            forall a t . (Tensor t, Fullfilled "__histogram" t a) =>
              ArgsHMap "__histogram" t a -> TensorApply t
__histogram args
  = let scalarArgs
          = catMaybes
              [("bin_cnt",) . showValue <$>
                 (args !? #bin_cnt :: Maybe (Maybe Int)),
               ("range",) . showValue <$> (args !? #range :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("bins",) <$> (args !? #bins :: Maybe t)]
      in apply "_histogram" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__hypot" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__hypot ::
        forall a t . (Tensor t, Fullfilled "__hypot" t a) =>
          ArgsHMap "__hypot" t a -> TensorApply t
__hypot args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_hypot" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__hypot_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__hypot_scalar ::
               forall a t . (Tensor t, Fullfilled "__hypot_scalar" t a) =>
                 ArgsHMap "__hypot_scalar" t a -> TensorApply t
__hypot_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_hypot_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__identity_with_attr_like_rhs" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__identity_with_attr_like_rhs ::
                              forall a t .
                                (Tensor t, Fullfilled "__identity_with_attr_like_rhs" t a) =>
                                ArgsHMap "__identity_with_attr_like_rhs" t a -> TensorApply t
__identity_with_attr_like_rhs args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in
      apply "_identity_with_attr_like_rhs" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__image_adjust_lighting" t =
     '[ '("alpha", AttrReq [Float]), '("data", AttrOpt t)]

__image_adjust_lighting ::
                        forall a t .
                          (Tensor t, Fullfilled "__image_adjust_lighting" t a) =>
                          ArgsHMap "__image_adjust_lighting" t a -> TensorApply t
__image_adjust_lighting args
  = let scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> (args !? #alpha :: Maybe [Float])]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_image_adjust_lighting" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__image_crop" t =
     '[ '("x", AttrReq Int), '("y", AttrReq Int),
        '("width", AttrReq Int), '("height", AttrReq Int),
        '("data", AttrOpt t)]

__image_crop ::
             forall a t . (Tensor t, Fullfilled "__image_crop" t a) =>
               ArgsHMap "__image_crop" t a -> TensorApply t
__image_crop args
  = let scalarArgs
          = catMaybes
              [("x",) . showValue <$> (args !? #x :: Maybe Int),
               ("y",) . showValue <$> (args !? #y :: Maybe Int),
               ("width",) . showValue <$> (args !? #width :: Maybe Int),
               ("height",) . showValue <$> (args !? #height :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_image_crop" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__image_flip_left_right" t =
     '[ '("data", AttrOpt t)]

__image_flip_left_right ::
                        forall a t .
                          (Tensor t, Fullfilled "__image_flip_left_right" t a) =>
                          ArgsHMap "__image_flip_left_right" t a -> TensorApply t
__image_flip_left_right args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_image_flip_left_right" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__image_flip_top_bottom" t =
     '[ '("data", AttrOpt t)]

__image_flip_top_bottom ::
                        forall a t .
                          (Tensor t, Fullfilled "__image_flip_top_bottom" t a) =>
                          ArgsHMap "__image_flip_top_bottom" t a -> TensorApply t
__image_flip_top_bottom args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_image_flip_top_bottom" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__image_normalize" t =
     '[ '("mean", AttrOpt [Float]), '("std", AttrOpt [Float]),
        '("data", AttrOpt t)]

__image_normalize ::
                  forall a t . (Tensor t, Fullfilled "__image_normalize" t a) =>
                    ArgsHMap "__image_normalize" t a -> TensorApply t
__image_normalize args
  = let scalarArgs
          = catMaybes
              [("mean",) . showValue <$> (args !? #mean :: Maybe [Float]),
               ("std",) . showValue <$> (args !? #std :: Maybe [Float])]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_image_normalize" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__image_random_brightness" t =
     '[ '("min_factor", AttrReq Float), '("max_factor", AttrReq Float),
        '("data", AttrOpt t)]

__image_random_brightness ::
                          forall a t .
                            (Tensor t, Fullfilled "__image_random_brightness" t a) =>
                            ArgsHMap "__image_random_brightness" t a -> TensorApply t
__image_random_brightness args
  = let scalarArgs
          = catMaybes
              [("min_factor",) . showValue <$>
                 (args !? #min_factor :: Maybe Float),
               ("max_factor",) . showValue <$>
                 (args !? #max_factor :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_image_random_brightness" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__image_random_color_jitter" t =
     '[ '("brightness", AttrReq Float), '("contrast", AttrReq Float),
        '("saturation", AttrReq Float), '("hue", AttrReq Float),
        '("data", AttrOpt t)]

__image_random_color_jitter ::
                            forall a t .
                              (Tensor t, Fullfilled "__image_random_color_jitter" t a) =>
                              ArgsHMap "__image_random_color_jitter" t a -> TensorApply t
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
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in
      apply "_image_random_color_jitter" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__image_random_contrast" t =
     '[ '("min_factor", AttrReq Float), '("max_factor", AttrReq Float),
        '("data", AttrOpt t)]

__image_random_contrast ::
                        forall a t .
                          (Tensor t, Fullfilled "__image_random_contrast" t a) =>
                          ArgsHMap "__image_random_contrast" t a -> TensorApply t
__image_random_contrast args
  = let scalarArgs
          = catMaybes
              [("min_factor",) . showValue <$>
                 (args !? #min_factor :: Maybe Float),
               ("max_factor",) . showValue <$>
                 (args !? #max_factor :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_image_random_contrast" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__image_random_flip_left_right" t =
     '[ '("data", AttrOpt t)]

__image_random_flip_left_right ::
                               forall a t .
                                 (Tensor t, Fullfilled "__image_random_flip_left_right" t a) =>
                                 ArgsHMap "__image_random_flip_left_right" t a -> TensorApply t
__image_random_flip_left_right args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in
      apply "_image_random_flip_left_right" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__image_random_flip_top_bottom" t =
     '[ '("data", AttrOpt t)]

__image_random_flip_top_bottom ::
                               forall a t .
                                 (Tensor t, Fullfilled "__image_random_flip_top_bottom" t a) =>
                                 ArgsHMap "__image_random_flip_top_bottom" t a -> TensorApply t
__image_random_flip_top_bottom args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in
      apply "_image_random_flip_top_bottom" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__image_random_hue" t =
     '[ '("min_factor", AttrReq Float), '("max_factor", AttrReq Float),
        '("data", AttrOpt t)]

__image_random_hue ::
                   forall a t . (Tensor t, Fullfilled "__image_random_hue" t a) =>
                     ArgsHMap "__image_random_hue" t a -> TensorApply t
__image_random_hue args
  = let scalarArgs
          = catMaybes
              [("min_factor",) . showValue <$>
                 (args !? #min_factor :: Maybe Float),
               ("max_factor",) . showValue <$>
                 (args !? #max_factor :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_image_random_hue" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__image_random_lighting" t =
     '[ '("alpha_std", AttrOpt Float), '("data", AttrOpt t)]

__image_random_lighting ::
                        forall a t .
                          (Tensor t, Fullfilled "__image_random_lighting" t a) =>
                          ArgsHMap "__image_random_lighting" t a -> TensorApply t
__image_random_lighting args
  = let scalarArgs
          = catMaybes
              [("alpha_std",) . showValue <$>
                 (args !? #alpha_std :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_image_random_lighting" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__image_random_saturation" t =
     '[ '("min_factor", AttrReq Float), '("max_factor", AttrReq Float),
        '("data", AttrOpt t)]

__image_random_saturation ::
                          forall a t .
                            (Tensor t, Fullfilled "__image_random_saturation" t a) =>
                            ArgsHMap "__image_random_saturation" t a -> TensorApply t
__image_random_saturation args
  = let scalarArgs
          = catMaybes
              [("min_factor",) . showValue <$>
                 (args !? #min_factor :: Maybe Float),
               ("max_factor",) . showValue <$>
                 (args !? #max_factor :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_image_random_saturation" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__image_resize" t =
     '[ '("size", AttrOpt [Int]), '("keep_ratio", AttrOpt Bool),
        '("interp", AttrOpt Int), '("data", AttrOpt t)]

__image_resize ::
               forall a t . (Tensor t, Fullfilled "__image_resize" t a) =>
                 ArgsHMap "__image_resize" t a -> TensorApply t
__image_resize args
  = let scalarArgs
          = catMaybes
              [("size",) . showValue <$> (args !? #size :: Maybe [Int]),
               ("keep_ratio",) . showValue <$>
                 (args !? #keep_ratio :: Maybe Bool),
               ("interp",) . showValue <$> (args !? #interp :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_image_resize" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__image_to_tensor" t =
     '[ '("data", AttrOpt t)]

__image_to_tensor ::
                  forall a t . (Tensor t, Fullfilled "__image_to_tensor" t a) =>
                    ArgsHMap "__image_to_tensor" t a -> TensorApply t
__image_to_tensor args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_image_to_tensor" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__imdecode" t =
     '[ '("index", AttrOpt Int), '("x0", AttrOpt Int),
        '("y0", AttrOpt Int), '("x1", AttrOpt Int), '("y1", AttrOpt Int),
        '("c", AttrOpt Int), '("size", AttrOpt Int), '("mean", AttrOpt t)]

__imdecode ::
           forall a t . (Tensor t, Fullfilled "__imdecode" t a) =>
             ArgsHMap "__imdecode" t a -> TensorApply t
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
          = catMaybes [("mean",) <$> (args !? #mean :: Maybe t)]
      in apply "_imdecode" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__lesser" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__lesser ::
         forall a t . (Tensor t, Fullfilled "__lesser" t a) =>
           ArgsHMap "__lesser" t a -> TensorApply t
__lesser args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_lesser" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__lesser_equal" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__lesser_equal ::
               forall a t . (Tensor t, Fullfilled "__lesser_equal" t a) =>
                 ArgsHMap "__lesser_equal" t a -> TensorApply t
__lesser_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_lesser_equal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__lesser_equal_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__lesser_equal_scalar ::
                      forall a t . (Tensor t, Fullfilled "__lesser_equal_scalar" t a) =>
                        ArgsHMap "__lesser_equal_scalar" t a -> TensorApply t
__lesser_equal_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_lesser_equal_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__lesser_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__lesser_scalar ::
                forall a t . (Tensor t, Fullfilled "__lesser_scalar" t a) =>
                  ArgsHMap "__lesser_scalar" t a -> TensorApply t
__lesser_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_lesser_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__linalg_det" t =
     '[ '("a", AttrOpt t)]

__linalg_det ::
             forall a t . (Tensor t, Fullfilled "__linalg_det" t a) =>
               ArgsHMap "__linalg_det" t a -> TensorApply t
__linalg_det args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_linalg_det" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__linalg_extractdiag" t =
     '[ '("offset", AttrOpt Int), '("a", AttrOpt t)]

__linalg_extractdiag ::
                     forall a t . (Tensor t, Fullfilled "__linalg_extractdiag" t a) =>
                       ArgsHMap "__linalg_extractdiag" t a -> TensorApply t
__linalg_extractdiag args
  = let scalarArgs
          = catMaybes
              [("offset",) . showValue <$> (args !? #offset :: Maybe Int)]
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_linalg_extractdiag" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__linalg_extracttrian" t =
     '[ '("offset", AttrOpt Int), '("lower", AttrOpt Bool),
        '("a", AttrOpt t)]

__linalg_extracttrian ::
                      forall a t . (Tensor t, Fullfilled "__linalg_extracttrian" t a) =>
                        ArgsHMap "__linalg_extracttrian" t a -> TensorApply t
__linalg_extracttrian args
  = let scalarArgs
          = catMaybes
              [("offset",) . showValue <$> (args !? #offset :: Maybe Int),
               ("lower",) . showValue <$> (args !? #lower :: Maybe Bool)]
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_linalg_extracttrian" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__linalg_gelqf" t =
     '[ '("a", AttrOpt t)]

__linalg_gelqf ::
               forall a t . (Tensor t, Fullfilled "__linalg_gelqf" t a) =>
                 ArgsHMap "__linalg_gelqf" t a -> TensorApply t
__linalg_gelqf args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_linalg_gelqf" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__linalg_gemm" t =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("alpha", AttrOpt Double), '("beta", AttrOpt Double),
        '("axis", AttrOpt Int), '("a", AttrOpt t), '("b", AttrOpt t),
        '("c", AttrOpt t)]

__linalg_gemm ::
              forall a t . (Tensor t, Fullfilled "__linalg_gemm" t a) =>
                ArgsHMap "__linalg_gemm" t a -> TensorApply t
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
              [("a",) <$> (args !? #a :: Maybe t),
               ("b",) <$> (args !? #b :: Maybe t),
               ("c",) <$> (args !? #c :: Maybe t)]
      in apply "_linalg_gemm" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__linalg_gemm2" t =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("alpha", AttrOpt Double), '("axis", AttrOpt Int),
        '("a", AttrOpt t), '("b", AttrOpt t)]

__linalg_gemm2 ::
               forall a t . (Tensor t, Fullfilled "__linalg_gemm2" t a) =>
                 ArgsHMap "__linalg_gemm2" t a -> TensorApply t
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
              [("a",) <$> (args !? #a :: Maybe t),
               ("b",) <$> (args !? #b :: Maybe t)]
      in apply "_linalg_gemm2" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__linalg_inverse" t =
     '[ '("a", AttrOpt t)]

__linalg_inverse ::
                 forall a t . (Tensor t, Fullfilled "__linalg_inverse" t a) =>
                   ArgsHMap "__linalg_inverse" t a -> TensorApply t
__linalg_inverse args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_linalg_inverse" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__linalg_makediag" t =
     '[ '("offset", AttrOpt Int), '("a", AttrOpt t)]

__linalg_makediag ::
                  forall a t . (Tensor t, Fullfilled "__linalg_makediag" t a) =>
                    ArgsHMap "__linalg_makediag" t a -> TensorApply t
__linalg_makediag args
  = let scalarArgs
          = catMaybes
              [("offset",) . showValue <$> (args !? #offset :: Maybe Int)]
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_linalg_makediag" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__linalg_maketrian" t =
     '[ '("offset", AttrOpt Int), '("lower", AttrOpt Bool),
        '("a", AttrOpt t)]

__linalg_maketrian ::
                   forall a t . (Tensor t, Fullfilled "__linalg_maketrian" t a) =>
                     ArgsHMap "__linalg_maketrian" t a -> TensorApply t
__linalg_maketrian args
  = let scalarArgs
          = catMaybes
              [("offset",) . showValue <$> (args !? #offset :: Maybe Int),
               ("lower",) . showValue <$> (args !? #lower :: Maybe Bool)]
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_linalg_maketrian" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__linalg_potrf" t =
     '[ '("a", AttrOpt t)]

__linalg_potrf ::
               forall a t . (Tensor t, Fullfilled "__linalg_potrf" t a) =>
                 ArgsHMap "__linalg_potrf" t a -> TensorApply t
__linalg_potrf args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_linalg_potrf" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__linalg_potri" t =
     '[ '("a", AttrOpt t)]

__linalg_potri ::
               forall a t . (Tensor t, Fullfilled "__linalg_potri" t a) =>
                 ArgsHMap "__linalg_potri" t a -> TensorApply t
__linalg_potri args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_linalg_potri" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__linalg_slogdet" t =
     '[ '("a", AttrOpt t)]

__linalg_slogdet ::
                 forall a t . (Tensor t, Fullfilled "__linalg_slogdet" t a) =>
                   ArgsHMap "__linalg_slogdet" t a -> TensorApply t
__linalg_slogdet args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_linalg_slogdet" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__linalg_sumlogdiag" t =
     '[ '("a", AttrOpt t)]

__linalg_sumlogdiag ::
                    forall a t . (Tensor t, Fullfilled "__linalg_sumlogdiag" t a) =>
                      ArgsHMap "__linalg_sumlogdiag" t a -> TensorApply t
__linalg_sumlogdiag args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_linalg_sumlogdiag" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__linalg_syevd" t =
     '[ '("a", AttrOpt t)]

__linalg_syevd ::
               forall a t . (Tensor t, Fullfilled "__linalg_syevd" t a) =>
                 ArgsHMap "__linalg_syevd" t a -> TensorApply t
__linalg_syevd args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_linalg_syevd" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__linalg_syrk" t =
     '[ '("transpose", AttrOpt Bool), '("alpha", AttrOpt Double),
        '("a", AttrOpt t)]

__linalg_syrk ::
              forall a t . (Tensor t, Fullfilled "__linalg_syrk" t a) =>
                ArgsHMap "__linalg_syrk" t a -> TensorApply t
__linalg_syrk args
  = let scalarArgs
          = catMaybes
              [("transpose",) . showValue <$> (args !? #transpose :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double)]
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_linalg_syrk" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__linalg_trmm" t =
     '[ '("transpose", AttrOpt Bool), '("rightside", AttrOpt Bool),
        '("lower", AttrOpt Bool), '("alpha", AttrOpt Double),
        '("a", AttrOpt t), '("b", AttrOpt t)]

__linalg_trmm ::
              forall a t . (Tensor t, Fullfilled "__linalg_trmm" t a) =>
                ArgsHMap "__linalg_trmm" t a -> TensorApply t
__linalg_trmm args
  = let scalarArgs
          = catMaybes
              [("transpose",) . showValue <$> (args !? #transpose :: Maybe Bool),
               ("rightside",) . showValue <$> (args !? #rightside :: Maybe Bool),
               ("lower",) . showValue <$> (args !? #lower :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double)]
        tensorKeyArgs
          = catMaybes
              [("a",) <$> (args !? #a :: Maybe t),
               ("b",) <$> (args !? #b :: Maybe t)]
      in apply "_linalg_trmm" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__linalg_trsm" t =
     '[ '("transpose", AttrOpt Bool), '("rightside", AttrOpt Bool),
        '("lower", AttrOpt Bool), '("alpha", AttrOpt Double),
        '("a", AttrOpt t), '("b", AttrOpt t)]

__linalg_trsm ::
              forall a t . (Tensor t, Fullfilled "__linalg_trsm" t a) =>
                ArgsHMap "__linalg_trsm" t a -> TensorApply t
__linalg_trsm args
  = let scalarArgs
          = catMaybes
              [("transpose",) . showValue <$> (args !? #transpose :: Maybe Bool),
               ("rightside",) . showValue <$> (args !? #rightside :: Maybe Bool),
               ("lower",) . showValue <$> (args !? #lower :: Maybe Bool),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Double)]
        tensorKeyArgs
          = catMaybes
              [("a",) <$> (args !? #a :: Maybe t),
               ("b",) <$> (args !? #b :: Maybe t)]
      in apply "_linalg_trsm" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__linspace" t =
     '[ '("start", AttrReq Double), '("stop", AttrOpt (Maybe Double)),
        '("step", AttrOpt Double), '("repeat", AttrOpt Int),
        '("infer_range", AttrOpt Bool), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"]))]

__linspace ::
           forall a t . (Tensor t, Fullfilled "__linspace" t a) =>
             ArgsHMap "__linspace" t a -> TensorApply t
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
                         '["float16", "float32", "float64", "int32", "int64", "int8",
                           "uint8"]))]
        tensorKeyArgs = catMaybes []
      in apply "_linspace" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__logical_and" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__logical_and ::
              forall a t . (Tensor t, Fullfilled "__logical_and" t a) =>
                ArgsHMap "__logical_and" t a -> TensorApply t
__logical_and args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_logical_and" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__logical_and_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__logical_and_scalar ::
                     forall a t . (Tensor t, Fullfilled "__logical_and_scalar" t a) =>
                       ArgsHMap "__logical_and_scalar" t a -> TensorApply t
__logical_and_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_logical_and_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__logical_or" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__logical_or ::
             forall a t . (Tensor t, Fullfilled "__logical_or" t a) =>
               ArgsHMap "__logical_or" t a -> TensorApply t
__logical_or args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_logical_or" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__logical_or_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__logical_or_scalar ::
                    forall a t . (Tensor t, Fullfilled "__logical_or_scalar" t a) =>
                      ArgsHMap "__logical_or_scalar" t a -> TensorApply t
__logical_or_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_logical_or_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__logical_xor" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__logical_xor ::
              forall a t . (Tensor t, Fullfilled "__logical_xor" t a) =>
                ArgsHMap "__logical_xor" t a -> TensorApply t
__logical_xor args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_logical_xor" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__logical_xor_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__logical_xor_scalar ::
                     forall a t . (Tensor t, Fullfilled "__logical_xor_scalar" t a) =>
                       ArgsHMap "__logical_xor_scalar" t a -> TensorApply t
__logical_xor_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_logical_xor_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__maximum" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__maximum ::
          forall a t . (Tensor t, Fullfilled "__maximum" t a) =>
            ArgsHMap "__maximum" t a -> TensorApply t
__maximum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_maximum" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__maximum_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__maximum_scalar ::
                 forall a t . (Tensor t, Fullfilled "__maximum_scalar" t a) =>
                   ArgsHMap "__maximum_scalar" t a -> TensorApply t
__maximum_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_maximum_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__minimum" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__minimum ::
          forall a t . (Tensor t, Fullfilled "__minimum" t a) =>
            ArgsHMap "__minimum" t a -> TensorApply t
__minimum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_minimum" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__minimum_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__minimum_scalar ::
                 forall a t . (Tensor t, Fullfilled "__minimum_scalar" t a) =>
                   ArgsHMap "__minimum_scalar" t a -> TensorApply t
__minimum_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_minimum_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__minus_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__minus_scalar ::
               forall a t . (Tensor t, Fullfilled "__minus_scalar" t a) =>
                 ArgsHMap "__minus_scalar" t a -> TensorApply t
__minus_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_minus_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__mod" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__mod ::
      forall a t . (Tensor t, Fullfilled "__mod" t a) =>
        ArgsHMap "__mod" t a -> TensorApply t
__mod args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_mod" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__mod_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__mod_scalar ::
             forall a t . (Tensor t, Fullfilled "__mod_scalar" t a) =>
               ArgsHMap "__mod_scalar" t a -> TensorApply t
__mod_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_mod_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__mp_adamw_update" t =
     '[ '("lr", AttrReq Float), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wd", AttrOpt Float), '("eta", AttrReq Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt t),
        '("grad", AttrOpt t), '("mean", AttrOpt t), '("var", AttrOpt t),
        '("weight32", AttrOpt t), '("rescale_grad", AttrOpt t)]

__mp_adamw_update ::
                  forall a t . (Tensor t, Fullfilled "__mp_adamw_update" t a) =>
                    ArgsHMap "__mp_adamw_update" t a -> TensorApply t
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
              [("weight",) <$> (args !? #weight :: Maybe t),
               ("grad",) <$> (args !? #grad :: Maybe t),
               ("mean",) <$> (args !? #mean :: Maybe t),
               ("var",) <$> (args !? #var :: Maybe t),
               ("weight32",) <$> (args !? #weight32 :: Maybe t),
               ("rescale_grad",) <$> (args !? #rescale_grad :: Maybe t)]
      in apply "_mp_adamw_update" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__mul_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__mul_scalar ::
             forall a t . (Tensor t, Fullfilled "__mul_scalar" t a) =>
               ArgsHMap "__mul_scalar" t a -> TensorApply t
__mul_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_mul_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__multi_adamw_update" t =
     '[ '("lrs", AttrReq [Float]), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wds", AttrReq [Float]), '("etas", AttrReq [Float]),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("data", AttrOpt [t])]

__multi_adamw_update ::
                     forall a t . (Tensor t, Fullfilled "__multi_adamw_update" t a) =>
                       ArgsHMap "__multi_adamw_update" t a -> TensorApply t
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
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "_multi_adamw_update" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__multi_mp_adamw_update" t =
     '[ '("lrs", AttrReq [Float]), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wds", AttrReq [Float]), '("etas", AttrReq [Float]),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("data", AttrOpt [t])]

__multi_mp_adamw_update ::
                        forall a t .
                          (Tensor t, Fullfilled "__multi_mp_adamw_update" t a) =>
                          ArgsHMap "__multi_mp_adamw_update" t a -> TensorApply t
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
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "_multi_mp_adamw_update" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__not_equal" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__not_equal ::
            forall a t . (Tensor t, Fullfilled "__not_equal" t a) =>
              ArgsHMap "__not_equal" t a -> TensorApply t
__not_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_not_equal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__not_equal_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__not_equal_scalar ::
                   forall a t . (Tensor t, Fullfilled "__not_equal_scalar" t a) =>
                     ArgsHMap "__not_equal_scalar" t a -> TensorApply t
__not_equal_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_not_equal_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__np_broadcast_to" t =
     '[ '("shape", AttrOpt [Int]), '("array", AttrOpt t)]

__np_broadcast_to ::
                  forall a t . (Tensor t, Fullfilled "__np_broadcast_to" t a) =>
                    ArgsHMap "__np_broadcast_to" t a -> TensorApply t
__np_broadcast_to args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("array",) <$> (args !? #array :: Maybe t)]
      in apply "_np_broadcast_to" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__np_copy" t = '[ '("a", AttrOpt t)]

__np_copy ::
          forall a t . (Tensor t, Fullfilled "__np_copy" t a) =>
            ArgsHMap "__np_copy" t a -> TensorApply t
__np_copy args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_np_copy" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__np_cumsum" t =
     '[ '("axis", AttrOpt (Maybe Int)),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["float16", "float32", "float64", "int32", "int64", "int8"]))),
        '("a", AttrOpt t)]

__np_cumsum ::
            forall a t . (Tensor t, Fullfilled "__np_cumsum" t a) =>
              ArgsHMap "__np_cumsum" t a -> TensorApply t
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
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_np_cumsum" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__np_dot" t =
     '[ '("a", AttrOpt t), '("b", AttrOpt t)]

__np_dot ::
         forall a t . (Tensor t, Fullfilled "__np_dot" t a) =>
           ArgsHMap "__np_dot" t a -> TensorApply t
__np_dot args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("a",) <$> (args !? #a :: Maybe t),
               ("b",) <$> (args !? #b :: Maybe t)]
      in apply "_np_dot" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__np_max" t =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("initial", AttrOpt (Maybe Double)), '("a", AttrOpt t)]

__np_max ::
         forall a t . (Tensor t, Fullfilled "__np_max" t a) =>
           ArgsHMap "__np_max" t a -> TensorApply t
__np_max args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("initial",) . showValue <$>
                 (args !? #initial :: Maybe (Maybe Double))]
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_np_max" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__np_min" t =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("initial", AttrOpt (Maybe Double)), '("a", AttrOpt t)]

__np_min ::
         forall a t . (Tensor t, Fullfilled "__np_min" t a) =>
           ArgsHMap "__np_min" t a -> TensorApply t
__np_min args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("initial",) . showValue <$>
                 (args !? #initial :: Maybe (Maybe Double))]
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_np_min" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__np_moveaxis" t =
     '[ '("source", AttrReq [Int]), '("destination", AttrReq [Int]),
        '("a", AttrOpt t)]

__np_moveaxis ::
              forall a t . (Tensor t, Fullfilled "__np_moveaxis" t a) =>
                ArgsHMap "__np_moveaxis" t a -> TensorApply t
__np_moveaxis args
  = let scalarArgs
          = catMaybes
              [("source",) . showValue <$> (args !? #source :: Maybe [Int]),
               ("destination",) . showValue <$>
                 (args !? #destination :: Maybe [Int])]
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_np_moveaxis" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__np_ones_like" t =
     '[ '("a", AttrOpt t)]

__np_ones_like ::
               forall a t . (Tensor t, Fullfilled "__np_ones_like" t a) =>
                 ArgsHMap "__np_ones_like" t a -> TensorApply t
__np_ones_like args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_np_ones_like" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__np_prod" t =
     '[ '("axis", AttrOpt (Maybe [Int])),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["bool", "float16", "float32", "float64", "int32", "int64",
                    "int8"]))),
        '("keepdims", AttrOpt Bool), '("initial", AttrOpt (Maybe Double)),
        '("a", AttrOpt t)]

__np_prod ::
          forall a t . (Tensor t, Fullfilled "__np_prod" t a) =>
            ArgsHMap "__np_prod" t a -> TensorApply t
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
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_np_prod" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__np_reshape" t =
     '[ '("newshape", AttrReq [Int]), '("order", AttrOpt Text),
        '("a", AttrOpt t)]

__np_reshape ::
             forall a t . (Tensor t, Fullfilled "__np_reshape" t a) =>
               ArgsHMap "__np_reshape" t a -> TensorApply t
__np_reshape args
  = let scalarArgs
          = catMaybes
              [("newshape",) . showValue <$> (args !? #newshape :: Maybe [Int]),
               ("order",) . showValue <$> (args !? #order :: Maybe Text)]
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_np_reshape" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__np_roll" t =
     '[ '("shift", AttrOpt (Maybe [Int])),
        '("axis", AttrOpt (Maybe [Int])), '("data", AttrOpt t)]

__np_roll ::
          forall a t . (Tensor t, Fullfilled "__np_roll" t a) =>
            ArgsHMap "__np_roll" t a -> TensorApply t
__np_roll args
  = let scalarArgs
          = catMaybes
              [("shift",) . showValue <$>
                 (args !? #shift :: Maybe (Maybe [Int])),
               ("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int]))]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_np_roll" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__np_squeeze" t =
     '[ '("axis", AttrOpt (Maybe [Int])), '("a", AttrOpt t)]

__np_squeeze ::
             forall a t . (Tensor t, Fullfilled "__np_squeeze" t a) =>
               ArgsHMap "__np_squeeze" t a -> TensorApply t
__np_squeeze args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int]))]
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_np_squeeze" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__np_sum" t =
     '[ '("axis", AttrOpt (Maybe [Int])),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["bool", "float16", "float32", "float64", "int32", "int64",
                    "int8"]))),
        '("keepdims", AttrOpt Bool), '("initial", AttrOpt (Maybe Double)),
        '("a", AttrOpt t)]

__np_sum ::
         forall a t . (Tensor t, Fullfilled "__np_sum" t a) =>
           ArgsHMap "__np_sum" t a -> TensorApply t
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
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_np_sum" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__np_trace" t =
     '[ '("offset", AttrOpt Int), '("axis1", AttrOpt Int),
        '("axis2", AttrOpt Int), '("data", AttrOpt t)]

__np_trace ::
           forall a t . (Tensor t, Fullfilled "__np_trace" t a) =>
             ArgsHMap "__np_trace" t a -> TensorApply t
__np_trace args
  = let scalarArgs
          = catMaybes
              [("offset",) . showValue <$> (args !? #offset :: Maybe Int),
               ("axis1",) . showValue <$> (args !? #axis1 :: Maybe Int),
               ("axis2",) . showValue <$> (args !? #axis2 :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_np_trace" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__np_transpose" t =
     '[ '("axes", AttrOpt [Int]), '("a", AttrOpt t)]

__np_transpose ::
               forall a t . (Tensor t, Fullfilled "__np_transpose" t a) =>
                 ArgsHMap "__np_transpose" t a -> TensorApply t
__np_transpose args
  = let scalarArgs
          = catMaybes
              [("axes",) . showValue <$> (args !? #axes :: Maybe [Int])]
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_np_transpose" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__np_zeros_like" t =
     '[ '("a", AttrOpt t)]

__np_zeros_like ::
                forall a t . (Tensor t, Fullfilled "__np_zeros_like" t a) =>
                  ArgsHMap "__np_zeros_like" t a -> TensorApply t
__np_zeros_like args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_np_zeros_like" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_absolute" t =
     '[ '("x", AttrOpt t)]

__npi_absolute ::
               forall a t . (Tensor t, Fullfilled "__npi_absolute" t a) =>
                 ArgsHMap "__npi_absolute" t a -> TensorApply t
__npi_absolute args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_absolute" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_add" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__npi_add ::
          forall a t . (Tensor t, Fullfilled "__npi_add" t a) =>
            ArgsHMap "__npi_add" t a -> TensorApply t
__npi_add args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_npi_add" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_add_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_add_scalar ::
                 forall a t . (Tensor t, Fullfilled "__npi_add_scalar" t a) =>
                   ArgsHMap "__npi_add_scalar" t a -> TensorApply t
__npi_add_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_add_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_arange" t =
     '[ '("start", AttrReq Double), '("stop", AttrOpt (Maybe Double)),
        '("step", AttrOpt Double), '("repeat", AttrOpt Int),
        '("infer_range", AttrOpt Bool), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"]))]

__npi_arange ::
             forall a t . (Tensor t, Fullfilled "__npi_arange" t a) =>
               ArgsHMap "__npi_arange" t a -> TensorApply t
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
                         '["float16", "float32", "float64", "int32", "int64", "int8",
                           "uint8"]))]
        tensorKeyArgs = catMaybes []
      in apply "_npi_arange" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_arccos" t =
     '[ '("x", AttrOpt t)]

__npi_arccos ::
             forall a t . (Tensor t, Fullfilled "__npi_arccos" t a) =>
               ArgsHMap "__npi_arccos" t a -> TensorApply t
__npi_arccos args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_arccos" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_arccosh" t =
     '[ '("x", AttrOpt t)]

__npi_arccosh ::
              forall a t . (Tensor t, Fullfilled "__npi_arccosh" t a) =>
                ArgsHMap "__npi_arccosh" t a -> TensorApply t
__npi_arccosh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_arccosh" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_arcsin" t =
     '[ '("x", AttrOpt t)]

__npi_arcsin ::
             forall a t . (Tensor t, Fullfilled "__npi_arcsin" t a) =>
               ArgsHMap "__npi_arcsin" t a -> TensorApply t
__npi_arcsin args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_arcsin" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_arcsinh" t =
     '[ '("x", AttrOpt t)]

__npi_arcsinh ::
              forall a t . (Tensor t, Fullfilled "__npi_arcsinh" t a) =>
                ArgsHMap "__npi_arcsinh" t a -> TensorApply t
__npi_arcsinh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_arcsinh" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_arctan" t =
     '[ '("x", AttrOpt t)]

__npi_arctan ::
             forall a t . (Tensor t, Fullfilled "__npi_arctan" t a) =>
               ArgsHMap "__npi_arctan" t a -> TensorApply t
__npi_arctan args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_arctan" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_arctan2" t =
     '[ '("x1", AttrOpt t), '("x2", AttrOpt t)]

__npi_arctan2 ::
              forall a t . (Tensor t, Fullfilled "__npi_arctan2" t a) =>
                ArgsHMap "__npi_arctan2" t a -> TensorApply t
__npi_arctan2 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("x1",) <$> (args !? #x1 :: Maybe t),
               ("x2",) <$> (args !? #x2 :: Maybe t)]
      in apply "_npi_arctan2" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_arctan2_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_arctan2_scalar ::
                     forall a t . (Tensor t, Fullfilled "__npi_arctan2_scalar" t a) =>
                       ArgsHMap "__npi_arctan2_scalar" t a -> TensorApply t
__npi_arctan2_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_arctan2_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_arctanh" t =
     '[ '("x", AttrOpt t)]

__npi_arctanh ::
              forall a t . (Tensor t, Fullfilled "__npi_arctanh" t a) =>
                ArgsHMap "__npi_arctanh" t a -> TensorApply t
__npi_arctanh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_arctanh" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_argmax" t =
     '[ '("axis", AttrOpt (Maybe Int)), '("keepdims", AttrOpt Bool),
        '("data", AttrOpt t)]

__npi_argmax ::
             forall a t . (Tensor t, Fullfilled "__npi_argmax" t a) =>
               ArgsHMap "__npi_argmax" t a -> TensorApply t
__npi_argmax args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_argmax" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_argmin" t =
     '[ '("axis", AttrOpt (Maybe Int)), '("keepdims", AttrOpt Bool),
        '("data", AttrOpt t)]

__npi_argmin ::
             forall a t . (Tensor t, Fullfilled "__npi_argmin" t a) =>
               ArgsHMap "__npi_argmin" t a -> TensorApply t
__npi_argmin args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_argmin" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_around" t =
     '[ '("decimals", AttrOpt Int), '("x", AttrOpt t)]

__npi_around ::
             forall a t . (Tensor t, Fullfilled "__npi_around" t a) =>
               ArgsHMap "__npi_around" t a -> TensorApply t
__npi_around args
  = let scalarArgs
          = catMaybes
              [("decimals",) . showValue <$> (args !? #decimals :: Maybe Int)]
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_around" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_bitwise_xor" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__npi_bitwise_xor ::
                  forall a t . (Tensor t, Fullfilled "__npi_bitwise_xor" t a) =>
                    ArgsHMap "__npi_bitwise_xor" t a -> TensorApply t
__npi_bitwise_xor args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_npi_bitwise_xor" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_bitwise_xor_scalar" t =
     '[ '("scalar", AttrOpt Int), '("data", AttrOpt t)]

__npi_bitwise_xor_scalar ::
                         forall a t .
                           (Tensor t, Fullfilled "__npi_bitwise_xor_scalar" t a) =>
                           ArgsHMap "__npi_bitwise_xor_scalar" t a -> TensorApply t
__npi_bitwise_xor_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_bitwise_xor_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_blackman" t =
     '[ '("m", AttrOpt Int), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"]))]

__npi_blackman ::
               forall a t . (Tensor t, Fullfilled "__npi_blackman" t a) =>
                 ArgsHMap "__npi_blackman" t a -> TensorApply t
__npi_blackman args
  = let scalarArgs
          = catMaybes
              [("m",) . showValue <$> (args !? #m :: Maybe Int),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["float16", "float32", "float64", "int32", "int64", "int8",
                           "uint8"]))]
        tensorKeyArgs = catMaybes []
      in apply "_npi_blackman" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_boolean_mask_assign_scalar" t =
     '[ '("value", AttrOpt Float), '("data", AttrOpt t),
        '("mask", AttrOpt t)]

__npi_boolean_mask_assign_scalar ::
                                 forall a t .
                                   (Tensor t, Fullfilled "__npi_boolean_mask_assign_scalar" t a) =>
                                   ArgsHMap "__npi_boolean_mask_assign_scalar" t a -> TensorApply t
__npi_boolean_mask_assign_scalar args
  = let scalarArgs
          = catMaybes
              [("value",) . showValue <$> (args !? #value :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("mask",) <$> (args !? #mask :: Maybe t)]
      in
      apply "_npi_boolean_mask_assign_scalar" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__npi_boolean_mask_assign_tensor" t =
     '[ '("data", AttrOpt t), '("mask", AttrOpt t),
        '("value", AttrOpt t)]

__npi_boolean_mask_assign_tensor ::
                                 forall a t .
                                   (Tensor t, Fullfilled "__npi_boolean_mask_assign_tensor" t a) =>
                                   ArgsHMap "__npi_boolean_mask_assign_tensor" t a -> TensorApply t
__npi_boolean_mask_assign_tensor args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("mask",) <$> (args !? #mask :: Maybe t),
               ("value",) <$> (args !? #value :: Maybe t)]
      in
      apply "_npi_boolean_mask_assign_tensor" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__npi_cbrt" t = '[ '("x", AttrOpt t)]

__npi_cbrt ::
           forall a t . (Tensor t, Fullfilled "__npi_cbrt" t a) =>
             ArgsHMap "__npi_cbrt" t a -> TensorApply t
__npi_cbrt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_cbrt" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_ceil" t = '[ '("x", AttrOpt t)]

__npi_ceil ::
           forall a t . (Tensor t, Fullfilled "__npi_ceil" t a) =>
             ArgsHMap "__npi_ceil" t a -> TensorApply t
__npi_ceil args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_ceil" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_choice" t =
     '[ '("a", AttrReq Int), '("size", AttrReq Int),
        '("ctx", AttrOpt Text), '("replace", AttrOpt Bool),
        '("weighted", AttrOpt Bool), '("input1", AttrOpt t),
        '("input2", AttrOpt t)]

__npi_choice ::
             forall a t . (Tensor t, Fullfilled "__npi_choice" t a) =>
               ArgsHMap "__npi_choice" t a -> TensorApply t
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
              [("input1",) <$> (args !? #input1 :: Maybe t),
               ("input2",) <$> (args !? #input2 :: Maybe t)]
      in apply "_npi_choice" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_column_stack" t =
     '[ '("num_args", AttrReq Int), '("data", AttrOpt [t])]

__npi_column_stack ::
                   forall a t . (Tensor t, Fullfilled "__npi_column_stack" t a) =>
                     ArgsHMap "__npi_column_stack" t a -> TensorApply t
__npi_column_stack args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "_npi_column_stack" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__npi_concatenate" t =
     '[ '("num_args", AttrReq Int), '("dim", AttrOpt Int),
        '("data", AttrOpt [t])]

__npi_concatenate ::
                  forall a t . (Tensor t, Fullfilled "__npi_concatenate" t a) =>
                    ArgsHMap "__npi_concatenate" t a -> TensorApply t
__npi_concatenate args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("dim",) . showValue <$> (args !? #dim :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "_npi_concatenate" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__npi_copysign" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__npi_copysign ::
               forall a t . (Tensor t, Fullfilled "__npi_copysign" t a) =>
                 ArgsHMap "__npi_copysign" t a -> TensorApply t
__npi_copysign args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_npi_copysign" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_copysign_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_copysign_scalar ::
                      forall a t . (Tensor t, Fullfilled "__npi_copysign_scalar" t a) =>
                        ArgsHMap "__npi_copysign_scalar" t a -> TensorApply t
__npi_copysign_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_copysign_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_cos" t = '[ '("x", AttrOpt t)]

__npi_cos ::
          forall a t . (Tensor t, Fullfilled "__npi_cos" t a) =>
            ArgsHMap "__npi_cos" t a -> TensorApply t
__npi_cos args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_cos" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_cosh" t = '[ '("x", AttrOpt t)]

__npi_cosh ::
           forall a t . (Tensor t, Fullfilled "__npi_cosh" t a) =>
             ArgsHMap "__npi_cosh" t a -> TensorApply t
__npi_cosh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_cosh" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_degrees" t =
     '[ '("x", AttrOpt t)]

__npi_degrees ::
              forall a t . (Tensor t, Fullfilled "__npi_degrees" t a) =>
                ArgsHMap "__npi_degrees" t a -> TensorApply t
__npi_degrees args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_degrees" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_diff" t =
     '[ '("n", AttrOpt Int), '("axis", AttrOpt Int), '("a", AttrOpt t)]

__npi_diff ::
           forall a t . (Tensor t, Fullfilled "__npi_diff" t a) =>
             ArgsHMap "__npi_diff" t a -> TensorApply t
__npi_diff args
  = let scalarArgs
          = catMaybes
              [("n",) . showValue <$> (args !? #n :: Maybe Int),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_npi_diff" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_dstack" t =
     '[ '("num_args", AttrReq Int), '("dim", AttrOpt Int),
        '("data", AttrOpt [t])]

__npi_dstack ::
             forall a t . (Tensor t, Fullfilled "__npi_dstack" t a) =>
               ArgsHMap "__npi_dstack" t a -> TensorApply t
__npi_dstack args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("dim",) . showValue <$> (args !? #dim :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "_npi_dstack" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__npi_einsum" t =
     '[ '("num_args", AttrReq Int), '("subscripts", AttrOpt Text),
        '("optimize", AttrOpt Int), '("data", AttrOpt [t])]

__npi_einsum ::
             forall a t . (Tensor t, Fullfilled "__npi_einsum" t a) =>
               ArgsHMap "__npi_einsum" t a -> TensorApply t
__npi_einsum args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("subscripts",) . showValue <$>
                 (args !? #subscripts :: Maybe Text),
               ("optimize",) . showValue <$> (args !? #optimize :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "_npi_einsum" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__npi_equal" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__npi_equal ::
            forall a t . (Tensor t, Fullfilled "__npi_equal" t a) =>
              ArgsHMap "__npi_equal" t a -> TensorApply t
__npi_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_npi_equal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_equal_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_equal_scalar ::
                   forall a t . (Tensor t, Fullfilled "__npi_equal_scalar" t a) =>
                     ArgsHMap "__npi_equal_scalar" t a -> TensorApply t
__npi_equal_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_equal_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_exp" t = '[ '("x", AttrOpt t)]

__npi_exp ::
          forall a t . (Tensor t, Fullfilled "__npi_exp" t a) =>
            ArgsHMap "__npi_exp" t a -> TensorApply t
__npi_exp args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_exp" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_expm1" t = '[ '("x", AttrOpt t)]

__npi_expm1 ::
            forall a t . (Tensor t, Fullfilled "__npi_expm1" t a) =>
              ArgsHMap "__npi_expm1" t a -> TensorApply t
__npi_expm1 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_expm1" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_fix" t = '[ '("x", AttrOpt t)]

__npi_fix ::
          forall a t . (Tensor t, Fullfilled "__npi_fix" t a) =>
            ArgsHMap "__npi_fix" t a -> TensorApply t
__npi_fix args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_fix" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_flip" t =
     '[ '("axis", AttrReq [Int]), '("data", AttrOpt t)]

__npi_flip ::
           forall a t . (Tensor t, Fullfilled "__npi_flip" t a) =>
             ArgsHMap "__npi_flip" t a -> TensorApply t
__npi_flip args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_flip" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_floor" t = '[ '("x", AttrOpt t)]

__npi_floor ::
            forall a t . (Tensor t, Fullfilled "__npi_floor" t a) =>
              ArgsHMap "__npi_floor" t a -> TensorApply t
__npi_floor args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_floor" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_greater" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__npi_greater ::
              forall a t . (Tensor t, Fullfilled "__npi_greater" t a) =>
                ArgsHMap "__npi_greater" t a -> TensorApply t
__npi_greater args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_npi_greater" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_greater_equal" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__npi_greater_equal ::
                    forall a t . (Tensor t, Fullfilled "__npi_greater_equal" t a) =>
                      ArgsHMap "__npi_greater_equal" t a -> TensorApply t
__npi_greater_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_npi_greater_equal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_greater_equal_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_greater_equal_scalar ::
                           forall a t .
                             (Tensor t, Fullfilled "__npi_greater_equal_scalar" t a) =>
                             ArgsHMap "__npi_greater_equal_scalar" t a -> TensorApply t
__npi_greater_equal_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in
      apply "_npi_greater_equal_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_greater_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_greater_scalar ::
                     forall a t . (Tensor t, Fullfilled "__npi_greater_scalar" t a) =>
                       ArgsHMap "__npi_greater_scalar" t a -> TensorApply t
__npi_greater_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_greater_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_hamming" t =
     '[ '("m", AttrOpt Int), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"]))]

__npi_hamming ::
              forall a t . (Tensor t, Fullfilled "__npi_hamming" t a) =>
                ArgsHMap "__npi_hamming" t a -> TensorApply t
__npi_hamming args
  = let scalarArgs
          = catMaybes
              [("m",) . showValue <$> (args !? #m :: Maybe Int),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["float16", "float32", "float64", "int32", "int64", "int8",
                           "uint8"]))]
        tensorKeyArgs = catMaybes []
      in apply "_npi_hamming" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_hanning" t =
     '[ '("m", AttrOpt Int), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"]))]

__npi_hanning ::
              forall a t . (Tensor t, Fullfilled "__npi_hanning" t a) =>
                ArgsHMap "__npi_hanning" t a -> TensorApply t
__npi_hanning args
  = let scalarArgs
          = catMaybes
              [("m",) . showValue <$> (args !? #m :: Maybe Int),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["float16", "float32", "float64", "int32", "int64", "int8",
                           "uint8"]))]
        tensorKeyArgs = catMaybes []
      in apply "_npi_hanning" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_hsplit" t =
     '[ '("indices", AttrReq [Int]), '("axis", AttrOpt Int),
        '("squeeze_axis", AttrOpt Bool), '("sections", AttrOpt Int),
        '("data", AttrOpt t)]

__npi_hsplit ::
             forall a t . (Tensor t, Fullfilled "__npi_hsplit" t a) =>
               ArgsHMap "__npi_hsplit" t a -> TensorApply t
__npi_hsplit args
  = let scalarArgs
          = catMaybes
              [("indices",) . showValue <$> (args !? #indices :: Maybe [Int]),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("squeeze_axis",) . showValue <$>
                 (args !? #squeeze_axis :: Maybe Bool),
               ("sections",) . showValue <$> (args !? #sections :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_hsplit" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_hsplit_backward" t = '[]

__npi_hsplit_backward ::
                      forall a t . (Tensor t, Fullfilled "__npi_hsplit_backward" t a) =>
                        ArgsHMap "__npi_hsplit_backward" t a -> TensorApply t
__npi_hsplit_backward args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_npi_hsplit_backward" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_hypot" t =
     '[ '("x1", AttrOpt t), '("x2", AttrOpt t)]

__npi_hypot ::
            forall a t . (Tensor t, Fullfilled "__npi_hypot" t a) =>
              ArgsHMap "__npi_hypot" t a -> TensorApply t
__npi_hypot args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("x1",) <$> (args !? #x1 :: Maybe t),
               ("x2",) <$> (args !? #x2 :: Maybe t)]
      in apply "_npi_hypot" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_identity" t =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bool", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"]))]

__npi_identity ::
               forall a t . (Tensor t, Fullfilled "__npi_identity" t a) =>
                 ArgsHMap "__npi_identity" t a -> TensorApply t
__npi_identity args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bool", "float16", "float32", "float64", "int32", "int64",
                           "int8", "uint8"]))]
        tensorKeyArgs = catMaybes []
      in apply "_npi_identity" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_indices" t =
     '[ '("dimensions", AttrReq [Int]),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"])),
        '("ctx", AttrOpt Text)]

__npi_indices ::
              forall a t . (Tensor t, Fullfilled "__npi_indices" t a) =>
                ArgsHMap "__npi_indices" t a -> TensorApply t
__npi_indices args
  = let scalarArgs
          = catMaybes
              [("dimensions",) . showValue <$>
                 (args !? #dimensions :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["float16", "float32", "float64", "int32", "int64", "int8",
                           "uint8"])),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text)]
        tensorKeyArgs = catMaybes []
      in apply "_npi_indices" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_lcm" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__npi_lcm ::
          forall a t . (Tensor t, Fullfilled "__npi_lcm" t a) =>
            ArgsHMap "__npi_lcm" t a -> TensorApply t
__npi_lcm args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_npi_lcm" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_lcm_scalar" t =
     '[ '("scalar", AttrOpt Int), '("data", AttrOpt t)]

__npi_lcm_scalar ::
                 forall a t . (Tensor t, Fullfilled "__npi_lcm_scalar" t a) =>
                   ArgsHMap "__npi_lcm_scalar" t a -> TensorApply t
__npi_lcm_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_lcm_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_ldexp" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__npi_ldexp ::
            forall a t . (Tensor t, Fullfilled "__npi_ldexp" t a) =>
              ArgsHMap "__npi_ldexp" t a -> TensorApply t
__npi_ldexp args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_npi_ldexp" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_ldexp_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_ldexp_scalar ::
                   forall a t . (Tensor t, Fullfilled "__npi_ldexp_scalar" t a) =>
                     ArgsHMap "__npi_ldexp_scalar" t a -> TensorApply t
__npi_ldexp_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_ldexp_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_less" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__npi_less ::
           forall a t . (Tensor t, Fullfilled "__npi_less" t a) =>
             ArgsHMap "__npi_less" t a -> TensorApply t
__npi_less args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_npi_less" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_less_equal" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__npi_less_equal ::
                 forall a t . (Tensor t, Fullfilled "__npi_less_equal" t a) =>
                   ArgsHMap "__npi_less_equal" t a -> TensorApply t
__npi_less_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_npi_less_equal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_less_equal_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_less_equal_scalar ::
                        forall a t .
                          (Tensor t, Fullfilled "__npi_less_equal_scalar" t a) =>
                          ArgsHMap "__npi_less_equal_scalar" t a -> TensorApply t
__npi_less_equal_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_less_equal_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_less_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_less_scalar ::
                  forall a t . (Tensor t, Fullfilled "__npi_less_scalar" t a) =>
                    ArgsHMap "__npi_less_scalar" t a -> TensorApply t
__npi_less_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_less_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_log" t = '[ '("x", AttrOpt t)]

__npi_log ::
          forall a t . (Tensor t, Fullfilled "__npi_log" t a) =>
            ArgsHMap "__npi_log" t a -> TensorApply t
__npi_log args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_log" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_log10" t = '[ '("x", AttrOpt t)]

__npi_log10 ::
            forall a t . (Tensor t, Fullfilled "__npi_log10" t a) =>
              ArgsHMap "__npi_log10" t a -> TensorApply t
__npi_log10 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_log10" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_log1p" t = '[ '("x", AttrOpt t)]

__npi_log1p ::
            forall a t . (Tensor t, Fullfilled "__npi_log1p" t a) =>
              ArgsHMap "__npi_log1p" t a -> TensorApply t
__npi_log1p args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_log1p" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_log2" t = '[ '("x", AttrOpt t)]

__npi_log2 ::
           forall a t . (Tensor t, Fullfilled "__npi_log2" t a) =>
             ArgsHMap "__npi_log2" t a -> TensorApply t
__npi_log2 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_log2" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_logical_not" t =
     '[ '("x", AttrOpt t)]

__npi_logical_not ::
                  forall a t . (Tensor t, Fullfilled "__npi_logical_not" t a) =>
                    ArgsHMap "__npi_logical_not" t a -> TensorApply t
__npi_logical_not args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_logical_not" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_logspace" t =
     '[ '("start", AttrReq Double), '("stop", AttrReq Double),
        '("num", AttrReq Int), '("endpoint", AttrOpt Bool),
        '("base", AttrOpt Double), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"]))]

__npi_logspace ::
               forall a t . (Tensor t, Fullfilled "__npi_logspace" t a) =>
                 ArgsHMap "__npi_logspace" t a -> TensorApply t
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
                         '["float16", "float32", "float64", "int32", "int64", "int8",
                           "uint8"]))]
        tensorKeyArgs = catMaybes []
      in apply "_npi_logspace" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_mean" t =
     '[ '("axis", AttrOpt (Maybe [Int])),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["bool", "float16", "float32", "float64", "int32", "int64",
                    "int8"]))),
        '("keepdims", AttrOpt Bool), '("initial", AttrOpt (Maybe Double)),
        '("a", AttrOpt t)]

__npi_mean ::
           forall a t . (Tensor t, Fullfilled "__npi_mean" t a) =>
             ArgsHMap "__npi_mean" t a -> TensorApply t
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
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_npi_mean" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_mod" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__npi_mod ::
          forall a t . (Tensor t, Fullfilled "__npi_mod" t a) =>
            ArgsHMap "__npi_mod" t a -> TensorApply t
__npi_mod args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_npi_mod" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_mod_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_mod_scalar ::
                 forall a t . (Tensor t, Fullfilled "__npi_mod_scalar" t a) =>
                   ArgsHMap "__npi_mod_scalar" t a -> TensorApply t
__npi_mod_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_mod_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_multinomial" t =
     '[ '("n", AttrReq Int), '("pvals", AttrOpt Int),
        '("size", AttrOpt (Maybe [Int])), '("a", AttrOpt t)]

__npi_multinomial ::
                  forall a t . (Tensor t, Fullfilled "__npi_multinomial" t a) =>
                    ArgsHMap "__npi_multinomial" t a -> TensorApply t
__npi_multinomial args
  = let scalarArgs
          = catMaybes
              [("n",) . showValue <$> (args !? #n :: Maybe Int),
               ("pvals",) . showValue <$> (args !? #pvals :: Maybe Int),
               ("size",) . showValue <$> (args !? #size :: Maybe (Maybe [Int]))]
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_npi_multinomial" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_multiply" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__npi_multiply ::
               forall a t . (Tensor t, Fullfilled "__npi_multiply" t a) =>
                 ArgsHMap "__npi_multiply" t a -> TensorApply t
__npi_multiply args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_npi_multiply" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_multiply_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_multiply_scalar ::
                      forall a t . (Tensor t, Fullfilled "__npi_multiply_scalar" t a) =>
                        ArgsHMap "__npi_multiply_scalar" t a -> TensorApply t
__npi_multiply_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_multiply_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_negative" t =
     '[ '("x", AttrOpt t)]

__npi_negative ::
               forall a t . (Tensor t, Fullfilled "__npi_negative" t a) =>
                 ArgsHMap "__npi_negative" t a -> TensorApply t
__npi_negative args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_negative" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_normal" t =
     '[ '("loc", AttrReq (Maybe Float)),
        '("scale", AttrReq (Maybe Float)),
        '("size", AttrOpt (Maybe [Int])), '("ctx", AttrOpt Text),
        '("dtype", AttrOpt (EnumType '["float16", "float32", "float64"])),
        '("input1", AttrOpt t), '("input2", AttrOpt t)]

__npi_normal ::
             forall a t . (Tensor t, Fullfilled "__npi_normal" t a) =>
               ArgsHMap "__npi_normal" t a -> TensorApply t
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
              [("input1",) <$> (args !? #input1 :: Maybe t),
               ("input2",) <$> (args !? #input2 :: Maybe t)]
      in apply "_npi_normal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_not_equal" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__npi_not_equal ::
                forall a t . (Tensor t, Fullfilled "__npi_not_equal" t a) =>
                  ArgsHMap "__npi_not_equal" t a -> TensorApply t
__npi_not_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_npi_not_equal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_not_equal_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_not_equal_scalar ::
                       forall a t . (Tensor t, Fullfilled "__npi_not_equal_scalar" t a) =>
                         ArgsHMap "__npi_not_equal_scalar" t a -> TensorApply t
__npi_not_equal_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_not_equal_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_ones" t =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bool", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"]))]

__npi_ones ::
           forall a t . (Tensor t, Fullfilled "__npi_ones" t a) =>
             ArgsHMap "__npi_ones" t a -> TensorApply t
__npi_ones args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bool", "float16", "float32", "float64", "int32", "int64",
                           "int8", "uint8"]))]
        tensorKeyArgs = catMaybes []
      in apply "_npi_ones" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_power" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__npi_power ::
            forall a t . (Tensor t, Fullfilled "__npi_power" t a) =>
              ArgsHMap "__npi_power" t a -> TensorApply t
__npi_power args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_npi_power" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_power_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_power_scalar ::
                   forall a t . (Tensor t, Fullfilled "__npi_power_scalar" t a) =>
                     ArgsHMap "__npi_power_scalar" t a -> TensorApply t
__npi_power_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_power_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_radians" t =
     '[ '("x", AttrOpt t)]

__npi_radians ::
              forall a t . (Tensor t, Fullfilled "__npi_radians" t a) =>
                ArgsHMap "__npi_radians" t a -> TensorApply t
__npi_radians args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_radians" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_rarctan2_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_rarctan2_scalar ::
                      forall a t . (Tensor t, Fullfilled "__npi_rarctan2_scalar" t a) =>
                        ArgsHMap "__npi_rarctan2_scalar" t a -> TensorApply t
__npi_rarctan2_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_rarctan2_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_rcopysign_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_rcopysign_scalar ::
                       forall a t . (Tensor t, Fullfilled "__npi_rcopysign_scalar" t a) =>
                         ArgsHMap "__npi_rcopysign_scalar" t a -> TensorApply t
__npi_rcopysign_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_rcopysign_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_reciprocal" t =
     '[ '("x", AttrOpt t)]

__npi_reciprocal ::
                 forall a t . (Tensor t, Fullfilled "__npi_reciprocal" t a) =>
                   ArgsHMap "__npi_reciprocal" t a -> TensorApply t
__npi_reciprocal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_reciprocal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_rint" t = '[ '("x", AttrOpt t)]

__npi_rint ::
           forall a t . (Tensor t, Fullfilled "__npi_rint" t a) =>
             ArgsHMap "__npi_rint" t a -> TensorApply t
__npi_rint args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_rint" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_rldexp_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_rldexp_scalar ::
                    forall a t . (Tensor t, Fullfilled "__npi_rldexp_scalar" t a) =>
                      ArgsHMap "__npi_rldexp_scalar" t a -> TensorApply t
__npi_rldexp_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_rldexp_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_rmod_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_rmod_scalar ::
                  forall a t . (Tensor t, Fullfilled "__npi_rmod_scalar" t a) =>
                    ArgsHMap "__npi_rmod_scalar" t a -> TensorApply t
__npi_rmod_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_rmod_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_rot90" t =
     '[ '("k", AttrOpt Int), '("axes", AttrOpt (Maybe [Int])),
        '("data", AttrOpt t)]

__npi_rot90 ::
            forall a t . (Tensor t, Fullfilled "__npi_rot90" t a) =>
              ArgsHMap "__npi_rot90" t a -> TensorApply t
__npi_rot90 args
  = let scalarArgs
          = catMaybes
              [("k",) . showValue <$> (args !? #k :: Maybe Int),
               ("axes",) . showValue <$> (args !? #axes :: Maybe (Maybe [Int]))]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_rot90" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_rpower_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_rpower_scalar ::
                    forall a t . (Tensor t, Fullfilled "__npi_rpower_scalar" t a) =>
                      ArgsHMap "__npi_rpower_scalar" t a -> TensorApply t
__npi_rpower_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_rpower_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_rsubtract_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_rsubtract_scalar ::
                       forall a t . (Tensor t, Fullfilled "__npi_rsubtract_scalar" t a) =>
                         ArgsHMap "__npi_rsubtract_scalar" t a -> TensorApply t
__npi_rsubtract_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_rsubtract_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_rtrue_divide_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_rtrue_divide_scalar ::
                          forall a t .
                            (Tensor t, Fullfilled "__npi_rtrue_divide_scalar" t a) =>
                            ArgsHMap "__npi_rtrue_divide_scalar" t a -> TensorApply t
__npi_rtrue_divide_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_rtrue_divide_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_share_memory" t =
     '[ '("a", AttrOpt t), '("b", AttrOpt t)]

__npi_share_memory ::
                   forall a t . (Tensor t, Fullfilled "__npi_share_memory" t a) =>
                     ArgsHMap "__npi_share_memory" t a -> TensorApply t
__npi_share_memory args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("a",) <$> (args !? #a :: Maybe t),
               ("b",) <$> (args !? #b :: Maybe t)]
      in apply "_npi_share_memory" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_sign" t = '[ '("x", AttrOpt t)]

__npi_sign ::
           forall a t . (Tensor t, Fullfilled "__npi_sign" t a) =>
             ArgsHMap "__npi_sign" t a -> TensorApply t
__npi_sign args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_sign" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_sin" t = '[ '("x", AttrOpt t)]

__npi_sin ::
          forall a t . (Tensor t, Fullfilled "__npi_sin" t a) =>
            ArgsHMap "__npi_sin" t a -> TensorApply t
__npi_sin args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_sin" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_sinh" t = '[ '("x", AttrOpt t)]

__npi_sinh ::
           forall a t . (Tensor t, Fullfilled "__npi_sinh" t a) =>
             ArgsHMap "__npi_sinh" t a -> TensorApply t
__npi_sinh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_sinh" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_sqrt" t = '[ '("x", AttrOpt t)]

__npi_sqrt ::
           forall a t . (Tensor t, Fullfilled "__npi_sqrt" t a) =>
             ArgsHMap "__npi_sqrt" t a -> TensorApply t
__npi_sqrt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_sqrt" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_square" t =
     '[ '("x", AttrOpt t)]

__npi_square ::
             forall a t . (Tensor t, Fullfilled "__npi_square" t a) =>
               ArgsHMap "__npi_square" t a -> TensorApply t
__npi_square args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_square" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_stack" t =
     '[ '("axis", AttrOpt Int), '("num_args", AttrReq Int),
        '("data", AttrOpt [t])]

__npi_stack ::
            forall a t . (Tensor t, Fullfilled "__npi_stack" t a) =>
              ArgsHMap "__npi_stack" t a -> TensorApply t
__npi_stack args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("num_args",) . showValue <$> (args !? #num_args :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "_npi_stack" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__npi_std" t =
     '[ '("axis", AttrOpt (Maybe [Int])),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["float16", "float32", "float64", "int32", "int64", "int8"]))),
        '("ddof", AttrOpt Int), '("keepdims", AttrOpt Bool),
        '("a", AttrOpt t)]

__npi_std ::
          forall a t . (Tensor t, Fullfilled "__npi_std" t a) =>
            ArgsHMap "__npi_std" t a -> TensorApply t
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
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_npi_std" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_subtract" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__npi_subtract ::
               forall a t . (Tensor t, Fullfilled "__npi_subtract" t a) =>
                 ArgsHMap "__npi_subtract" t a -> TensorApply t
__npi_subtract args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_npi_subtract" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_subtract_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_subtract_scalar ::
                      forall a t . (Tensor t, Fullfilled "__npi_subtract_scalar" t a) =>
                        ArgsHMap "__npi_subtract_scalar" t a -> TensorApply t
__npi_subtract_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_subtract_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_svd" t = '[ '("a", AttrOpt t)]

__npi_svd ::
          forall a t . (Tensor t, Fullfilled "__npi_svd" t a) =>
            ArgsHMap "__npi_svd" t a -> TensorApply t
__npi_svd args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_npi_svd" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_tan" t = '[ '("x", AttrOpt t)]

__npi_tan ::
          forall a t . (Tensor t, Fullfilled "__npi_tan" t a) =>
            ArgsHMap "__npi_tan" t a -> TensorApply t
__npi_tan args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_tan" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_tanh" t = '[ '("x", AttrOpt t)]

__npi_tanh ::
           forall a t . (Tensor t, Fullfilled "__npi_tanh" t a) =>
             ArgsHMap "__npi_tanh" t a -> TensorApply t
__npi_tanh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_tanh" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_tensordot" t =
     '[ '("a_axes_summed", AttrReq [Int]),
        '("b_axes_summed", AttrReq [Int]), '("a", AttrOpt t),
        '("b", AttrOpt t)]

__npi_tensordot ::
                forall a t . (Tensor t, Fullfilled "__npi_tensordot" t a) =>
                  ArgsHMap "__npi_tensordot" t a -> TensorApply t
__npi_tensordot args
  = let scalarArgs
          = catMaybes
              [("a_axes_summed",) . showValue <$>
                 (args !? #a_axes_summed :: Maybe [Int]),
               ("b_axes_summed",) . showValue <$>
                 (args !? #b_axes_summed :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes
              [("a",) <$> (args !? #a :: Maybe t),
               ("b",) <$> (args !? #b :: Maybe t)]
      in apply "_npi_tensordot" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_tensordot_int_axes" t =
     '[ '("axes", AttrReq Int), '("a", AttrOpt t), '("b", AttrOpt t)]

__npi_tensordot_int_axes ::
                         forall a t .
                           (Tensor t, Fullfilled "__npi_tensordot_int_axes" t a) =>
                           ArgsHMap "__npi_tensordot_int_axes" t a -> TensorApply t
__npi_tensordot_int_axes args
  = let scalarArgs
          = catMaybes
              [("axes",) . showValue <$> (args !? #axes :: Maybe Int)]
        tensorKeyArgs
          = catMaybes
              [("a",) <$> (args !? #a :: Maybe t),
               ("b",) <$> (args !? #b :: Maybe t)]
      in apply "_npi_tensordot_int_axes" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_tril" t =
     '[ '("k", AttrOpt Int), '("data", AttrOpt t)]

__npi_tril ::
           forall a t . (Tensor t, Fullfilled "__npi_tril" t a) =>
             ArgsHMap "__npi_tril" t a -> TensorApply t
__npi_tril args
  = let scalarArgs
          = catMaybes [("k",) . showValue <$> (args !? #k :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_tril" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_true_divide" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__npi_true_divide ::
                  forall a t . (Tensor t, Fullfilled "__npi_true_divide" t a) =>
                    ArgsHMap "__npi_true_divide" t a -> TensorApply t
__npi_true_divide args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_npi_true_divide" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_true_divide_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__npi_true_divide_scalar ::
                         forall a t .
                           (Tensor t, Fullfilled "__npi_true_divide_scalar" t a) =>
                           ArgsHMap "__npi_true_divide_scalar" t a -> TensorApply t
__npi_true_divide_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_true_divide_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_trunc" t = '[ '("x", AttrOpt t)]

__npi_trunc ::
            forall a t . (Tensor t, Fullfilled "__npi_trunc" t a) =>
              ArgsHMap "__npi_trunc" t a -> TensorApply t
__npi_trunc args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npi_trunc" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_uniform" t =
     '[ '("low", AttrReq (Maybe Float)),
        '("high", AttrReq (Maybe Float)), '("size", AttrOpt (Maybe [Int])),
        '("ctx", AttrOpt Text),
        '("dtype", AttrOpt (EnumType '["float16", "float32", "float64"])),
        '("input1", AttrOpt t), '("input2", AttrOpt t)]

__npi_uniform ::
              forall a t . (Tensor t, Fullfilled "__npi_uniform" t a) =>
                ArgsHMap "__npi_uniform" t a -> TensorApply t
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
              [("input1",) <$> (args !? #input1 :: Maybe t),
               ("input2",) <$> (args !? #input2 :: Maybe t)]
      in apply "_npi_uniform" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_unique" t =
     '[ '("return_index", AttrOpt Bool),
        '("return_inverse", AttrOpt Bool),
        '("return_counts", AttrOpt Bool), '("axis", AttrOpt (Maybe Int)),
        '("data", AttrOpt t)]

__npi_unique ::
             forall a t . (Tensor t, Fullfilled "__npi_unique" t a) =>
               ArgsHMap "__npi_unique" t a -> TensorApply t
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
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npi_unique" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_var" t =
     '[ '("axis", AttrOpt (Maybe [Int])),
        '("dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["float16", "float32", "float64", "int32", "int64", "int8"]))),
        '("ddof", AttrOpt Int), '("keepdims", AttrOpt Bool),
        '("a", AttrOpt t)]

__npi_var ::
          forall a t . (Tensor t, Fullfilled "__npi_var" t a) =>
            ArgsHMap "__npi_var" t a -> TensorApply t
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
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_npi_var" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npi_vstack" t =
     '[ '("num_args", AttrReq Int), '("data", AttrOpt [t])]

__npi_vstack ::
             forall a t . (Tensor t, Fullfilled "__npi_vstack" t a) =>
               ArgsHMap "__npi_vstack" t a -> TensorApply t
__npi_vstack args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "_npi_vstack" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__npi_zeros" t =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bool", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"]))]

__npi_zeros ::
            forall a t . (Tensor t, Fullfilled "__npi_zeros" t a) =>
              ArgsHMap "__npi_zeros" t a -> TensorApply t
__npi_zeros args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bool", "float16", "float32", "float64", "int32", "int64",
                           "int8", "uint8"]))]
        tensorKeyArgs = catMaybes []
      in apply "_npi_zeros" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npx_nonzero" t =
     '[ '("x", AttrOpt t)]

__npx_nonzero ::
              forall a t . (Tensor t, Fullfilled "__npx_nonzero" t a) =>
                ArgsHMap "__npx_nonzero" t a -> TensorApply t
__npx_nonzero args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes [("x",) <$> (args !? #x :: Maybe t)]
      in apply "_npx_nonzero" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npx_relu" t =
     '[ '("data", AttrOpt t)]

__npx_relu ::
           forall a t . (Tensor t, Fullfilled "__npx_relu" t a) =>
             ArgsHMap "__npx_relu" t a -> TensorApply t
__npx_relu args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npx_relu" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npx_reshape" t =
     '[ '("newshape", AttrReq [Int]), '("reverse", AttrOpt Bool),
        '("order", AttrOpt Text), '("a", AttrOpt t)]

__npx_reshape ::
              forall a t . (Tensor t, Fullfilled "__npx_reshape" t a) =>
                ArgsHMap "__npx_reshape" t a -> TensorApply t
__npx_reshape args
  = let scalarArgs
          = catMaybes
              [("newshape",) . showValue <$> (args !? #newshape :: Maybe [Int]),
               ("reverse",) . showValue <$> (args !? #reverse :: Maybe Bool),
               ("order",) . showValue <$> (args !? #order :: Maybe Text)]
        tensorKeyArgs = catMaybes [("a",) <$> (args !? #a :: Maybe t)]
      in apply "_npx_reshape" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__npx_sigmoid" t =
     '[ '("data", AttrOpt t)]

__npx_sigmoid ::
              forall a t . (Tensor t, Fullfilled "__npx_sigmoid" t a) =>
                ArgsHMap "__npx_sigmoid" t a -> TensorApply t
__npx_sigmoid args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_npx_sigmoid" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__onehot_encode" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__onehot_encode ::
                forall a t . (Tensor t, Fullfilled "__onehot_encode" t a) =>
                  ArgsHMap "__onehot_encode" t a -> TensorApply t
__onehot_encode args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_onehot_encode" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__ones" t =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bool", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"]))]

__ones ::
       forall a t . (Tensor t, Fullfilled "__ones" t a) =>
         ArgsHMap "__ones" t a -> TensorApply t
__ones args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bool", "float16", "float32", "float64", "int32", "int64",
                           "int8", "uint8"]))]
        tensorKeyArgs = catMaybes []
      in apply "_ones" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__plus_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__plus_scalar ::
              forall a t . (Tensor t, Fullfilled "__plus_scalar" t a) =>
                ArgsHMap "__plus_scalar" t a -> TensorApply t
__plus_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_plus_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__power" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__power ::
        forall a t . (Tensor t, Fullfilled "__power" t a) =>
          ArgsHMap "__power" t a -> TensorApply t
__power args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_power" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__power_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__power_scalar ::
               forall a t . (Tensor t, Fullfilled "__power_scalar" t a) =>
                 ArgsHMap "__power_scalar" t a -> TensorApply t
__power_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_power_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__random_exponential" t =
     '[ '("lam", AttrOpt Float), '("shape", AttrOpt [Int]),
        '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

__random_exponential ::
                     forall a t . (Tensor t, Fullfilled "__random_exponential" t a) =>
                       ArgsHMap "__random_exponential" t a -> TensorApply t
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
      in apply "_random_exponential" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__random_exponential_like" t =
     '[ '("lam", AttrOpt Float), '("data", AttrOpt t)]

__random_exponential_like ::
                          forall a t .
                            (Tensor t, Fullfilled "__random_exponential_like" t a) =>
                            ArgsHMap "__random_exponential_like" t a -> TensorApply t
__random_exponential_like args
  = let scalarArgs
          = catMaybes
              [("lam",) . showValue <$> (args !? #lam :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_random_exponential_like" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__random_gamma" t =
     '[ '("alpha", AttrOpt Float), '("beta", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

__random_gamma ::
               forall a t . (Tensor t, Fullfilled "__random_gamma" t a) =>
                 ArgsHMap "__random_gamma" t a -> TensorApply t
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
      in apply "_random_gamma" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__random_gamma_like" t =
     '[ '("alpha", AttrOpt Float), '("beta", AttrOpt Float),
        '("data", AttrOpt t)]

__random_gamma_like ::
                    forall a t . (Tensor t, Fullfilled "__random_gamma_like" t a) =>
                      ArgsHMap "__random_gamma_like" t a -> TensorApply t
__random_gamma_like args
  = let scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> (args !? #alpha :: Maybe Float),
               ("beta",) . showValue <$> (args !? #beta :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_random_gamma_like" scalarArgs (Left tensorKeyArgs)

type instance
     ParameterList "__random_generalized_negative_binomial" t =
     '[ '("mu", AttrOpt Float), '("alpha", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

__random_generalized_negative_binomial ::
                                       forall a t .
                                         (Tensor t,
                                          Fullfilled "__random_generalized_negative_binomial" t
                                            a) =>
                                         ArgsHMap "__random_generalized_negative_binomial" t a ->
                                           TensorApply t
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
      apply "_random_generalized_negative_binomial" scalarArgs
        (Left tensorKeyArgs)

type instance
     ParameterList "__random_generalized_negative_binomial_like" t =
     '[ '("mu", AttrOpt Float), '("alpha", AttrOpt Float),
        '("data", AttrOpt t)]

__random_generalized_negative_binomial_like ::
                                            forall a t .
                                              (Tensor t,
                                               Fullfilled
                                                 "__random_generalized_negative_binomial_like" t
                                                 a) =>
                                              ArgsHMap "__random_generalized_negative_binomial_like"
                                                t
                                                a
                                                -> TensorApply t
__random_generalized_negative_binomial_like args
  = let scalarArgs
          = catMaybes
              [("mu",) . showValue <$> (args !? #mu :: Maybe Float),
               ("alpha",) . showValue <$> (args !? #alpha :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in
      apply "_random_generalized_negative_binomial_like" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__random_negative_binomial" t =
     '[ '("k", AttrOpt Int), '("p", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

__random_negative_binomial ::
                           forall a t .
                             (Tensor t, Fullfilled "__random_negative_binomial" t a) =>
                             ArgsHMap "__random_negative_binomial" t a -> TensorApply t
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
      apply "_random_negative_binomial" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__random_negative_binomial_like" t =
     '[ '("k", AttrOpt Int), '("p", AttrOpt Float),
        '("data", AttrOpt t)]

__random_negative_binomial_like ::
                                forall a t .
                                  (Tensor t, Fullfilled "__random_negative_binomial_like" t a) =>
                                  ArgsHMap "__random_negative_binomial_like" t a -> TensorApply t
__random_negative_binomial_like args
  = let scalarArgs
          = catMaybes
              [("k",) . showValue <$> (args !? #k :: Maybe Int),
               ("p",) . showValue <$> (args !? #p :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in
      apply "_random_negative_binomial_like" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__random_normal" t =
     '[ '("loc", AttrOpt Float), '("scale", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

__random_normal ::
                forall a t . (Tensor t, Fullfilled "__random_normal" t a) =>
                  ArgsHMap "__random_normal" t a -> TensorApply t
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
      in apply "_random_normal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__random_normal_like" t =
     '[ '("loc", AttrOpt Float), '("scale", AttrOpt Float),
        '("data", AttrOpt t)]

__random_normal_like ::
                     forall a t . (Tensor t, Fullfilled "__random_normal_like" t a) =>
                       ArgsHMap "__random_normal_like" t a -> TensorApply t
__random_normal_like args
  = let scalarArgs
          = catMaybes
              [("loc",) . showValue <$> (args !? #loc :: Maybe Float),
               ("scale",) . showValue <$> (args !? #scale :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_random_normal_like" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__random_pdf_dirichlet" t =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt t),
        '("alpha", AttrOpt t)]

__random_pdf_dirichlet ::
                       forall a t . (Tensor t, Fullfilled "__random_pdf_dirichlet" t a) =>
                         ArgsHMap "__random_pdf_dirichlet" t a -> TensorApply t
__random_pdf_dirichlet args
  = let scalarArgs
          = catMaybes
              [("is_log",) . showValue <$> (args !? #is_log :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("sample",) <$> (args !? #sample :: Maybe t),
               ("alpha",) <$> (args !? #alpha :: Maybe t)]
      in apply "_random_pdf_dirichlet" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__random_pdf_exponential" t =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt t),
        '("lam", AttrOpt t)]

__random_pdf_exponential ::
                         forall a t .
                           (Tensor t, Fullfilled "__random_pdf_exponential" t a) =>
                           ArgsHMap "__random_pdf_exponential" t a -> TensorApply t
__random_pdf_exponential args
  = let scalarArgs
          = catMaybes
              [("is_log",) . showValue <$> (args !? #is_log :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("sample",) <$> (args !? #sample :: Maybe t),
               ("lam",) <$> (args !? #lam :: Maybe t)]
      in apply "_random_pdf_exponential" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__random_pdf_gamma" t =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt t),
        '("alpha", AttrOpt t), '("beta", AttrOpt t)]

__random_pdf_gamma ::
                   forall a t . (Tensor t, Fullfilled "__random_pdf_gamma" t a) =>
                     ArgsHMap "__random_pdf_gamma" t a -> TensorApply t
__random_pdf_gamma args
  = let scalarArgs
          = catMaybes
              [("is_log",) . showValue <$> (args !? #is_log :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("sample",) <$> (args !? #sample :: Maybe t),
               ("alpha",) <$> (args !? #alpha :: Maybe t),
               ("beta",) <$> (args !? #beta :: Maybe t)]
      in apply "_random_pdf_gamma" scalarArgs (Left tensorKeyArgs)

type instance
     ParameterList "__random_pdf_generalized_negative_binomial" t =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt t),
        '("mu", AttrOpt t), '("alpha", AttrOpt t)]

__random_pdf_generalized_negative_binomial ::
                                           forall a t .
                                             (Tensor t,
                                              Fullfilled
                                                "__random_pdf_generalized_negative_binomial" t a) =>
                                             ArgsHMap "__random_pdf_generalized_negative_binomial" t
                                               a
                                               -> TensorApply t
__random_pdf_generalized_negative_binomial args
  = let scalarArgs
          = catMaybes
              [("is_log",) . showValue <$> (args !? #is_log :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("sample",) <$> (args !? #sample :: Maybe t),
               ("mu",) <$> (args !? #mu :: Maybe t),
               ("alpha",) <$> (args !? #alpha :: Maybe t)]
      in
      apply "_random_pdf_generalized_negative_binomial" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__random_pdf_negative_binomial" t =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt t),
        '("k", AttrOpt t), '("p", AttrOpt t)]

__random_pdf_negative_binomial ::
                               forall a t .
                                 (Tensor t, Fullfilled "__random_pdf_negative_binomial" t a) =>
                                 ArgsHMap "__random_pdf_negative_binomial" t a -> TensorApply t
__random_pdf_negative_binomial args
  = let scalarArgs
          = catMaybes
              [("is_log",) . showValue <$> (args !? #is_log :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("sample",) <$> (args !? #sample :: Maybe t),
               ("k",) <$> (args !? #k :: Maybe t),
               ("p",) <$> (args !? #p :: Maybe t)]
      in
      apply "_random_pdf_negative_binomial" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__random_pdf_normal" t =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt t),
        '("mu", AttrOpt t), '("sigma", AttrOpt t)]

__random_pdf_normal ::
                    forall a t . (Tensor t, Fullfilled "__random_pdf_normal" t a) =>
                      ArgsHMap "__random_pdf_normal" t a -> TensorApply t
__random_pdf_normal args
  = let scalarArgs
          = catMaybes
              [("is_log",) . showValue <$> (args !? #is_log :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("sample",) <$> (args !? #sample :: Maybe t),
               ("mu",) <$> (args !? #mu :: Maybe t),
               ("sigma",) <$> (args !? #sigma :: Maybe t)]
      in apply "_random_pdf_normal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__random_pdf_poisson" t =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt t),
        '("lam", AttrOpt t)]

__random_pdf_poisson ::
                     forall a t . (Tensor t, Fullfilled "__random_pdf_poisson" t a) =>
                       ArgsHMap "__random_pdf_poisson" t a -> TensorApply t
__random_pdf_poisson args
  = let scalarArgs
          = catMaybes
              [("is_log",) . showValue <$> (args !? #is_log :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("sample",) <$> (args !? #sample :: Maybe t),
               ("lam",) <$> (args !? #lam :: Maybe t)]
      in apply "_random_pdf_poisson" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__random_pdf_uniform" t =
     '[ '("is_log", AttrOpt Bool), '("sample", AttrOpt t),
        '("low", AttrOpt t), '("high", AttrOpt t)]

__random_pdf_uniform ::
                     forall a t . (Tensor t, Fullfilled "__random_pdf_uniform" t a) =>
                       ArgsHMap "__random_pdf_uniform" t a -> TensorApply t
__random_pdf_uniform args
  = let scalarArgs
          = catMaybes
              [("is_log",) . showValue <$> (args !? #is_log :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes
              [("sample",) <$> (args !? #sample :: Maybe t),
               ("low",) <$> (args !? #low :: Maybe t),
               ("high",) <$> (args !? #high :: Maybe t)]
      in apply "_random_pdf_uniform" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__random_poisson" t =
     '[ '("lam", AttrOpt Float), '("shape", AttrOpt [Int]),
        '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

__random_poisson ::
                 forall a t . (Tensor t, Fullfilled "__random_poisson" t a) =>
                   ArgsHMap "__random_poisson" t a -> TensorApply t
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
      in apply "_random_poisson" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__random_poisson_like" t =
     '[ '("lam", AttrOpt Float), '("data", AttrOpt t)]

__random_poisson_like ::
                      forall a t . (Tensor t, Fullfilled "__random_poisson_like" t a) =>
                        ArgsHMap "__random_poisson_like" t a -> TensorApply t
__random_poisson_like args
  = let scalarArgs
          = catMaybes
              [("lam",) . showValue <$> (args !? #lam :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_random_poisson_like" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__random_uniform" t =
     '[ '("low", AttrOpt Float), '("high", AttrOpt Float),
        '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"]))]

__random_uniform ::
                 forall a t . (Tensor t, Fullfilled "__random_uniform" t a) =>
                   ArgsHMap "__random_uniform" t a -> TensorApply t
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
      in apply "_random_uniform" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__random_uniform_like" t =
     '[ '("low", AttrOpt Float), '("high", AttrOpt Float),
        '("data", AttrOpt t)]

__random_uniform_like ::
                      forall a t . (Tensor t, Fullfilled "__random_uniform_like" t a) =>
                        ArgsHMap "__random_uniform_like" t a -> TensorApply t
__random_uniform_like args
  = let scalarArgs
          = catMaybes
              [("low",) . showValue <$> (args !? #low :: Maybe Float),
               ("high",) . showValue <$> (args !? #high :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_random_uniform_like" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__ravel_multi_index" t =
     '[ '("shape", AttrOpt [Int]), '("data", AttrOpt t)]

__ravel_multi_index ::
                    forall a t . (Tensor t, Fullfilled "__ravel_multi_index" t a) =>
                      ArgsHMap "__ravel_multi_index" t a -> TensorApply t
__ravel_multi_index args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_ravel_multi_index" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__rdiv_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__rdiv_scalar ::
              forall a t . (Tensor t, Fullfilled "__rdiv_scalar" t a) =>
                ArgsHMap "__rdiv_scalar" t a -> TensorApply t
__rdiv_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_rdiv_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__rminus_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__rminus_scalar ::
                forall a t . (Tensor t, Fullfilled "__rminus_scalar" t a) =>
                  ArgsHMap "__rminus_scalar" t a -> TensorApply t
__rminus_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_rminus_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__rmod_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__rmod_scalar ::
              forall a t . (Tensor t, Fullfilled "__rmod_scalar" t a) =>
                ArgsHMap "__rmod_scalar" t a -> TensorApply t
__rmod_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_rmod_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__rnn_param_concat" t =
     '[ '("num_args", AttrReq Int), '("dim", AttrOpt Int),
        '("data", AttrOpt [t])]

__rnn_param_concat ::
                   forall a t . (Tensor t, Fullfilled "__rnn_param_concat" t a) =>
                     ArgsHMap "__rnn_param_concat" t a -> TensorApply t
__rnn_param_concat args
  = let scalarArgs
          = catMaybes
              [("num_args",) . showValue <$> (args !? #num_args :: Maybe Int),
               ("dim",) . showValue <$> (args !? #dim :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "_rnn_param_concat" scalarArgs (Right tensorVarArgs)

type instance ParameterList "__rpower_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__rpower_scalar ::
                forall a t . (Tensor t, Fullfilled "__rpower_scalar" t a) =>
                  ArgsHMap "__rpower_scalar" t a -> TensorApply t
__rpower_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_rpower_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__sample_exponential" t =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("lam", AttrOpt t)]

__sample_exponential ::
                     forall a t . (Tensor t, Fullfilled "__sample_exponential" t a) =>
                       ArgsHMap "__sample_exponential" t a -> TensorApply t
__sample_exponential args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorKeyArgs = catMaybes [("lam",) <$> (args !? #lam :: Maybe t)]
      in apply "_sample_exponential" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__sample_gamma" t =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("alpha", AttrOpt t), '("beta", AttrOpt t)]

__sample_gamma ::
               forall a t . (Tensor t, Fullfilled "__sample_gamma" t a) =>
                 ArgsHMap "__sample_gamma" t a -> TensorApply t
__sample_gamma args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorKeyArgs
          = catMaybes
              [("alpha",) <$> (args !? #alpha :: Maybe t),
               ("beta",) <$> (args !? #beta :: Maybe t)]
      in apply "_sample_gamma" scalarArgs (Left tensorKeyArgs)

type instance
     ParameterList "__sample_generalized_negative_binomial" t =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("mu", AttrOpt t), '("alpha", AttrOpt t)]

__sample_generalized_negative_binomial ::
                                       forall a t .
                                         (Tensor t,
                                          Fullfilled "__sample_generalized_negative_binomial" t
                                            a) =>
                                         ArgsHMap "__sample_generalized_negative_binomial" t a ->
                                           TensorApply t
__sample_generalized_negative_binomial args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorKeyArgs
          = catMaybes
              [("mu",) <$> (args !? #mu :: Maybe t),
               ("alpha",) <$> (args !? #alpha :: Maybe t)]
      in
      apply "_sample_generalized_negative_binomial" scalarArgs
        (Left tensorKeyArgs)

type instance ParameterList "__sample_multinomial" t =
     '[ '("shape", AttrOpt [Int]), '("get_prob", AttrOpt Bool),
        '("dtype",
          AttrOpt
            (EnumType '["float16", "float32", "float64", "int32", "uint8"])),
        '("data", AttrOpt t)]

__sample_multinomial ::
                     forall a t . (Tensor t, Fullfilled "__sample_multinomial" t a) =>
                       ArgsHMap "__sample_multinomial" t a -> TensorApply t
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
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_sample_multinomial" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__sample_negative_binomial" t =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("k", AttrOpt t), '("p", AttrOpt t)]

__sample_negative_binomial ::
                           forall a t .
                             (Tensor t, Fullfilled "__sample_negative_binomial" t a) =>
                             ArgsHMap "__sample_negative_binomial" t a -> TensorApply t
__sample_negative_binomial args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorKeyArgs
          = catMaybes
              [("k",) <$> (args !? #k :: Maybe t),
               ("p",) <$> (args !? #p :: Maybe t)]
      in
      apply "_sample_negative_binomial" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__sample_normal" t =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("mu", AttrOpt t), '("sigma", AttrOpt t)]

__sample_normal ::
                forall a t . (Tensor t, Fullfilled "__sample_normal" t a) =>
                  ArgsHMap "__sample_normal" t a -> TensorApply t
__sample_normal args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorKeyArgs
          = catMaybes
              [("mu",) <$> (args !? #mu :: Maybe t),
               ("sigma",) <$> (args !? #sigma :: Maybe t)]
      in apply "_sample_normal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__sample_poisson" t =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("lam", AttrOpt t)]

__sample_poisson ::
                 forall a t . (Tensor t, Fullfilled "__sample_poisson" t a) =>
                   ArgsHMap "__sample_poisson" t a -> TensorApply t
__sample_poisson args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorKeyArgs = catMaybes [("lam",) <$> (args !? #lam :: Maybe t)]
      in apply "_sample_poisson" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__sample_uniform" t =
     '[ '("shape", AttrOpt [Int]),
        '("dtype",
          AttrOpt (EnumType '["None", "float16", "float32", "float64"])),
        '("low", AttrOpt t), '("high", AttrOpt t)]

__sample_uniform ::
                 forall a t . (Tensor t, Fullfilled "__sample_uniform" t a) =>
                   ArgsHMap "__sample_uniform" t a -> TensorApply t
__sample_uniform args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe (EnumType '["None", "float16", "float32", "float64"]))]
        tensorKeyArgs
          = catMaybes
              [("low",) <$> (args !? #low :: Maybe t),
               ("high",) <$> (args !? #high :: Maybe t)]
      in apply "_sample_uniform" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__sample_unique_zipfian" t =
     '[ '("range_max", AttrReq Int), '("shape", AttrOpt [Int])]

__sample_unique_zipfian ::
                        forall a t .
                          (Tensor t, Fullfilled "__sample_unique_zipfian" t a) =>
                          ArgsHMap "__sample_unique_zipfian" t a -> TensorApply t
__sample_unique_zipfian args
  = let scalarArgs
          = catMaybes
              [("range_max",) . showValue <$> (args !? #range_max :: Maybe Int),
               ("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorKeyArgs = catMaybes []
      in apply "_sample_unique_zipfian" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__scatter_elemwise_div" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__scatter_elemwise_div ::
                       forall a t . (Tensor t, Fullfilled "__scatter_elemwise_div" t a) =>
                         ArgsHMap "__scatter_elemwise_div" t a -> TensorApply t
__scatter_elemwise_div args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_scatter_elemwise_div" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__scatter_minus_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__scatter_minus_scalar ::
                       forall a t . (Tensor t, Fullfilled "__scatter_minus_scalar" t a) =>
                         ArgsHMap "__scatter_minus_scalar" t a -> TensorApply t
__scatter_minus_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_scatter_minus_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__scatter_plus_scalar" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

__scatter_plus_scalar ::
                      forall a t . (Tensor t, Fullfilled "__scatter_plus_scalar" t a) =>
                        ArgsHMap "__scatter_plus_scalar" t a -> TensorApply t
__scatter_plus_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_scatter_plus_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__scatter_set_nd" t =
     '[ '("shape", AttrReq [Int]), '("lhs", AttrOpt t),
        '("rhs", AttrOpt t), '("indices", AttrOpt t)]

__scatter_set_nd ::
                 forall a t . (Tensor t, Fullfilled "__scatter_set_nd" t a) =>
                   ArgsHMap "__scatter_set_nd" t a -> TensorApply t
__scatter_set_nd args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t),
               ("indices",) <$> (args !? #indices :: Maybe t)]
      in apply "_scatter_set_nd" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__set_value" t =
     '[ '("src", AttrOpt Float)]

__set_value ::
            forall a t . (Tensor t, Fullfilled "__set_value" t a) =>
              ArgsHMap "__set_value" t a -> TensorApply t
__set_value args
  = let scalarArgs
          = catMaybes
              [("src",) . showValue <$> (args !? #src :: Maybe Float)]
        tensorKeyArgs = catMaybes []
      in apply "_set_value" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__sg_mkldnn_conv" t = '[]

__sg_mkldnn_conv ::
                 forall a t . (Tensor t, Fullfilled "__sg_mkldnn_conv" t a) =>
                   ArgsHMap "__sg_mkldnn_conv" t a -> TensorApply t
__sg_mkldnn_conv args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_sg_mkldnn_conv" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__sg_mkldnn_fully_connected" t = '[]

__sg_mkldnn_fully_connected ::
                            forall a t .
                              (Tensor t, Fullfilled "__sg_mkldnn_fully_connected" t a) =>
                              ArgsHMap "__sg_mkldnn_fully_connected" t a -> TensorApply t
__sg_mkldnn_fully_connected args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in
      apply "_sg_mkldnn_fully_connected" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__shuffle" t =
     '[ '("data", AttrOpt t)]

__shuffle ::
          forall a t . (Tensor t, Fullfilled "__shuffle" t a) =>
            ArgsHMap "__shuffle" t a -> TensorApply t
__shuffle args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_shuffle" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__slice_assign" t =
     '[ '("begin", AttrReq [Int]), '("end", AttrReq [Int]),
        '("step", AttrOpt [Int]), '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

__slice_assign ::
               forall a t . (Tensor t, Fullfilled "__slice_assign" t a) =>
                 ArgsHMap "__slice_assign" t a -> TensorApply t
__slice_assign args
  = let scalarArgs
          = catMaybes
              [("begin",) . showValue <$> (args !? #begin :: Maybe [Int]),
               ("end",) . showValue <$> (args !? #end :: Maybe [Int]),
               ("step",) . showValue <$> (args !? #step :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "_slice_assign" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__slice_assign_scalar" t =
     '[ '("scalar", AttrOpt Double), '("begin", AttrReq [Int]),
        '("end", AttrReq [Int]), '("step", AttrOpt [Int]),
        '("data", AttrOpt t)]

__slice_assign_scalar ::
                      forall a t . (Tensor t, Fullfilled "__slice_assign_scalar" t a) =>
                        ArgsHMap "__slice_assign_scalar" t a -> TensorApply t
__slice_assign_scalar args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Double),
               ("begin",) . showValue <$> (args !? #begin :: Maybe [Int]),
               ("end",) . showValue <$> (args !? #end :: Maybe [Int]),
               ("step",) . showValue <$> (args !? #step :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_slice_assign_scalar" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__sparse_adagrad_update" t =
     '[ '("lr", AttrReq Float), '("epsilon", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt t),
        '("grad", AttrOpt t), '("history", AttrOpt t)]

__sparse_adagrad_update ::
                        forall a t .
                          (Tensor t, Fullfilled "__sparse_adagrad_update" t a) =>
                          ArgsHMap "__sparse_adagrad_update" t a -> TensorApply t
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
              [("weight",) <$> (args !? #weight :: Maybe t),
               ("grad",) <$> (args !? #grad :: Maybe t),
               ("history",) <$> (args !? #history :: Maybe t)]
      in apply "_sparse_adagrad_update" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__sparse_retain" t =
     '[ '("data", AttrOpt t), '("indices", AttrOpt t)]

__sparse_retain ::
                forall a t . (Tensor t, Fullfilled "__sparse_retain" t a) =>
                  ArgsHMap "__sparse_retain" t a -> TensorApply t
__sparse_retain args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("indices",) <$> (args !? #indices :: Maybe t)]
      in apply "_sparse_retain" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__split_v2" t =
     '[ '("indices", AttrReq [Int]), '("axis", AttrOpt Int),
        '("squeeze_axis", AttrOpt Bool), '("sections", AttrOpt Int),
        '("data", AttrOpt t)]

__split_v2 ::
           forall a t . (Tensor t, Fullfilled "__split_v2" t a) =>
             ArgsHMap "__split_v2" t a -> TensorApply t
__split_v2 args
  = let scalarArgs
          = catMaybes
              [("indices",) . showValue <$> (args !? #indices :: Maybe [Int]),
               ("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("squeeze_axis",) . showValue <$>
                 (args !? #squeeze_axis :: Maybe Bool),
               ("sections",) . showValue <$> (args !? #sections :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_split_v2" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__split_v2_backward" t = '[]

__split_v2_backward ::
                    forall a t . (Tensor t, Fullfilled "__split_v2_backward" t a) =>
                      ArgsHMap "__split_v2_backward" t a -> TensorApply t
__split_v2_backward args
  = let scalarArgs = catMaybes []
        tensorKeyArgs = catMaybes []
      in apply "_split_v2_backward" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__square_sum" t =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt t)]

__square_sum ::
             forall a t . (Tensor t, Fullfilled "__square_sum" t a) =>
               ArgsHMap "__square_sum" t a -> TensorApply t
__square_sum args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_square_sum" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__unravel_index" t =
     '[ '("shape", AttrOpt [Int]), '("data", AttrOpt t)]

__unravel_index ::
                forall a t . (Tensor t, Fullfilled "__unravel_index" t a) =>
                  ArgsHMap "__unravel_index" t a -> TensorApply t
__unravel_index args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "_unravel_index" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__zeros" t =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype",
          AttrOpt
            (EnumType
               '["bool", "float16", "float32", "float64", "int32", "int64",
                 "int8", "uint8"]))]

__zeros ::
        forall a t . (Tensor t, Fullfilled "__zeros" t a) =>
          ArgsHMap "__zeros" t a -> TensorApply t
__zeros args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["bool", "float16", "float32", "float64", "int32", "int64",
                           "int8", "uint8"]))]
        tensorKeyArgs = catMaybes []
      in apply "_zeros" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "__zeros_without_dtype" t =
     '[ '("shape", AttrOpt [Int]), '("ctx", AttrOpt Text),
        '("dtype", AttrOpt Int)]

__zeros_without_dtype ::
                      forall a t . (Tensor t, Fullfilled "__zeros_without_dtype" t a) =>
                        ArgsHMap "__zeros_without_dtype" t a -> TensorApply t
__zeros_without_dtype args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int]),
               ("ctx",) . showValue <$> (args !? #ctx :: Maybe Text),
               ("dtype",) . showValue <$> (args !? #dtype :: Maybe Int)]
        tensorKeyArgs = catMaybes []
      in apply "_zeros_without_dtype" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_abs" t = '[ '("data", AttrOpt t)]

_abs ::
     forall a t . (Tensor t, Fullfilled "_abs" t a) =>
       ArgsHMap "_abs" t a -> TensorApply t
_abs args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "abs" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_adam_update" t =
     '[ '("lr", AttrReq Float), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("lazy_update", AttrOpt Bool),
        '("weight", AttrOpt t), '("grad", AttrOpt t), '("mean", AttrOpt t),
        '("var", AttrOpt t)]

_adam_update ::
             forall a t . (Tensor t, Fullfilled "_adam_update" t a) =>
               ArgsHMap "_adam_update" t a -> TensorApply t
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
              [("weight",) <$> (args !? #weight :: Maybe t),
               ("grad",) <$> (args !? #grad :: Maybe t),
               ("mean",) <$> (args !? #mean :: Maybe t),
               ("var",) <$> (args !? #var :: Maybe t)]
      in apply "adam_update" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_add_n" t = '[ '("args", AttrOpt [t])]

_add_n ::
       forall a t . (Tensor t, Fullfilled "_add_n" t a) =>
         ArgsHMap "_add_n" t a -> TensorApply t
_add_n args
  = let scalarArgs
          = catMaybes [Just ("num_args", showValue (length tensorVarArgs))]
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #args :: Maybe [t])
      in apply "add_n" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_all_finite" t =
     '[ '("init_output", AttrOpt Bool), '("data", AttrOpt t)]

_all_finite ::
            forall a t . (Tensor t, Fullfilled "_all_finite" t a) =>
              ArgsHMap "_all_finite" t a -> TensorApply t
_all_finite args
  = let scalarArgs
          = catMaybes
              [("init_output",) . showValue <$>
                 (args !? #init_output :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "all_finite" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_amp_cast" t =
     '[ '("dtype",
          AttrReq
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"])),
        '("data", AttrOpt t)]

_amp_cast ::
          forall a t . (Tensor t, Fullfilled "_amp_cast" t a) =>
            ArgsHMap "_amp_cast" t a -> TensorApply t
_amp_cast args
  = let scalarArgs
          = catMaybes
              [("dtype",) . showValue <$>
                 (args !? #dtype ::
                    Maybe
                      (EnumType
                         '["float16", "float32", "float64", "int32", "int64", "int8",
                           "uint8"]))]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "amp_cast" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_amp_multicast" t =
     '[ '("num_outputs", AttrReq Int), '("cast_narrow", AttrOpt Bool),
        '("data", AttrOpt [t])]

_amp_multicast ::
               forall a t . (Tensor t, Fullfilled "_amp_multicast" t a) =>
                 ArgsHMap "_amp_multicast" t a -> TensorApply t
_amp_multicast args
  = let scalarArgs
          = catMaybes
              [("num_outputs",) . showValue <$>
                 (args !? #num_outputs :: Maybe Int),
               ("cast_narrow",) . showValue <$>
                 (args !? #cast_narrow :: Maybe Bool)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "amp_multicast" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_arccos" t = '[ '("data", AttrOpt t)]

_arccos ::
        forall a t . (Tensor t, Fullfilled "_arccos" t a) =>
          ArgsHMap "_arccos" t a -> TensorApply t
_arccos args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "arccos" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_arccosh" t = '[ '("data", AttrOpt t)]

_arccosh ::
         forall a t . (Tensor t, Fullfilled "_arccosh" t a) =>
           ArgsHMap "_arccosh" t a -> TensorApply t
_arccosh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "arccosh" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_arcsin" t = '[ '("data", AttrOpt t)]

_arcsin ::
        forall a t . (Tensor t, Fullfilled "_arcsin" t a) =>
          ArgsHMap "_arcsin" t a -> TensorApply t
_arcsin args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "arcsin" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_arcsinh" t = '[ '("data", AttrOpt t)]

_arcsinh ::
         forall a t . (Tensor t, Fullfilled "_arcsinh" t a) =>
           ArgsHMap "_arcsinh" t a -> TensorApply t
_arcsinh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "arcsinh" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_arctan" t = '[ '("data", AttrOpt t)]

_arctan ::
        forall a t . (Tensor t, Fullfilled "_arctan" t a) =>
          ArgsHMap "_arctan" t a -> TensorApply t
_arctan args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "arctan" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_arctanh" t = '[ '("data", AttrOpt t)]

_arctanh ::
         forall a t . (Tensor t, Fullfilled "_arctanh" t a) =>
           ArgsHMap "_arctanh" t a -> TensorApply t
_arctanh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "arctanh" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_argmax" t =
     '[ '("axis", AttrOpt (Maybe Int)), '("keepdims", AttrOpt Bool),
        '("data", AttrOpt t)]

_argmax ::
        forall a t . (Tensor t, Fullfilled "_argmax" t a) =>
          ArgsHMap "_argmax" t a -> TensorApply t
_argmax args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "argmax" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_argmax_channel" t =
     '[ '("data", AttrOpt t)]

_argmax_channel ::
                forall a t . (Tensor t, Fullfilled "_argmax_channel" t a) =>
                  ArgsHMap "_argmax_channel" t a -> TensorApply t
_argmax_channel args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "argmax_channel" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_argmin" t =
     '[ '("axis", AttrOpt (Maybe Int)), '("keepdims", AttrOpt Bool),
        '("data", AttrOpt t)]

_argmin ::
        forall a t . (Tensor t, Fullfilled "_argmin" t a) =>
          ArgsHMap "_argmin" t a -> TensorApply t
_argmin args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "argmin" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_argsort" t =
     '[ '("axis", AttrOpt (Maybe Int)), '("is_ascend", AttrOpt Bool),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "uint8"])),
        '("data", AttrOpt t)]

_argsort ::
         forall a t . (Tensor t, Fullfilled "_argsort" t a) =>
           ArgsHMap "_argsort" t a -> TensorApply t
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
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "argsort" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_batch_dot" t =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("forward_stype",
          AttrOpt (Maybe (EnumType '["csr", "default", "row_sparse"]))),
        '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_batch_dot ::
           forall a t . (Tensor t, Fullfilled "_batch_dot" t a) =>
             ArgsHMap "_batch_dot" t a -> TensorApply t
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
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "batch_dot" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_batch_take" t =
     '[ '("a", AttrOpt t), '("indices", AttrOpt t)]

_batch_take ::
            forall a t . (Tensor t, Fullfilled "_batch_take" t a) =>
              ArgsHMap "_batch_take" t a -> TensorApply t
_batch_take args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("a",) <$> (args !? #a :: Maybe t),
               ("indices",) <$> (args !? #indices :: Maybe t)]
      in apply "batch_take" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_broadcast_add" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_broadcast_add ::
               forall a t . (Tensor t, Fullfilled "_broadcast_add" t a) =>
                 ArgsHMap "_broadcast_add" t a -> TensorApply t
_broadcast_add args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "broadcast_add" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_broadcast_axis" t =
     '[ '("axis", AttrOpt [Int]), '("size", AttrOpt [Int]),
        '("data", AttrOpt t)]

_broadcast_axis ::
                forall a t . (Tensor t, Fullfilled "_broadcast_axis" t a) =>
                  ArgsHMap "_broadcast_axis" t a -> TensorApply t
_broadcast_axis args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int]),
               ("size",) . showValue <$> (args !? #size :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "broadcast_axis" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_broadcast_div" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_broadcast_div ::
               forall a t . (Tensor t, Fullfilled "_broadcast_div" t a) =>
                 ArgsHMap "_broadcast_div" t a -> TensorApply t
_broadcast_div args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "broadcast_div" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_broadcast_equal" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_broadcast_equal ::
                 forall a t . (Tensor t, Fullfilled "_broadcast_equal" t a) =>
                   ArgsHMap "_broadcast_equal" t a -> TensorApply t
_broadcast_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "broadcast_equal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_broadcast_greater" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_broadcast_greater ::
                   forall a t . (Tensor t, Fullfilled "_broadcast_greater" t a) =>
                     ArgsHMap "_broadcast_greater" t a -> TensorApply t
_broadcast_greater args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "broadcast_greater" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_broadcast_greater_equal" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_broadcast_greater_equal ::
                         forall a t .
                           (Tensor t, Fullfilled "_broadcast_greater_equal" t a) =>
                           ArgsHMap "_broadcast_greater_equal" t a -> TensorApply t
_broadcast_greater_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "broadcast_greater_equal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_broadcast_hypot" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_broadcast_hypot ::
                 forall a t . (Tensor t, Fullfilled "_broadcast_hypot" t a) =>
                   ArgsHMap "_broadcast_hypot" t a -> TensorApply t
_broadcast_hypot args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "broadcast_hypot" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_broadcast_lesser" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_broadcast_lesser ::
                  forall a t . (Tensor t, Fullfilled "_broadcast_lesser" t a) =>
                    ArgsHMap "_broadcast_lesser" t a -> TensorApply t
_broadcast_lesser args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "broadcast_lesser" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_broadcast_lesser_equal" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_broadcast_lesser_equal ::
                        forall a t .
                          (Tensor t, Fullfilled "_broadcast_lesser_equal" t a) =>
                          ArgsHMap "_broadcast_lesser_equal" t a -> TensorApply t
_broadcast_lesser_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "broadcast_lesser_equal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_broadcast_like" t =
     '[ '("lhs_axes", AttrOpt (Maybe [Int])),
        '("rhs_axes", AttrOpt (Maybe [Int])), '("lhs", AttrOpt t),
        '("rhs", AttrOpt t)]

_broadcast_like ::
                forall a t . (Tensor t, Fullfilled "_broadcast_like" t a) =>
                  ArgsHMap "_broadcast_like" t a -> TensorApply t
_broadcast_like args
  = let scalarArgs
          = catMaybes
              [("lhs_axes",) . showValue <$>
                 (args !? #lhs_axes :: Maybe (Maybe [Int])),
               ("rhs_axes",) . showValue <$>
                 (args !? #rhs_axes :: Maybe (Maybe [Int]))]
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "broadcast_like" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_broadcast_logical_and" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_broadcast_logical_and ::
                       forall a t . (Tensor t, Fullfilled "_broadcast_logical_and" t a) =>
                         ArgsHMap "_broadcast_logical_and" t a -> TensorApply t
_broadcast_logical_and args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "broadcast_logical_and" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_broadcast_logical_or" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_broadcast_logical_or ::
                      forall a t . (Tensor t, Fullfilled "_broadcast_logical_or" t a) =>
                        ArgsHMap "_broadcast_logical_or" t a -> TensorApply t
_broadcast_logical_or args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "broadcast_logical_or" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_broadcast_logical_xor" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_broadcast_logical_xor ::
                       forall a t . (Tensor t, Fullfilled "_broadcast_logical_xor" t a) =>
                         ArgsHMap "_broadcast_logical_xor" t a -> TensorApply t
_broadcast_logical_xor args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "broadcast_logical_xor" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_broadcast_maximum" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_broadcast_maximum ::
                   forall a t . (Tensor t, Fullfilled "_broadcast_maximum" t a) =>
                     ArgsHMap "_broadcast_maximum" t a -> TensorApply t
_broadcast_maximum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "broadcast_maximum" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_broadcast_minimum" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_broadcast_minimum ::
                   forall a t . (Tensor t, Fullfilled "_broadcast_minimum" t a) =>
                     ArgsHMap "_broadcast_minimum" t a -> TensorApply t
_broadcast_minimum args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "broadcast_minimum" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_broadcast_mod" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_broadcast_mod ::
               forall a t . (Tensor t, Fullfilled "_broadcast_mod" t a) =>
                 ArgsHMap "_broadcast_mod" t a -> TensorApply t
_broadcast_mod args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "broadcast_mod" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_broadcast_mul" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_broadcast_mul ::
               forall a t . (Tensor t, Fullfilled "_broadcast_mul" t a) =>
                 ArgsHMap "_broadcast_mul" t a -> TensorApply t
_broadcast_mul args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "broadcast_mul" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_broadcast_not_equal" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_broadcast_not_equal ::
                     forall a t . (Tensor t, Fullfilled "_broadcast_not_equal" t a) =>
                       ArgsHMap "_broadcast_not_equal" t a -> TensorApply t
_broadcast_not_equal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "broadcast_not_equal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_broadcast_power" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_broadcast_power ::
                 forall a t . (Tensor t, Fullfilled "_broadcast_power" t a) =>
                   ArgsHMap "_broadcast_power" t a -> TensorApply t
_broadcast_power args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "broadcast_power" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_broadcast_sub" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_broadcast_sub ::
               forall a t . (Tensor t, Fullfilled "_broadcast_sub" t a) =>
                 ArgsHMap "_broadcast_sub" t a -> TensorApply t
_broadcast_sub args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "broadcast_sub" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_broadcast_to" t =
     '[ '("shape", AttrOpt [Int]), '("data", AttrOpt t)]

_broadcast_to ::
              forall a t . (Tensor t, Fullfilled "_broadcast_to" t a) =>
                ArgsHMap "_broadcast_to" t a -> TensorApply t
_broadcast_to args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "broadcast_to" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_cast_storage" t =
     '[ '("stype",
          AttrReq (EnumType '["csr", "default", "row_sparse"])),
        '("data", AttrOpt t)]

_cast_storage ::
              forall a t . (Tensor t, Fullfilled "_cast_storage" t a) =>
                ArgsHMap "_cast_storage" t a -> TensorApply t
_cast_storage args
  = let scalarArgs
          = catMaybes
              [("stype",) . showValue <$>
                 (args !? #stype ::
                    Maybe (EnumType '["csr", "default", "row_sparse"]))]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "cast_storage" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_cbrt" t = '[ '("data", AttrOpt t)]

_cbrt ::
      forall a t . (Tensor t, Fullfilled "_cbrt" t a) =>
        ArgsHMap "_cbrt" t a -> TensorApply t
_cbrt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "cbrt" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_ceil" t = '[ '("data", AttrOpt t)]

_ceil ::
      forall a t . (Tensor t, Fullfilled "_ceil" t a) =>
        ArgsHMap "_ceil" t a -> TensorApply t
_ceil args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "ceil" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_clip" t =
     '[ '("a_min", AttrReq Float), '("a_max", AttrReq Float),
        '("data", AttrOpt t)]

_clip ::
      forall a t . (Tensor t, Fullfilled "_clip" t a) =>
        ArgsHMap "_clip" t a -> TensorApply t
_clip args
  = let scalarArgs
          = catMaybes
              [("a_min",) . showValue <$> (args !? #a_min :: Maybe Float),
               ("a_max",) . showValue <$> (args !? #a_max :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "clip" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_cos" t = '[ '("data", AttrOpt t)]

_cos ::
     forall a t . (Tensor t, Fullfilled "_cos" t a) =>
       ArgsHMap "_cos" t a -> TensorApply t
_cos args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "cos" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_cosh" t = '[ '("data", AttrOpt t)]

_cosh ::
      forall a t . (Tensor t, Fullfilled "_cosh" t a) =>
        ArgsHMap "_cosh" t a -> TensorApply t
_cosh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "cosh" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_degrees" t = '[ '("data", AttrOpt t)]

_degrees ::
         forall a t . (Tensor t, Fullfilled "_degrees" t a) =>
           ArgsHMap "_degrees" t a -> TensorApply t
_degrees args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "degrees" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_depth_to_space" t =
     '[ '("block_size", AttrReq Int), '("data", AttrOpt t)]

_depth_to_space ::
                forall a t . (Tensor t, Fullfilled "_depth_to_space" t a) =>
                  ArgsHMap "_depth_to_space" t a -> TensorApply t
_depth_to_space args
  = let scalarArgs
          = catMaybes
              [("block_size",) . showValue <$>
                 (args !? #block_size :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "depth_to_space" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_diag" t =
     '[ '("k", AttrOpt Int), '("axis1", AttrOpt Int),
        '("axis2", AttrOpt Int), '("data", AttrOpt t)]

_diag ::
      forall a t . (Tensor t, Fullfilled "_diag" t a) =>
        ArgsHMap "_diag" t a -> TensorApply t
_diag args
  = let scalarArgs
          = catMaybes
              [("k",) . showValue <$> (args !? #k :: Maybe Int),
               ("axis1",) . showValue <$> (args !? #axis1 :: Maybe Int),
               ("axis2",) . showValue <$> (args !? #axis2 :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "diag" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_dot" t =
     '[ '("transpose_a", AttrOpt Bool), '("transpose_b", AttrOpt Bool),
        '("forward_stype",
          AttrOpt (Maybe (EnumType '["csr", "default", "row_sparse"]))),
        '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_dot ::
     forall a t . (Tensor t, Fullfilled "_dot" t a) =>
       ArgsHMap "_dot" t a -> TensorApply t
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
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "dot" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_elemwise_add" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_elemwise_add ::
              forall a t . (Tensor t, Fullfilled "_elemwise_add" t a) =>
                ArgsHMap "_elemwise_add" t a -> TensorApply t
_elemwise_add args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "elemwise_add" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_elemwise_div" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_elemwise_div ::
              forall a t . (Tensor t, Fullfilled "_elemwise_div" t a) =>
                ArgsHMap "_elemwise_div" t a -> TensorApply t
_elemwise_div args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "elemwise_div" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_elemwise_mul" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_elemwise_mul ::
              forall a t . (Tensor t, Fullfilled "_elemwise_mul" t a) =>
                ArgsHMap "_elemwise_mul" t a -> TensorApply t
_elemwise_mul args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "elemwise_mul" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_elemwise_sub" t =
     '[ '("lhs", AttrOpt t), '("rhs", AttrOpt t)]

_elemwise_sub ::
              forall a t . (Tensor t, Fullfilled "_elemwise_sub" t a) =>
                ArgsHMap "_elemwise_sub" t a -> TensorApply t
_elemwise_sub args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "elemwise_sub" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_erf" t = '[ '("data", AttrOpt t)]

_erf ::
     forall a t . (Tensor t, Fullfilled "_erf" t a) =>
       ArgsHMap "_erf" t a -> TensorApply t
_erf args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "erf" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_erfinv" t = '[ '("data", AttrOpt t)]

_erfinv ::
        forall a t . (Tensor t, Fullfilled "_erfinv" t a) =>
          ArgsHMap "_erfinv" t a -> TensorApply t
_erfinv args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "erfinv" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_exp" t = '[ '("data", AttrOpt t)]

_exp ::
     forall a t . (Tensor t, Fullfilled "_exp" t a) =>
       ArgsHMap "_exp" t a -> TensorApply t
_exp args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "exp" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_expand_dims" t =
     '[ '("axis", AttrReq Int), '("data", AttrOpt t)]

_expand_dims ::
             forall a t . (Tensor t, Fullfilled "_expand_dims" t a) =>
               ArgsHMap "_expand_dims" t a -> TensorApply t
_expand_dims args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "expand_dims" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_expm1" t = '[ '("data", AttrOpt t)]

_expm1 ::
       forall a t . (Tensor t, Fullfilled "_expm1" t a) =>
         ArgsHMap "_expm1" t a -> TensorApply t
_expm1 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "expm1" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_fill_element_0index" t =
     '[ '("lhs", AttrOpt t), '("mhs", AttrOpt t), '("rhs", AttrOpt t)]

_fill_element_0index ::
                     forall a t . (Tensor t, Fullfilled "_fill_element_0index" t a) =>
                       ArgsHMap "_fill_element_0index" t a -> TensorApply t
_fill_element_0index args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("mhs",) <$> (args !? #mhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "fill_element_0index" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_fix" t = '[ '("data", AttrOpt t)]

_fix ::
     forall a t . (Tensor t, Fullfilled "_fix" t a) =>
       ArgsHMap "_fix" t a -> TensorApply t
_fix args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "fix" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_floor" t = '[ '("data", AttrOpt t)]

_floor ::
       forall a t . (Tensor t, Fullfilled "_floor" t a) =>
         ArgsHMap "_floor" t a -> TensorApply t
_floor args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "floor" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_ftml_update" t =
     '[ '("lr", AttrReq Float), '("beta1", AttrOpt Float),
        '("beta2", AttrOpt Float), '("epsilon", AttrOpt Double),
        '("t", AttrReq Int), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float), '("clip_grad", AttrOpt Float),
        '("weight", AttrOpt t), '("grad", AttrOpt t), '("d", AttrOpt t),
        '("v", AttrOpt t), '("z", AttrOpt t)]

_ftml_update ::
             forall a t . (Tensor t, Fullfilled "_ftml_update" t a) =>
               ArgsHMap "_ftml_update" t a -> TensorApply t
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
              [("weight",) <$> (args !? #weight :: Maybe t),
               ("grad",) <$> (args !? #grad :: Maybe t),
               ("d",) <$> (args !? #d :: Maybe t),
               ("v",) <$> (args !? #v :: Maybe t),
               ("z",) <$> (args !? #z :: Maybe t)]
      in apply "ftml_update" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_ftrl_update" t =
     '[ '("lr", AttrReq Float), '("lamda1", AttrOpt Float),
        '("beta", AttrOpt Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt t),
        '("grad", AttrOpt t), '("z", AttrOpt t), '("n", AttrOpt t)]

_ftrl_update ::
             forall a t . (Tensor t, Fullfilled "_ftrl_update" t a) =>
               ArgsHMap "_ftrl_update" t a -> TensorApply t
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
              [("weight",) <$> (args !? #weight :: Maybe t),
               ("grad",) <$> (args !? #grad :: Maybe t),
               ("z",) <$> (args !? #z :: Maybe t),
               ("n",) <$> (args !? #n :: Maybe t)]
      in apply "ftrl_update" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_gamma" t = '[ '("data", AttrOpt t)]

_gamma ::
       forall a t . (Tensor t, Fullfilled "_gamma" t a) =>
         ArgsHMap "_gamma" t a -> TensorApply t
_gamma args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "gamma" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_gammaln" t = '[ '("data", AttrOpt t)]

_gammaln ::
         forall a t . (Tensor t, Fullfilled "_gammaln" t a) =>
           ArgsHMap "_gammaln" t a -> TensorApply t
_gammaln args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "gammaln" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_gather_nd" t =
     '[ '("data", AttrOpt t), '("indices", AttrOpt t)]

_gather_nd ::
           forall a t . (Tensor t, Fullfilled "_gather_nd" t a) =>
             ArgsHMap "_gather_nd" t a -> TensorApply t
_gather_nd args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("indices",) <$> (args !? #indices :: Maybe t)]
      in apply "gather_nd" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_hard_sigmoid" t =
     '[ '("alpha", AttrOpt Float), '("beta", AttrOpt Float),
        '("data", AttrOpt t)]

_hard_sigmoid ::
              forall a t . (Tensor t, Fullfilled "_hard_sigmoid" t a) =>
                ArgsHMap "_hard_sigmoid" t a -> TensorApply t
_hard_sigmoid args
  = let scalarArgs
          = catMaybes
              [("alpha",) . showValue <$> (args !? #alpha :: Maybe Float),
               ("beta",) . showValue <$> (args !? #beta :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "hard_sigmoid" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_khatri_rao" t =
     '[ '("args", AttrOpt [t])]

_khatri_rao ::
            forall a t . (Tensor t, Fullfilled "_khatri_rao" t a) =>
              ArgsHMap "_khatri_rao" t a -> TensorApply t
_khatri_rao args
  = let scalarArgs
          = catMaybes [Just ("num_args", showValue (length tensorVarArgs))]
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #args :: Maybe [t])
      in apply "khatri_rao" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_lamb_update_phase1" t =
     '[ '("beta1", AttrOpt Float), '("beta2", AttrOpt Float),
        '("epsilon", AttrOpt Float), '("t", AttrReq Int),
        '("bias_correction", AttrOpt Bool), '("wd", AttrReq Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt t),
        '("grad", AttrOpt t), '("mean", AttrOpt t), '("var", AttrOpt t)]

_lamb_update_phase1 ::
                    forall a t . (Tensor t, Fullfilled "_lamb_update_phase1" t a) =>
                      ArgsHMap "_lamb_update_phase1" t a -> TensorApply t
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
              [("weight",) <$> (args !? #weight :: Maybe t),
               ("grad",) <$> (args !? #grad :: Maybe t),
               ("mean",) <$> (args !? #mean :: Maybe t),
               ("var",) <$> (args !? #var :: Maybe t)]
      in apply "lamb_update_phase1" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_lamb_update_phase2" t =
     '[ '("lr", AttrReq Float), '("lower_bound", AttrOpt Float),
        '("upper_bound", AttrOpt Float), '("weight", AttrOpt t),
        '("g", AttrOpt t), '("r1", AttrOpt t), '("r2", AttrOpt t)]

_lamb_update_phase2 ::
                    forall a t . (Tensor t, Fullfilled "_lamb_update_phase2" t a) =>
                      ArgsHMap "_lamb_update_phase2" t a -> TensorApply t
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
              [("weight",) <$> (args !? #weight :: Maybe t),
               ("g",) <$> (args !? #g :: Maybe t),
               ("r1",) <$> (args !? #r1 :: Maybe t),
               ("r2",) <$> (args !? #r2 :: Maybe t)]
      in apply "lamb_update_phase2" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_log" t = '[ '("data", AttrOpt t)]

_log ::
     forall a t . (Tensor t, Fullfilled "_log" t a) =>
       ArgsHMap "_log" t a -> TensorApply t
_log args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "log" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_log10" t = '[ '("data", AttrOpt t)]

_log10 ::
       forall a t . (Tensor t, Fullfilled "_log10" t a) =>
         ArgsHMap "_log10" t a -> TensorApply t
_log10 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "log10" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_log1p" t = '[ '("data", AttrOpt t)]

_log1p ::
       forall a t . (Tensor t, Fullfilled "_log1p" t a) =>
         ArgsHMap "_log1p" t a -> TensorApply t
_log1p args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "log1p" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_log2" t = '[ '("data", AttrOpt t)]

_log2 ::
      forall a t . (Tensor t, Fullfilled "_log2" t a) =>
        ArgsHMap "_log2" t a -> TensorApply t
_log2 args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "log2" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_log_softmax" t =
     '[ '("axis", AttrOpt Int),
        '("temperature", AttrOpt (Maybe Double)),
        '("dtype",
          AttrOpt (Maybe (EnumType '["float16", "float32", "float64"]))),
        '("use_length", AttrOpt (Maybe Bool)), '("data", AttrOpt t)]

_log_softmax ::
             forall a t . (Tensor t, Fullfilled "_log_softmax" t a) =>
               ArgsHMap "_log_softmax" t a -> TensorApply t
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
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "log_softmax" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_logical_not" t =
     '[ '("data", AttrOpt t)]

_logical_not ::
             forall a t . (Tensor t, Fullfilled "_logical_not" t a) =>
               ArgsHMap "_logical_not" t a -> TensorApply t
_logical_not args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "logical_not" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_make_loss" t =
     '[ '("data", AttrOpt t)]

_make_loss ::
           forall a t . (Tensor t, Fullfilled "_make_loss" t a) =>
             ArgsHMap "_make_loss" t a -> TensorApply t
_make_loss args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "make_loss" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_max" t =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt t)]

_max ::
     forall a t . (Tensor t, Fullfilled "_max" t a) =>
       ArgsHMap "_max" t a -> TensorApply t
_max args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "max" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_mean" t =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt t)]

_mean ::
      forall a t . (Tensor t, Fullfilled "_mean" t a) =>
        ArgsHMap "_mean" t a -> TensorApply t
_mean args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "mean" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_min" t =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt t)]

_min ::
     forall a t . (Tensor t, Fullfilled "_min" t a) =>
       ArgsHMap "_min" t a -> TensorApply t
_min args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "min" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_moments" t =
     '[ '("axes", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("data", AttrOpt t)]

_moments ::
         forall a t . (Tensor t, Fullfilled "_moments" t a) =>
           ArgsHMap "_moments" t a -> TensorApply t
_moments args
  = let scalarArgs
          = catMaybes
              [("axes",) . showValue <$> (args !? #axes :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "moments" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_mp_lamb_update_phase1" t =
     '[ '("beta1", AttrOpt Float), '("beta2", AttrOpt Float),
        '("epsilon", AttrOpt Float), '("t", AttrReq Int),
        '("bias_correction", AttrOpt Bool), '("wd", AttrReq Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt t),
        '("grad", AttrOpt t), '("mean", AttrOpt t), '("var", AttrOpt t),
        '("weight32", AttrOpt t)]

_mp_lamb_update_phase1 ::
                       forall a t . (Tensor t, Fullfilled "_mp_lamb_update_phase1" t a) =>
                         ArgsHMap "_mp_lamb_update_phase1" t a -> TensorApply t
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
              [("weight",) <$> (args !? #weight :: Maybe t),
               ("grad",) <$> (args !? #grad :: Maybe t),
               ("mean",) <$> (args !? #mean :: Maybe t),
               ("var",) <$> (args !? #var :: Maybe t),
               ("weight32",) <$> (args !? #weight32 :: Maybe t)]
      in apply "mp_lamb_update_phase1" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_mp_lamb_update_phase2" t =
     '[ '("lr", AttrReq Float), '("lower_bound", AttrOpt Float),
        '("upper_bound", AttrOpt Float), '("weight", AttrOpt t),
        '("g", AttrOpt t), '("r1", AttrOpt t), '("r2", AttrOpt t),
        '("weight32", AttrOpt t)]

_mp_lamb_update_phase2 ::
                       forall a t . (Tensor t, Fullfilled "_mp_lamb_update_phase2" t a) =>
                         ArgsHMap "_mp_lamb_update_phase2" t a -> TensorApply t
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
              [("weight",) <$> (args !? #weight :: Maybe t),
               ("g",) <$> (args !? #g :: Maybe t),
               ("r1",) <$> (args !? #r1 :: Maybe t),
               ("r2",) <$> (args !? #r2 :: Maybe t),
               ("weight32",) <$> (args !? #weight32 :: Maybe t)]
      in apply "mp_lamb_update_phase2" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_mp_nag_mom_update" t =
     '[ '("lr", AttrReq Float), '("momentum", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt t),
        '("grad", AttrOpt t), '("mom", AttrOpt t),
        '("weight32", AttrOpt t)]

_mp_nag_mom_update ::
                   forall a t . (Tensor t, Fullfilled "_mp_nag_mom_update" t a) =>
                     ArgsHMap "_mp_nag_mom_update" t a -> TensorApply t
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
              [("weight",) <$> (args !? #weight :: Maybe t),
               ("grad",) <$> (args !? #grad :: Maybe t),
               ("mom",) <$> (args !? #mom :: Maybe t),
               ("weight32",) <$> (args !? #weight32 :: Maybe t)]
      in apply "mp_nag_mom_update" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_mp_sgd_mom_update" t =
     '[ '("lr", AttrReq Float), '("momentum", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("lazy_update", AttrOpt Bool),
        '("weight", AttrOpt t), '("grad", AttrOpt t), '("mom", AttrOpt t),
        '("weight32", AttrOpt t)]

_mp_sgd_mom_update ::
                   forall a t . (Tensor t, Fullfilled "_mp_sgd_mom_update" t a) =>
                     ArgsHMap "_mp_sgd_mom_update" t a -> TensorApply t
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
              [("weight",) <$> (args !? #weight :: Maybe t),
               ("grad",) <$> (args !? #grad :: Maybe t),
               ("mom",) <$> (args !? #mom :: Maybe t),
               ("weight32",) <$> (args !? #weight32 :: Maybe t)]
      in apply "mp_sgd_mom_update" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_mp_sgd_update" t =
     '[ '("lr", AttrReq Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("lazy_update", AttrOpt Bool),
        '("weight", AttrOpt t), '("grad", AttrOpt t),
        '("weight32", AttrOpt t)]

_mp_sgd_update ::
               forall a t . (Tensor t, Fullfilled "_mp_sgd_update" t a) =>
                 ArgsHMap "_mp_sgd_update" t a -> TensorApply t
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
              [("weight",) <$> (args !? #weight :: Maybe t),
               ("grad",) <$> (args !? #grad :: Maybe t),
               ("weight32",) <$> (args !? #weight32 :: Maybe t)]
      in apply "mp_sgd_update" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_multi_all_finite" t =
     '[ '("num_arrays", AttrOpt Int), '("init_output", AttrOpt Bool),
        '("data", AttrOpt [t])]

_multi_all_finite ::
                  forall a t . (Tensor t, Fullfilled "_multi_all_finite" t a) =>
                    ArgsHMap "_multi_all_finite" t a -> TensorApply t
_multi_all_finite args
  = let scalarArgs
          = catMaybes
              [("num_arrays",) . showValue <$>
                 (args !? #num_arrays :: Maybe Int),
               ("init_output",) . showValue <$>
                 (args !? #init_output :: Maybe Bool)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "multi_all_finite" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_multi_lars" t =
     '[ '("eta", AttrReq Float), '("eps", AttrReq Float),
        '("rescale_grad", AttrOpt Float), '("lrs", AttrOpt t),
        '("weights_sum_sq", AttrOpt t), '("grads_sum_sq", AttrOpt t),
        '("wds", AttrOpt t)]

_multi_lars ::
            forall a t . (Tensor t, Fullfilled "_multi_lars" t a) =>
              ArgsHMap "_multi_lars" t a -> TensorApply t
_multi_lars args
  = let scalarArgs
          = catMaybes
              [("eta",) . showValue <$> (args !? #eta :: Maybe Float),
               ("eps",) . showValue <$> (args !? #eps :: Maybe Float),
               ("rescale_grad",) . showValue <$>
                 (args !? #rescale_grad :: Maybe Float)]
        tensorKeyArgs
          = catMaybes
              [("lrs",) <$> (args !? #lrs :: Maybe t),
               ("weights_sum_sq",) <$> (args !? #weights_sum_sq :: Maybe t),
               ("grads_sum_sq",) <$> (args !? #grads_sum_sq :: Maybe t),
               ("wds",) <$> (args !? #wds :: Maybe t)]
      in apply "multi_lars" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_multi_mp_sgd_mom_update" t =
     '[ '("lrs", AttrReq [Float]), '("wds", AttrReq [Float]),
        '("momentum", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("data", AttrOpt [t])]

_multi_mp_sgd_mom_update ::
                         forall a t .
                           (Tensor t, Fullfilled "_multi_mp_sgd_mom_update" t a) =>
                           ArgsHMap "_multi_mp_sgd_mom_update" t a -> TensorApply t
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
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "multi_mp_sgd_mom_update" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_multi_mp_sgd_update" t =
     '[ '("lrs", AttrReq [Float]), '("wds", AttrReq [Float]),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("data", AttrOpt [t])]

_multi_mp_sgd_update ::
                     forall a t . (Tensor t, Fullfilled "_multi_mp_sgd_update" t a) =>
                       ArgsHMap "_multi_mp_sgd_update" t a -> TensorApply t
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
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "multi_mp_sgd_update" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_multi_sgd_mom_update" t =
     '[ '("lrs", AttrReq [Float]), '("wds", AttrReq [Float]),
        '("momentum", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("data", AttrOpt [t])]

_multi_sgd_mom_update ::
                      forall a t . (Tensor t, Fullfilled "_multi_sgd_mom_update" t a) =>
                        ArgsHMap "_multi_sgd_mom_update" t a -> TensorApply t
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
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "multi_sgd_mom_update" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_multi_sgd_update" t =
     '[ '("lrs", AttrReq [Float]), '("wds", AttrReq [Float]),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("data", AttrOpt [t])]

_multi_sgd_update ::
                  forall a t . (Tensor t, Fullfilled "_multi_sgd_update" t a) =>
                    ArgsHMap "_multi_sgd_update" t a -> TensorApply t
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
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "multi_sgd_update" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_multi_sum_sq" t =
     '[ '("num_arrays", AttrReq Int), '("data", AttrOpt [t])]

_multi_sum_sq ::
              forall a t . (Tensor t, Fullfilled "_multi_sum_sq" t a) =>
                ArgsHMap "_multi_sum_sq" t a -> TensorApply t
_multi_sum_sq args
  = let scalarArgs
          = catMaybes
              [("num_arrays",) . showValue <$>
                 (args !? #num_arrays :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "multi_sum_sq" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_nag_mom_update" t =
     '[ '("lr", AttrReq Float), '("momentum", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt t),
        '("grad", AttrOpt t), '("mom", AttrOpt t)]

_nag_mom_update ::
                forall a t . (Tensor t, Fullfilled "_nag_mom_update" t a) =>
                  ArgsHMap "_nag_mom_update" t a -> TensorApply t
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
              [("weight",) <$> (args !? #weight :: Maybe t),
               ("grad",) <$> (args !? #grad :: Maybe t),
               ("mom",) <$> (args !? #mom :: Maybe t)]
      in apply "nag_mom_update" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_nanprod" t =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt t)]

_nanprod ::
         forall a t . (Tensor t, Fullfilled "_nanprod" t a) =>
           ArgsHMap "_nanprod" t a -> TensorApply t
_nanprod args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "nanprod" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_nansum" t =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt t)]

_nansum ::
        forall a t . (Tensor t, Fullfilled "_nansum" t a) =>
          ArgsHMap "_nansum" t a -> TensorApply t
_nansum args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "nansum" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_negative" t =
     '[ '("data", AttrOpt t)]

_negative ::
          forall a t . (Tensor t, Fullfilled "_negative" t a) =>
            ArgsHMap "_negative" t a -> TensorApply t
_negative args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "negative" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_norm" t =
     '[ '("ord", AttrOpt Int), '("axis", AttrOpt (Maybe [Int])),
        '("out_dtype",
          AttrOpt
            (Maybe
               (EnumType
                  '["float16", "float32", "float64", "int32", "int64", "int8"]))),
        '("keepdims", AttrOpt Bool), '("data", AttrOpt t)]

_norm ::
      forall a t . (Tensor t, Fullfilled "_norm" t a) =>
        ArgsHMap "_norm" t a -> TensorApply t
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
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "norm" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_one_hot" t =
     '[ '("depth", AttrReq Int), '("on_value", AttrOpt Double),
        '("off_value", AttrOpt Double),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "int8",
                 "uint8"])),
        '("indices", AttrOpt t)]

_one_hot ::
         forall a t . (Tensor t, Fullfilled "_one_hot" t a) =>
           ArgsHMap "_one_hot" t a -> TensorApply t
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
                         '["float16", "float32", "float64", "int32", "int64", "int8",
                           "uint8"]))]
        tensorKeyArgs
          = catMaybes [("indices",) <$> (args !? #indices :: Maybe t)]
      in apply "one_hot" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_ones_like" t =
     '[ '("data", AttrOpt t)]

_ones_like ::
           forall a t . (Tensor t, Fullfilled "_ones_like" t a) =>
             ArgsHMap "_ones_like" t a -> TensorApply t
_ones_like args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "ones_like" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_pick" t =
     '[ '("axis", AttrOpt (Maybe Int)), '("keepdims", AttrOpt Bool),
        '("mode", AttrOpt (EnumType '["clip", "wrap"])),
        '("data", AttrOpt t), '("index", AttrOpt t)]

_pick ::
      forall a t . (Tensor t, Fullfilled "_pick" t a) =>
        ArgsHMap "_pick" t a -> TensorApply t
_pick args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("mode",) . showValue <$>
                 (args !? #mode :: Maybe (EnumType '["clip", "wrap"]))]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("index",) <$> (args !? #index :: Maybe t)]
      in apply "pick" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_preloaded_multi_mp_sgd_mom_update" t
     =
     '[ '("momentum", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("data", AttrOpt [t])]

_preloaded_multi_mp_sgd_mom_update ::
                                   forall a t .
                                     (Tensor t,
                                      Fullfilled "_preloaded_multi_mp_sgd_mom_update" t a) =>
                                     ArgsHMap "_preloaded_multi_mp_sgd_mom_update" t a ->
                                       TensorApply t
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
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in
      apply "preloaded_multi_mp_sgd_mom_update" scalarArgs
        (Right tensorVarArgs)

type instance ParameterList "_preloaded_multi_mp_sgd_update" t =
     '[ '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("data", AttrOpt [t])]

_preloaded_multi_mp_sgd_update ::
                               forall a t .
                                 (Tensor t, Fullfilled "_preloaded_multi_mp_sgd_update" t a) =>
                                 ArgsHMap "_preloaded_multi_mp_sgd_update" t a -> TensorApply t
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
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in
      apply "preloaded_multi_mp_sgd_update" scalarArgs
        (Right tensorVarArgs)

type instance ParameterList "_preloaded_multi_sgd_mom_update" t =
     '[ '("momentum", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("data", AttrOpt [t])]

_preloaded_multi_sgd_mom_update ::
                                forall a t .
                                  (Tensor t, Fullfilled "_preloaded_multi_sgd_mom_update" t a) =>
                                  ArgsHMap "_preloaded_multi_sgd_mom_update" t a -> TensorApply t
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
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in
      apply "preloaded_multi_sgd_mom_update" scalarArgs
        (Right tensorVarArgs)

type instance ParameterList "_preloaded_multi_sgd_update" t =
     '[ '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("num_weights", AttrOpt Int),
        '("data", AttrOpt [t])]

_preloaded_multi_sgd_update ::
                            forall a t .
                              (Tensor t, Fullfilled "_preloaded_multi_sgd_update" t a) =>
                              ArgsHMap "_preloaded_multi_sgd_update" t a -> TensorApply t
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
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in
      apply "preloaded_multi_sgd_update" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_prod" t =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt t)]

_prod ::
      forall a t . (Tensor t, Fullfilled "_prod" t a) =>
        ArgsHMap "_prod" t a -> TensorApply t
_prod args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "prod" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_radians" t = '[ '("data", AttrOpt t)]

_radians ::
         forall a t . (Tensor t, Fullfilled "_radians" t a) =>
           ArgsHMap "_radians" t a -> TensorApply t
_radians args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "radians" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_rcbrt" t = '[ '("data", AttrOpt t)]

_rcbrt ::
       forall a t . (Tensor t, Fullfilled "_rcbrt" t a) =>
         ArgsHMap "_rcbrt" t a -> TensorApply t
_rcbrt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "rcbrt" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_reciprocal" t =
     '[ '("data", AttrOpt t)]

_reciprocal ::
            forall a t . (Tensor t, Fullfilled "_reciprocal" t a) =>
              ArgsHMap "_reciprocal" t a -> TensorApply t
_reciprocal args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "reciprocal" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_relu" t = '[ '("data", AttrOpt t)]

_relu ::
      forall a t . (Tensor t, Fullfilled "_relu" t a) =>
        ArgsHMap "_relu" t a -> TensorApply t
_relu args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "relu" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_repeat" t =
     '[ '("repeats", AttrReq Int), '("axis", AttrOpt (Maybe Int)),
        '("data", AttrOpt t)]

_repeat ::
        forall a t . (Tensor t, Fullfilled "_repeat" t a) =>
          ArgsHMap "_repeat" t a -> TensorApply t
_repeat args
  = let scalarArgs
          = catMaybes
              [("repeats",) . showValue <$> (args !? #repeats :: Maybe Int),
               ("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int))]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "repeat" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_reset_arrays" t =
     '[ '("num_arrays", AttrReq Int), '("data", AttrOpt [t])]

_reset_arrays ::
              forall a t . (Tensor t, Fullfilled "_reset_arrays" t a) =>
                ArgsHMap "_reset_arrays" t a -> TensorApply t
_reset_arrays args
  = let scalarArgs
          = catMaybes
              [("num_arrays",) . showValue <$>
                 (args !? #num_arrays :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "reset_arrays" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_reshape_like" t =
     '[ '("lhs_begin", AttrOpt (Maybe Int)),
        '("lhs_end", AttrOpt (Maybe Int)),
        '("rhs_begin", AttrOpt (Maybe Int)),
        '("rhs_end", AttrOpt (Maybe Int)), '("lhs", AttrOpt t),
        '("rhs", AttrOpt t)]

_reshape_like ::
              forall a t . (Tensor t, Fullfilled "_reshape_like" t a) =>
                ArgsHMap "_reshape_like" t a -> TensorApply t
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
              [("lhs",) <$> (args !? #lhs :: Maybe t),
               ("rhs",) <$> (args !? #rhs :: Maybe t)]
      in apply "reshape_like" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_reverse" t =
     '[ '("axis", AttrReq [Int]), '("data", AttrOpt t)]

_reverse ::
         forall a t . (Tensor t, Fullfilled "_reverse" t a) =>
           ArgsHMap "_reverse" t a -> TensorApply t
_reverse args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "reverse" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_rint" t = '[ '("data", AttrOpt t)]

_rint ::
      forall a t . (Tensor t, Fullfilled "_rint" t a) =>
        ArgsHMap "_rint" t a -> TensorApply t
_rint args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "rint" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_rmsprop_update" t =
     '[ '("lr", AttrReq Float), '("gamma1", AttrOpt Float),
        '("epsilon", AttrOpt Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("clip_weights", AttrOpt Float), '("weight", AttrOpt t),
        '("grad", AttrOpt t), '("n", AttrOpt t)]

_rmsprop_update ::
                forall a t . (Tensor t, Fullfilled "_rmsprop_update" t a) =>
                  ArgsHMap "_rmsprop_update" t a -> TensorApply t
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
              [("weight",) <$> (args !? #weight :: Maybe t),
               ("grad",) <$> (args !? #grad :: Maybe t),
               ("n",) <$> (args !? #n :: Maybe t)]
      in apply "rmsprop_update" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_rmspropalex_update" t =
     '[ '("lr", AttrReq Float), '("gamma1", AttrOpt Float),
        '("gamma2", AttrOpt Float), '("epsilon", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float),
        '("clip_weights", AttrOpt Float), '("weight", AttrOpt t),
        '("grad", AttrOpt t), '("n", AttrOpt t), '("g", AttrOpt t),
        '("delta", AttrOpt t)]

_rmspropalex_update ::
                    forall a t . (Tensor t, Fullfilled "_rmspropalex_update" t a) =>
                      ArgsHMap "_rmspropalex_update" t a -> TensorApply t
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
              [("weight",) <$> (args !? #weight :: Maybe t),
               ("grad",) <$> (args !? #grad :: Maybe t),
               ("n",) <$> (args !? #n :: Maybe t),
               ("g",) <$> (args !? #g :: Maybe t),
               ("delta",) <$> (args !? #delta :: Maybe t)]
      in apply "rmspropalex_update" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_round" t = '[ '("data", AttrOpt t)]

_round ::
       forall a t . (Tensor t, Fullfilled "_round" t a) =>
         ArgsHMap "_round" t a -> TensorApply t
_round args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "round" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_rsqrt" t = '[ '("data", AttrOpt t)]

_rsqrt ::
       forall a t . (Tensor t, Fullfilled "_rsqrt" t a) =>
         ArgsHMap "_rsqrt" t a -> TensorApply t
_rsqrt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "rsqrt" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_scatter_nd" t =
     '[ '("shape", AttrReq [Int]), '("data", AttrOpt t),
        '("indices", AttrOpt t)]

_scatter_nd ::
            forall a t . (Tensor t, Fullfilled "_scatter_nd" t a) =>
              ArgsHMap "_scatter_nd" t a -> TensorApply t
_scatter_nd args
  = let scalarArgs
          = catMaybes
              [("shape",) . showValue <$> (args !? #shape :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("indices",) <$> (args !? #indices :: Maybe t)]
      in apply "scatter_nd" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_sgd_mom_update" t =
     '[ '("lr", AttrReq Float), '("momentum", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("lazy_update", AttrOpt Bool),
        '("weight", AttrOpt t), '("grad", AttrOpt t), '("mom", AttrOpt t)]

_sgd_mom_update ::
                forall a t . (Tensor t, Fullfilled "_sgd_mom_update" t a) =>
                  ArgsHMap "_sgd_mom_update" t a -> TensorApply t
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
              [("weight",) <$> (args !? #weight :: Maybe t),
               ("grad",) <$> (args !? #grad :: Maybe t),
               ("mom",) <$> (args !? #mom :: Maybe t)]
      in apply "sgd_mom_update" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_sgd_update" t =
     '[ '("lr", AttrReq Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("lazy_update", AttrOpt Bool),
        '("weight", AttrOpt t), '("grad", AttrOpt t)]

_sgd_update ::
            forall a t . (Tensor t, Fullfilled "_sgd_update" t a) =>
              ArgsHMap "_sgd_update" t a -> TensorApply t
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
              [("weight",) <$> (args !? #weight :: Maybe t),
               ("grad",) <$> (args !? #grad :: Maybe t)]
      in apply "sgd_update" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_shape_array" t =
     '[ '("data", AttrOpt t)]

_shape_array ::
             forall a t . (Tensor t, Fullfilled "_shape_array" t a) =>
               ArgsHMap "_shape_array" t a -> TensorApply t
_shape_array args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "shape_array" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_sigmoid" t = '[ '("data", AttrOpt t)]

_sigmoid ::
         forall a t . (Tensor t, Fullfilled "_sigmoid" t a) =>
           ArgsHMap "_sigmoid" t a -> TensorApply t
_sigmoid args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "sigmoid" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_sign" t = '[ '("data", AttrOpt t)]

_sign ::
      forall a t . (Tensor t, Fullfilled "_sign" t a) =>
        ArgsHMap "_sign" t a -> TensorApply t
_sign args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "sign" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_signsgd_update" t =
     '[ '("lr", AttrReq Float), '("wd", AttrOpt Float),
        '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("weight", AttrOpt t),
        '("grad", AttrOpt t)]

_signsgd_update ::
                forall a t . (Tensor t, Fullfilled "_signsgd_update" t a) =>
                  ArgsHMap "_signsgd_update" t a -> TensorApply t
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
              [("weight",) <$> (args !? #weight :: Maybe t),
               ("grad",) <$> (args !? #grad :: Maybe t)]
      in apply "signsgd_update" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_signum_update" t =
     '[ '("lr", AttrReq Float), '("momentum", AttrOpt Float),
        '("wd", AttrOpt Float), '("rescale_grad", AttrOpt Float),
        '("clip_gradient", AttrOpt Float), '("wd_lh", AttrOpt Float),
        '("weight", AttrOpt t), '("grad", AttrOpt t), '("mom", AttrOpt t)]

_signum_update ::
               forall a t . (Tensor t, Fullfilled "_signum_update" t a) =>
                 ArgsHMap "_signum_update" t a -> TensorApply t
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
              [("weight",) <$> (args !? #weight :: Maybe t),
               ("grad",) <$> (args !? #grad :: Maybe t),
               ("mom",) <$> (args !? #mom :: Maybe t)]
      in apply "signum_update" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_sin" t = '[ '("data", AttrOpt t)]

_sin ::
     forall a t . (Tensor t, Fullfilled "_sin" t a) =>
       ArgsHMap "_sin" t a -> TensorApply t
_sin args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "sin" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_sinh" t = '[ '("data", AttrOpt t)]

_sinh ::
      forall a t . (Tensor t, Fullfilled "_sinh" t a) =>
        ArgsHMap "_sinh" t a -> TensorApply t
_sinh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "sinh" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_size_array" t =
     '[ '("data", AttrOpt t)]

_size_array ::
            forall a t . (Tensor t, Fullfilled "_size_array" t a) =>
              ArgsHMap "_size_array" t a -> TensorApply t
_size_array args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "size_array" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_slice" t =
     '[ '("begin", AttrReq [Int]), '("end", AttrReq [Int]),
        '("step", AttrOpt [Int]), '("data", AttrOpt t)]

_slice ::
       forall a t . (Tensor t, Fullfilled "_slice" t a) =>
         ArgsHMap "_slice" t a -> TensorApply t
_slice args
  = let scalarArgs
          = catMaybes
              [("begin",) . showValue <$> (args !? #begin :: Maybe [Int]),
               ("end",) . showValue <$> (args !? #end :: Maybe [Int]),
               ("step",) . showValue <$> (args !? #step :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "slice" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_slice_axis" t =
     '[ '("axis", AttrReq Int), '("begin", AttrReq Int),
        '("end", AttrReq (Maybe Int)), '("data", AttrOpt t)]

_slice_axis ::
            forall a t . (Tensor t, Fullfilled "_slice_axis" t a) =>
              ArgsHMap "_slice_axis" t a -> TensorApply t
_slice_axis args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("begin",) . showValue <$> (args !? #begin :: Maybe Int),
               ("end",) . showValue <$> (args !? #end :: Maybe (Maybe Int))]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "slice_axis" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_slice_like" t =
     '[ '("axes", AttrOpt [Int]), '("data", AttrOpt t),
        '("shape_like", AttrOpt t)]

_slice_like ::
            forall a t . (Tensor t, Fullfilled "_slice_like" t a) =>
              ArgsHMap "_slice_like" t a -> TensorApply t
_slice_like args
  = let scalarArgs
          = catMaybes
              [("axes",) . showValue <$> (args !? #axes :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("shape_like",) <$> (args !? #shape_like :: Maybe t)]
      in apply "slice_like" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_smooth_l1" t =
     '[ '("scalar", AttrOpt Float), '("data", AttrOpt t)]

_smooth_l1 ::
           forall a t . (Tensor t, Fullfilled "_smooth_l1" t a) =>
             ArgsHMap "_smooth_l1" t a -> TensorApply t
_smooth_l1 args
  = let scalarArgs
          = catMaybes
              [("scalar",) . showValue <$> (args !? #scalar :: Maybe Float)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "smooth_l1" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_softmax" t =
     '[ '("axis", AttrOpt Int),
        '("temperature", AttrOpt (Maybe Double)),
        '("dtype",
          AttrOpt (Maybe (EnumType '["float16", "float32", "float64"]))),
        '("use_length", AttrOpt (Maybe Bool)), '("data", AttrOpt t),
        '("length", AttrOpt t)]

_softmax ::
         forall a t . (Tensor t, Fullfilled "_softmax" t a) =>
           ArgsHMap "_softmax" t a -> TensorApply t
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
              [("data",) <$> (args !? #data :: Maybe t),
               ("length",) <$> (args !? #length :: Maybe t)]
      in apply "softmax" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_softmax_cross_entropy" t =
     '[ '("data", AttrOpt t), '("label", AttrOpt t)]

_softmax_cross_entropy ::
                       forall a t . (Tensor t, Fullfilled "_softmax_cross_entropy" t a) =>
                         ArgsHMap "_softmax_cross_entropy" t a -> TensorApply t
_softmax_cross_entropy args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("data",) <$> (args !? #data :: Maybe t),
               ("label",) <$> (args !? #label :: Maybe t)]
      in apply "softmax_cross_entropy" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_softmin" t =
     '[ '("axis", AttrOpt Int),
        '("temperature", AttrOpt (Maybe Double)),
        '("dtype",
          AttrOpt (Maybe (EnumType '["float16", "float32", "float64"]))),
        '("use_length", AttrOpt (Maybe Bool)), '("data", AttrOpt t)]

_softmin ::
         forall a t . (Tensor t, Fullfilled "_softmin" t a) =>
           ArgsHMap "_softmin" t a -> TensorApply t
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
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "softmin" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_softsign" t =
     '[ '("data", AttrOpt t)]

_softsign ::
          forall a t . (Tensor t, Fullfilled "_softsign" t a) =>
            ArgsHMap "_softsign" t a -> TensorApply t
_softsign args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "softsign" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_sort" t =
     '[ '("axis", AttrOpt (Maybe Int)), '("is_ascend", AttrOpt Bool),
        '("data", AttrOpt t)]

_sort ::
      forall a t . (Tensor t, Fullfilled "_sort" t a) =>
        ArgsHMap "_sort" t a -> TensorApply t
_sort args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe Int)),
               ("is_ascend",) . showValue <$> (args !? #is_ascend :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "sort" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_space_to_depth" t =
     '[ '("block_size", AttrReq Int), '("data", AttrOpt t)]

_space_to_depth ::
                forall a t . (Tensor t, Fullfilled "_space_to_depth" t a) =>
                  ArgsHMap "_space_to_depth" t a -> TensorApply t
_space_to_depth args
  = let scalarArgs
          = catMaybes
              [("block_size",) . showValue <$>
                 (args !? #block_size :: Maybe Int)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "space_to_depth" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_sqrt" t = '[ '("data", AttrOpt t)]

_sqrt ::
      forall a t . (Tensor t, Fullfilled "_sqrt" t a) =>
        ArgsHMap "_sqrt" t a -> TensorApply t
_sqrt args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "sqrt" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_square" t = '[ '("data", AttrOpt t)]

_square ::
        forall a t . (Tensor t, Fullfilled "_square" t a) =>
          ArgsHMap "_square" t a -> TensorApply t
_square args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "square" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_squeeze" t =
     '[ '("axis", AttrOpt (Maybe [Int])), '("data", AttrOpt t)]

_squeeze ::
         forall a t . (Tensor t, Fullfilled "_squeeze" t a) =>
           ArgsHMap "_squeeze" t a -> TensorApply t
_squeeze args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int]))]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "squeeze" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_stack" t =
     '[ '("axis", AttrOpt Int), '("num_args", AttrReq Int),
        '("data", AttrOpt [t])]

_stack ::
       forall a t . (Tensor t, Fullfilled "_stack" t a) =>
         ArgsHMap "_stack" t a -> TensorApply t
_stack args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("num_args",) . showValue <$> (args !? #num_args :: Maybe Int)]
        tensorKeyArgs = catMaybes []
        tensorVarArgs = fromMaybe [] (args !? #data :: Maybe [t])
      in apply "stack" scalarArgs (Right tensorVarArgs)

type instance ParameterList "_sum" t =
     '[ '("axis", AttrOpt (Maybe [Int])), '("keepdims", AttrOpt Bool),
        '("exclude", AttrOpt Bool), '("data", AttrOpt t)]

_sum ::
     forall a t . (Tensor t, Fullfilled "_sum" t a) =>
       ArgsHMap "_sum" t a -> TensorApply t
_sum args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe (Maybe [Int])),
               ("keepdims",) . showValue <$> (args !? #keepdims :: Maybe Bool),
               ("exclude",) . showValue <$> (args !? #exclude :: Maybe Bool)]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "sum" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_take" t =
     '[ '("axis", AttrOpt Int),
        '("mode", AttrOpt (EnumType '["clip", "raise", "wrap"])),
        '("a", AttrOpt t), '("indices", AttrOpt t)]

_take ::
      forall a t . (Tensor t, Fullfilled "_take" t a) =>
        ArgsHMap "_take" t a -> TensorApply t
_take args
  = let scalarArgs
          = catMaybes
              [("axis",) . showValue <$> (args !? #axis :: Maybe Int),
               ("mode",) . showValue <$>
                 (args !? #mode :: Maybe (EnumType '["clip", "raise", "wrap"]))]
        tensorKeyArgs
          = catMaybes
              [("a",) <$> (args !? #a :: Maybe t),
               ("indices",) <$> (args !? #indices :: Maybe t)]
      in apply "take" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_tan" t = '[ '("data", AttrOpt t)]

_tan ::
     forall a t . (Tensor t, Fullfilled "_tan" t a) =>
       ArgsHMap "_tan" t a -> TensorApply t
_tan args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "tan" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_tanh" t = '[ '("data", AttrOpt t)]

_tanh ::
      forall a t . (Tensor t, Fullfilled "_tanh" t a) =>
        ArgsHMap "_tanh" t a -> TensorApply t
_tanh args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "tanh" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_tile" t =
     '[ '("reps", AttrReq [Int]), '("data", AttrOpt t)]

_tile ::
      forall a t . (Tensor t, Fullfilled "_tile" t a) =>
        ArgsHMap "_tile" t a -> TensorApply t
_tile args
  = let scalarArgs
          = catMaybes
              [("reps",) . showValue <$> (args !? #reps :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "tile" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_topk" t =
     '[ '("axis", AttrOpt (Maybe Int)), '("k", AttrOpt Int),
        '("ret_typ",
          AttrOpt (EnumType '["both", "indices", "mask", "value"])),
        '("is_ascend", AttrOpt Bool),
        '("dtype",
          AttrOpt
            (EnumType
               '["float16", "float32", "float64", "int32", "int64", "uint8"])),
        '("data", AttrOpt t)]

_topk ::
      forall a t . (Tensor t, Fullfilled "_topk" t a) =>
        ArgsHMap "_topk" t a -> TensorApply t
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
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "topk" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_transpose" t =
     '[ '("axes", AttrOpt [Int]), '("data", AttrOpt t)]

_transpose ::
           forall a t . (Tensor t, Fullfilled "_transpose" t a) =>
             ArgsHMap "_transpose" t a -> TensorApply t
_transpose args
  = let scalarArgs
          = catMaybes
              [("axes",) . showValue <$> (args !? #axes :: Maybe [Int])]
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "transpose" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_trunc" t = '[ '("data", AttrOpt t)]

_trunc ::
       forall a t . (Tensor t, Fullfilled "_trunc" t a) =>
         ArgsHMap "_trunc" t a -> TensorApply t
_trunc args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "trunc" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_where" t =
     '[ '("condition", AttrOpt t), '("x", AttrOpt t), '("y", AttrOpt t)]

_where ::
       forall a t . (Tensor t, Fullfilled "_where" t a) =>
         ArgsHMap "_where" t a -> TensorApply t
_where args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes
              [("condition",) <$> (args !? #condition :: Maybe t),
               ("x",) <$> (args !? #x :: Maybe t),
               ("y",) <$> (args !? #y :: Maybe t)]
      in apply "where" scalarArgs (Left tensorKeyArgs)

type instance ParameterList "_zeros_like" t =
     '[ '("data", AttrOpt t)]

_zeros_like ::
            forall a t . (Tensor t, Fullfilled "_zeros_like" t a) =>
              ArgsHMap "_zeros_like" t a -> TensorApply t
_zeros_like args
  = let scalarArgs = catMaybes []
        tensorKeyArgs
          = catMaybes [("data",) <$> (args !? #data :: Maybe t)]
      in apply "zeros_like" scalarArgs (Left tensorKeyArgs)